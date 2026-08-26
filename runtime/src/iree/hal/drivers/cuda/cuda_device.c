// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/cuda_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/event_pool.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/cuda/cuda_allocator.h"
#include "iree/hal/drivers/cuda/cuda_buffer.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/cuda_status_util.h"
#include "iree/hal/drivers/cuda/event_pool.h"
#include "iree/hal/drivers/cuda/event_semaphore.h"
#include "iree/hal/drivers/cuda/graph_command_buffer.h"
#include "iree/hal/drivers/cuda/memory_pools.h"
#include "iree/hal/drivers/cuda/nccl_channel.h"
#include "iree/hal/drivers/cuda/nccl_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/nop_executable_cache.h"
#include "iree/hal/drivers/cuda/stream_command_buffer.h"
#include "iree/hal/drivers/cuda/timepoint_pool.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/deferred_work_queue.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"
#include "iree/hal/utils/stream_tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_cuda_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the CUDA symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  const iree_hal_cuda_dynamic_symbols_t* cuda_symbols;
  const iree_hal_cuda_nccl_dynamic_symbols_t* nccl_symbols;

  // Parameters used to control device behavior.
  iree_hal_cuda_device_params_t params;

  CUcontext cu_context;
  CUdevice cu_device;

  // One CUstream per queue_affinity bit. Up to IREE_HAL_CUDA_QUEUE_COUNT
  // distinct streams; queue_execute/queue_alloca/etc. pick among them by
  // ctz(queue_affinity). On a single physical GPU CUDA can interleave
  // kernels from different streams subject to SM availability, so this lets
  // q0 and q1 work run concurrently when they fit. Backward-compatible
  // single-stream code paths use streams[0] (i.e., queue 0).
#define IREE_HAL_CUDA_QUEUE_COUNT 2
  CUstream dispatch_cu_streams[IREE_HAL_CUDA_QUEUE_COUNT];

  // Dedicated stream used to bridge foreign (non-CUDA) semaphore waits into
  // CUDA events. cuLaunchHostFunc on this stream performs the blocking wait
  // on a local-task semaphore; a subsequent cuEventRecord produces a CUDA
  // event that the dispatch stream can wait on via cuStreamWaitEvent. This
  // keeps the VM submit thread unblocked while honoring cross-device deps.
  CUstream foreign_bridge_cu_stream;

  // One tracing context per dispatch stream so kernels launched on
  // streams[q] are visible in Tracy under their own GPU zone. Otherwise a
  // single tracing context would only record events for streams[0].
  iree_hal_stream_tracing_context_t* tracing_contexts[IREE_HAL_CUDA_QUEUE_COUNT];
  // Legacy alias — external consumers and some existing call sites expect a
  // single tracing_context handle. Kept pointing at tracing_contexts[0].
  iree_hal_stream_tracing_context_t* tracing_context;

  iree_allocator_t host_allocator;

  // Host/device event pools, used for backing semaphore timepoints.
  iree_event_pool_t* host_event_pool;
  iree_hal_cuda_event_pool_t* device_event_pool;
  // Timepoint pools, shared by various semaphores.
  iree_hal_cuda_timepoint_pool_t* timepoint_pool;

  // One deferred work queue per dispatch stream. Each DWQ orders workloads
  // for its stream and releases to the GPU when constraints are met. They
  // buffer submissions and allocations internally before they are ready.
  // This queue couples with HAL semaphores backed by iree_event_t and
  // CUevent objects. For backward compatibility, paths that don't route by
  // queue_affinity (e.g., signal-driven issue from a semaphore) issue on
  // all DWQs so downstream actions wake up regardless of which queue their
  // producing submission targeted.
  iree_hal_deferred_work_queue_t* work_queues[IREE_HAL_CUDA_QUEUE_COUNT];

  // Device memory pools and allocators.
  bool supports_memory_pools;
  iree_hal_cuda_memory_pools_t memory_pools;
  iree_hal_allocator_t* device_allocator;

  // Integrated GPU (Tegra/Jetson). CPU and GPU share physical memory.
  // On these systems we route transient (stream.resource.alloca) allocations
  // through the sync allocator so the result is host-mappable, enabling
  // cross-device consumption (CPU dispatch reading a GPU-produced buffer).
  bool is_integrated;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  iree_hal_device_topology_info_t topology_info;
} iree_hal_cuda_device_t;

static const iree_hal_device_vtable_t iree_hal_cuda_device_vtable;
static const iree_hal_deferred_work_queue_device_interface_vtable_t
    iree_hal_cuda_deferred_work_queue_device_interface_vtable;

// We put a CUEvent into a iree_hal_deferred_work_queue_native_event_t.
static_assert(sizeof(CUevent) <=
                  sizeof(iree_hal_deferred_work_queue_native_event_t),
              "Unexpected event size");
typedef struct iree_hal_cuda_deferred_work_queue_device_interface_t {
  iree_hal_deferred_work_queue_device_interface_t base;
  iree_hal_device_t* device;
  CUdevice cu_device;
  CUcontext cu_context;
  CUstream dispatch_cu_stream;
  // Tracing context paired with dispatch_cu_stream; stream command buffers
  // created by this DWQ record events into it so Tracy shows per-queue
  // GPU zones.
  iree_hal_stream_tracing_context_t* tracing_context;
  iree_allocator_t host_allocator;
  const iree_hal_cuda_dynamic_symbols_t* cuda_symbols;
} iree_hal_cuda_deferred_work_queue_device_interface_t;

static void iree_hal_cuda_deferred_work_queue_device_interface_destroy(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  iree_allocator_free(device_interface->host_allocator, device_interface);
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_bind_to_thread(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_CURESULT_TO_STATUS(device_interface->cuda_symbols,
                                 cuCtxSetCurrent(device_interface->cu_context),
                                 "cuCtxSetCurrent");
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_wait_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t event) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_CURESULT_TO_STATUS(
      device_interface->cuda_symbols,
      cuStreamWaitEvent(device_interface->dispatch_cu_stream, (CUevent)event,
                        CU_EVENT_WAIT_DEFAULT),
      "cuStreamWaitEvent");
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_create_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t* out_event) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_CURESULT_TO_STATUS(
      device_interface->cuda_symbols,
      cuEventCreate((CUevent*)out_event, CU_EVENT_DEFAULT), "cuEventCreate");
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_record_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t event) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_CURESULT_TO_STATUS(
      device_interface->cuda_symbols,
      cuEventRecord((CUevent)event, device_interface->dispatch_cu_stream),
      "cuEventCreate");
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_synchronize_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t event) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_CURESULT_TO_STATUS(device_interface->cuda_symbols,
                                 cuEventSynchronize((CUevent)event));
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_destroy_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t event) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_CURESULT_TO_STATUS(device_interface->cuda_symbols,
                                 cuEventDestroy((CUevent)event));
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_semaphore_acquire_timepoint_device_signal_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    struct iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_hal_deferred_work_queue_native_event_t* out_event) {
  // Foreign semaphores (e.g. from local-task) can't participate in device
  // signaling — they don't have CUevents. Return NULL and let the host
  // callback (semaphore_list_signal) handle them.
  if (!iree_hal_cuda_semaphore_isa(semaphore)) {
    *out_event = NULL;
    return iree_ok_status();
  }
  return iree_hal_cuda_event_semaphore_acquire_timepoint_device_signal(
      semaphore, value, (CUevent*)out_event);
}

static bool
iree_hal_cuda_deferred_work_queue_device_interface_acquire_host_wait_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    struct iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_hal_deferred_work_queue_host_device_event_t* out_event) {
  return iree_hal_cuda_semaphore_acquire_event_host_wait(
      semaphore, value, (iree_hal_cuda_event_t**)out_event);
}

static void
iree_hal_cuda_deferred_work_queue_device_interface_release_wait_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_host_device_event_t wait_event) {
  iree_hal_cuda_event_release(wait_event);
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_device_wait_on_host_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_host_device_event_t wait_event) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_CURESULT_TO_STATUS(
      device_interface->cuda_symbols,
      cuStreamWaitEvent(
          device_interface->dispatch_cu_stream,
          iree_hal_cuda_event_handle((iree_hal_cuda_event_t*)wait_event), 0),
      "cuStreamWaitEvent");
}

static void*
iree_hal_cuda_deferred_work_queue_device_interface_native_event_from_wait_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_host_device_event_t event) {
  return iree_hal_cuda_event_handle((iree_hal_cuda_event_t*)event);
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_create_stream_command_buffer(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_hal_command_buffer_t** out) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  // Build the stream command buffer on THIS DWQ's dispatch stream. Going
  // through iree_hal_cuda_device_create_stream_command_buffer would hardcode
  // streams[0], collapsing per-queue_affinity routing onto one CUstream.
  iree_hal_cuda_device_t* device =
      (iree_hal_cuda_device_t*)device_interface->device;
  return iree_hal_cuda_stream_command_buffer_create(
      iree_hal_device_allocator(device_interface->device), device->cuda_symbols,
      device->nccl_symbols, device_interface->tracing_context, mode, categories,
      /*binding_capacity=*/0, device_interface->dispatch_cu_stream,
      &device->block_pool, device->host_allocator, out);
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_submit_command_buffer(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_command_buffer_t* command_buffer) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(base_device_interface);
  iree_status_t status = iree_ok_status();
  if (iree_hal_cuda_stream_command_buffer_isa(command_buffer)) {
    // Stream command buffer so nothing to do but notify it was submitted.
    iree_hal_cuda_stream_notify_submitted_commands(command_buffer);
  } else {
    CUgraphExec exec =
        iree_hal_cuda_graph_command_buffer_handle(command_buffer);
    status = IREE_CURESULT_TO_STATUS(
        device_interface->cuda_symbols,
        cuGraphLaunch(exec, device_interface->dispatch_cu_stream));
    if (IREE_LIKELY(iree_status_is_ok(status))) {
      iree_hal_cuda_graph_tracing_notify_submitted_commands(command_buffer);
    }
  }
  return status;
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_async_alloc(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_buffer_t* buffer) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(
          base_device_interface);
  // Access the parent device directly (the device_interface->device was set
  // during device creation and points to our iree_hal_cuda_device_t).
  iree_hal_cuda_device_t* device =
      (iree_hal_cuda_device_t*)device_interface->device;

  iree_device_size_t allocation_size =
      iree_hal_buffer_allocation_size(buffer);

  // Select memory pool based on buffer memory type.
  CUmemoryPool memory_pool =
      iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
          ? device->memory_pools.device_local
          : device->memory_pools.other;

  // Perform the async allocation on the dispatch stream.
  CUdeviceptr device_ptr = 0;
  IREE_RETURN_IF_ERROR(IREE_CURESULT_TO_STATUS(
      device_interface->cuda_symbols,
      cuMemAllocFromPoolAsync(&device_ptr, (size_t)allocation_size, memory_pool,
                              device_interface->dispatch_cu_stream),
      "cuMemAllocFromPoolAsync"));

  // Set the device pointer on the pre-created buffer.
  iree_hal_cuda_buffer_set_device_pointer(buffer, device_ptr);

  // Update pool statistics.
  iree_hal_cuda_memory_pool_track_alloc(&device->memory_pools, buffer);

  return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_deferred_work_queue_device_interface_async_dealloc(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_buffer_t* buffer) {
  iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_cuda_deferred_work_queue_device_interface_t*)(
          base_device_interface);
  // Access the parent device directly.
  iree_hal_cuda_device_t* device =
      (iree_hal_cuda_device_t*)device_interface->device;

  // Only process async pool buffers.
  if (iree_hal_cuda_buffer_type(buffer) == IREE_HAL_CUDA_BUFFER_TYPE_ASYNC) {
    CUdeviceptr device_ptr = iree_hal_cuda_buffer_device_pointer(buffer);
    IREE_RETURN_IF_ERROR(IREE_CURESULT_TO_STATUS(
        device_interface->cuda_symbols,
        cuMemFreeAsync(device_ptr, device_interface->dispatch_cu_stream),
        "cuMemFreeAsync"));

    // Drop the release callback to prevent double-free.
    iree_hal_cuda_buffer_drop_release_callback(buffer);

    // Update pool statistics.
    iree_hal_cuda_memory_pool_track_free(&device->memory_pools, buffer);
  }

  return iree_ok_status();
}

typedef struct iree_hal_cuda_tracing_device_interface_t {
  iree_hal_stream_tracing_device_interface_t base;
  CUdevice cu_device;
  CUcontext cu_context;
  CUstream dispatch_cu_stream;
  iree_allocator_t host_allocator;
  const iree_hal_cuda_dynamic_symbols_t* cuda_symbols;
} iree_hal_cuda_tracing_device_interface_t;
static const iree_hal_stream_tracing_device_interface_vtable_t
    iree_hal_cuda_tracing_device_interface_vtable_t;

void iree_hal_cuda_tracing_device_interface_destroy(
    iree_hal_stream_tracing_device_interface_t* base_device_interface) {
  iree_hal_cuda_tracing_device_interface_t* device_interface =
      (iree_hal_cuda_tracing_device_interface_t*)base_device_interface;

  iree_allocator_free(device_interface->host_allocator, device_interface);
}

iree_status_t iree_hal_cuda_tracing_device_interface_synchronize_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_cuda_tracing_device_interface_t* device_interface =
      (iree_hal_cuda_tracing_device_interface_t*)base_device_interface;

  return IREE_CURESULT_TO_STATUS(device_interface->cuda_symbols,
                                 cuEventSynchronize((CUevent)base_event));
}

iree_status_t iree_hal_cuda_tracing_device_interface_create_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t* base_event) {
  iree_hal_cuda_tracing_device_interface_t* device_interface =
      (iree_hal_cuda_tracing_device_interface_t*)base_device_interface;

  return IREE_CURESULT_TO_STATUS(
      device_interface->cuda_symbols,
      cuEventCreate((CUevent*)base_event, CU_EVENT_DEFAULT));
}

iree_status_t iree_hal_cuda_tracing_device_interface_query_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_cuda_tracing_device_interface_t* device_interface =
      (iree_hal_cuda_tracing_device_interface_t*)base_device_interface;

  return IREE_CURESULT_TO_STATUS(device_interface->cuda_symbols,
                                 cuEventQuery((CUevent)base_event));
}

void iree_hal_cuda_tracing_device_interface_event_elapsed_time(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    float* relative_millis, iree_hal_stream_tracing_native_event_t start_event,
    iree_hal_stream_tracing_native_event_t end_event) {
  iree_hal_cuda_tracing_device_interface_t* device_interface =
      (iree_hal_cuda_tracing_device_interface_t*)base_device_interface;

  IREE_CUDA_IGNORE_ERROR(
      device_interface->cuda_symbols,
      cuEventElapsedTime(relative_millis, (CUevent)start_event,
                         (CUevent)end_event));
}

void iree_hal_cuda_tracing_device_interface_destroy_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_cuda_tracing_device_interface_t* device_interface =
      (iree_hal_cuda_tracing_device_interface_t*)base_device_interface;

  IREE_CUDA_IGNORE_ERROR(device_interface->cuda_symbols,
                         cuEventDestroy((CUevent)base_event));
}

iree_status_t iree_hal_cuda_tracing_device_interface_record_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_cuda_tracing_device_interface_t* device_interface =
      (iree_hal_cuda_tracing_device_interface_t*)base_device_interface;

  return IREE_CURESULT_TO_STATUS(
      device_interface->cuda_symbols,
      cuEventRecord((CUevent)base_event, device_interface->dispatch_cu_stream));
}

iree_status_t
iree_hal_cuda_tracing_device_interface_add_graph_event_record_node(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_graph_node_t* out_node,
    iree_hal_stream_tracing_native_graph_t graph,
    iree_hal_stream_tracing_native_graph_node_t* dependency_nodes,
    size_t dependency_nodes_count,
    iree_hal_stream_tracing_native_event_t event) {
  iree_hal_cuda_tracing_device_interface_t* device_interface =
      (iree_hal_cuda_tracing_device_interface_t*)base_device_interface;

  return IREE_CURESULT_TO_STATUS(
      device_interface->cuda_symbols,
      cuGraphAddEventRecordNode((CUgraphNode*)out_node, (CUgraph)graph,
                                (CUgraphNode*)dependency_nodes,
                                dependency_nodes_count, (CUevent)event));
}

static iree_hal_cuda_device_t* iree_hal_cuda_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_device_vtable);
  return (iree_hal_cuda_device_t*)base_value;
}

static iree_hal_cuda_device_t* iree_hal_cuda_device_cast_unsafe(
    iree_hal_device_t* base_value) {
  return (iree_hal_cuda_device_t*)base_value;
}

// Maps an iree_hal_queue_affinity_t bitmask to a CUDA stream / DWQ index.
// Convention: the lowest set bit selects the queue (0-indexed, clamped to
// IREE_HAL_CUDA_QUEUE_COUNT - 1). IREE_HAL_QUEUE_AFFINITY_ANY (all bits)
// resolves to queue 0. This matches the stream-affinity intent of
// hal.device.promise<@gpu, [N]> in the source MLIR.
static int iree_hal_cuda_device_select_queue_index(
    iree_hal_queue_affinity_t queue_affinity) {
  if (queue_affinity == 0) return 0;
  // __builtin_ctzll returns the bit position of the lowest set bit.
  int idx = (int)__builtin_ctzll((unsigned long long)queue_affinity);
  if (idx >= IREE_HAL_CUDA_QUEUE_COUNT) idx = IREE_HAL_CUDA_QUEUE_COUNT - 1;
  return idx;
}

IREE_API_EXPORT void iree_hal_cuda_device_params_initialize(
    iree_hal_cuda_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->event_pool_capacity = 32;
  out_params->queue_count = 1;
  out_params->command_buffer_mode = IREE_HAL_CUDA_COMMAND_BUFFER_MODE_GRAPH;
  out_params->stream_tracing = 0;
  out_params->async_allocations = true;
  // Keep freed pool memory around instead of returning it to the OS on every
  // cuMemFreeAsync. Without this, release_threshold defaults to 0 and each
  // per-iter alloc has to regrow the pool = ~4-6ms on Jetson. UINT64_MAX means
  // "never release" — trim must be explicit.
  out_params->memory_pools.device_local.release_threshold = UINT64_MAX;
  out_params->memory_pools.other.release_threshold = UINT64_MAX;
}

static iree_status_t iree_hal_cuda_device_check_params(
    const iree_hal_cuda_device_params_t* params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  if (params->queue_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one queue is required");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_cuda_device_params_t* params, CUdevice cu_device,
    CUstream dispatch_streams[IREE_HAL_CUDA_QUEUE_COUNT], CUcontext context,
    const iree_hal_cuda_dynamic_symbols_t* cuda_symbols,
    const iree_hal_cuda_nccl_dynamic_symbols_t* nccl_symbols,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_cuda_device_t* device = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_cuda_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + iree_sizeof_struct(*device));
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  device->cuda_symbols = cuda_symbols;
  device->nccl_symbols = nccl_symbols;
  device->params = *params;
  device->cu_context = context;
  device->cu_device = cu_device;
  for (int q = 0; q < IREE_HAL_CUDA_QUEUE_COUNT; ++q) {
    device->dispatch_cu_streams[q] = dispatch_streams[q];
  }
  device->host_allocator = host_allocator;

  iree_status_t status = iree_ok_status();

  // Allocate per-queue tracing contexts BEFORE creating DWQs so each DWQ's
  // device_interface can point at the matching tracing context. Without
  // this, stream[q>0] kernels run with no tracing attached and stay
  // invisible in Tracy.
  if (device->params.stream_tracing) {
    if (device->params.stream_tracing >=
            IREE_HAL_STREAM_TRACING_VERBOSITY_MAX ||
        device->params.stream_tracing < IREE_HAL_STREAM_TRACING_VERBOSITY_OFF) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid stream_tracing argument: expected to be between %d and %d",
          IREE_HAL_STREAM_TRACING_VERBOSITY_OFF,
          IREE_HAL_STREAM_TRACING_VERBOSITY_MAX);
    }

    // One tracing context per dispatch stream. Each gets its own device
    // interface so Tracy/IREE's stream tracing can label GPU zones per
    // queue_affinity bit. Without this, kernels on streams[q>0] run with no
    // tracing events attached and are invisible in Tracy.
    for (int q = 0; q < IREE_HAL_CUDA_QUEUE_COUNT; ++q) {
      iree_hal_cuda_tracing_device_interface_t* tracing_device_interface =
          NULL;
      status = iree_allocator_malloc(
          host_allocator, sizeof(iree_hal_cuda_tracing_device_interface_t),
          (void**)&tracing_device_interface);
      if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
        iree_hal_device_release((iree_hal_device_t*)device);
        return status;
      }

      tracing_device_interface->base.vtable =
          &iree_hal_cuda_tracing_device_interface_vtable_t;
      tracing_device_interface->cu_context = context;
      tracing_device_interface->cu_device = cu_device;
      tracing_device_interface->dispatch_cu_stream = dispatch_streams[q];
      tracing_device_interface->host_allocator = host_allocator;
      tracing_device_interface->cuda_symbols = cuda_symbols;

      status = iree_hal_stream_tracing_context_allocate(
          (iree_hal_stream_tracing_device_interface_t*)tracing_device_interface,
          device->identifier, device->params.stream_tracing,
          &device->block_pool, host_allocator, &device->tracing_contexts[q]);
      if (!iree_status_is_ok(status)) break;
    }
    // Keep the legacy single-handle field pointing at queue 0 for call sites
    // that don't thread queue_affinity through (e.g., graph command buffer
    // creation).
    device->tracing_context = device->tracing_contexts[0];
  }

  // Create one DWQ + device_interface per dispatch stream so each queue can
  // be issued/completed independently on its own CUDA stream. Must come
  // AFTER tracing context allocation so we can hand each device_interface
  // the matching per-queue tracing_context.
  if (iree_status_is_ok(status)) {
    for (int q = 0; q < IREE_HAL_CUDA_QUEUE_COUNT; ++q) {
      iree_hal_cuda_deferred_work_queue_device_interface_t* device_interface;
      status = iree_allocator_malloc(
          host_allocator,
          sizeof(iree_hal_cuda_deferred_work_queue_device_interface_t),
          (void**)&device_interface);
      if (!iree_status_is_ok(status)) {
        iree_hal_device_release((iree_hal_device_t*)device);
        return status;
      }
      device_interface->base.vtable =
          &iree_hal_cuda_deferred_work_queue_device_interface_vtable;
      device_interface->cu_context = context;
      device_interface->cuda_symbols = cuda_symbols;
      device_interface->cu_device = cu_device;
      device_interface->device = (iree_hal_device_t*)device;
      device_interface->dispatch_cu_stream = dispatch_streams[q];
      device_interface->tracing_context = device->tracing_contexts[q];
      device_interface->host_allocator = host_allocator;

      status = iree_hal_deferred_work_queue_create(
          (iree_hal_deferred_work_queue_device_interface_t*)device_interface,
          &device->block_pool, host_allocator, &device->work_queues[q]);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Integrated (Tegra/Jetson) detection — used in queue_alloca to route
  // transients through the sync allocator so they are CPU+GPU accessible.
  if (iree_status_is_ok(status)) {
    int is_integrated = 0;
    status = IREE_CURESULT_TO_STATUS(
        cuda_symbols,
        cuDeviceGetAttribute(&is_integrated,
                             CU_DEVICE_ATTRIBUTE_INTEGRATED, cu_device),
        "cuDeviceGetAttribute");
    if (iree_status_is_ok(status)) {
      device->is_integrated = is_integrated != 0;
    }
  }

  // Memory pool support is conditional.
  if (iree_status_is_ok(status) && params->async_allocations) {
    int supports_memory_pools = 0;
    status = IREE_CURESULT_TO_STATUS(
        cuda_symbols,
        cuDeviceGetAttribute(&supports_memory_pools,
                             CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                             cu_device),
        "cuDeviceGetAttribute");
    device->supports_memory_pools = supports_memory_pools != 0;
  }

  // Create memory pools first so that we can share them with the allocator.
  if (iree_status_is_ok(status) && device->supports_memory_pools) {
    status = iree_hal_cuda_memory_pools_initialize(
        (iree_hal_device_t*)device, cuda_symbols, cu_device,
        &params->memory_pools, host_allocator, &device->memory_pools);
  }

  if (iree_status_is_ok(status)) {
    // Allocator uses queue-0 stream for its allocations (cuMemAllocAsync/
    // cuMemFreeAsync). Per-queue allocators would complicate memory-pool
    // management and are not needed for correctness — allocations and frees
    // are stream-ordered on the allocator's own stream regardless of which
    // dispatch stream later uses the buffer.
    status = iree_hal_cuda_allocator_create(
        (iree_hal_device_t*)device, cuda_symbols, cu_device, context,
        dispatch_streams[0],
        device->supports_memory_pools ? &device->memory_pools : NULL,
        host_allocator, &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

// Enables peer access from the current device context to all other devices
// that support it. This allows cuMemcpyAsync to work across device contexts,
// which is required for multi-device execution where buffers allocated on one
// device may be accessed from another device's stream.
static iree_status_t iree_hal_cuda_device_enable_peer_access(
    const iree_hal_cuda_dynamic_symbols_t* symbols, CUdevice device_id) {
  int device_count = 0;
  IREE_CUDA_RETURN_IF_ERROR(symbols, cuDeviceGetCount(&device_count),
                            "cuDeviceGetCount");
  for (int j = 0; j < device_count; ++j) {
    if (j == (int)device_id) continue;
    int can_access = 0;
    CUresult result = symbols->cuDeviceCanAccessPeer(&can_access, device_id, j);
    if (result != CUDA_SUCCESS || !can_access) continue;
    CUdevice peer_device;
    result = symbols->cuDeviceGet(&peer_device, j);
    if (result != CUDA_SUCCESS) continue;
    CUcontext peer_context = NULL;
    result = symbols->cuDevicePrimaryCtxRetain(&peer_context, peer_device);
    if (result != CUDA_SUCCESS) continue;
    result = symbols->cuCtxEnablePeerAccess(peer_context, 0);
    if (result != CUDA_SUCCESS &&
        result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
      symbols->cuDevicePrimaryCtxRelease(peer_device);
      return iree_hal_cuda_result_to_status(symbols, result, __FILE__,
                                            __LINE__);
    }
    // Release the retain we did — the primary context stays alive as long as
    // the peer device retains it elsewhere.
    symbols->cuDevicePrimaryCtxRelease(peer_device);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_cuda_device_params_t* params,
    const iree_hal_cuda_dynamic_symbols_t* cuda_symbols,
    const iree_hal_cuda_nccl_dynamic_symbols_t* nccl_symbols, CUdevice device,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(cuda_symbols);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_cuda_device_check_params(params);

  // Get the main context for the device.
  CUcontext context = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_CURESULT_TO_STATUS(
        cuda_symbols, cuDevicePrimaryCtxRetain(&context, device));
  }
  if (iree_status_is_ok(status)) {
    status = IREE_CURESULT_TO_STATUS(cuda_symbols, cuCtxSetCurrent(context));
  }

  // Enable peer access to all other devices so that cross-device memory
  // operations work (e.g., cuMemcpyAsync between device contexts).
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_device_enable_peer_access(cuda_symbols, device);
  }

  // Create one dispatch stream per queue_affinity bit. CUDA can overlap
  // kernel execution across streams (subject to SM availability) so this
  // enables independent HAL queues on the same device to run concurrently.
  CUstream dispatch_streams[IREE_HAL_CUDA_QUEUE_COUNT] = {0};
  if (iree_status_is_ok(status)) {
    for (int q = 0; q < IREE_HAL_CUDA_QUEUE_COUNT; ++q) {
      status = IREE_CURESULT_TO_STATUS(
          cuda_symbols,
          cuStreamCreate(&dispatch_streams[q], CU_STREAM_NON_BLOCKING));
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Dedicated stream for bridging foreign-semaphore waits into CUDA events.
  CUstream foreign_bridge_stream = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_CURESULT_TO_STATUS(
        cuda_symbols,
        cuStreamCreate(&foreign_bridge_stream, CU_STREAM_NON_BLOCKING));
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_device_create_internal(
        driver, identifier, params, device, dispatch_streams, context,
        cuda_symbols, nccl_symbols, host_allocator, out_device);
    if (iree_status_is_ok(status)) {
      iree_hal_cuda_device_cast(*out_device)->foreign_bridge_cu_stream =
          foreign_bridge_stream;
    }
  } else {
    // Release resources we have acquired thus far.
    if (foreign_bridge_stream)
      cuda_symbols->cuStreamDestroy(foreign_bridge_stream);
    for (int q = 0; q < IREE_HAL_CUDA_QUEUE_COUNT; ++q) {
      if (dispatch_streams[q]) cuda_symbols->cuStreamDestroy(dispatch_streams[q]);
    }
    if (context) cuda_symbols->cuDevicePrimaryCtxRelease(device);
  }

  iree_event_pool_t* host_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_event_pool_allocate(params->event_pool_capacity,
                                      host_allocator, &host_event_pool);
  }

  iree_hal_cuda_event_pool_t* device_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_event_pool_allocate(
        cuda_symbols, params->event_pool_capacity, host_allocator,
        &device_event_pool);
  }

  iree_hal_cuda_timepoint_pool_t* timepoint_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_timepoint_pool_allocate(
        host_event_pool, device_event_pool, params->event_pool_capacity,
        host_allocator, &timepoint_pool);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_cuda_device_t* cuda_device =
        iree_hal_cuda_device_cast(*out_device);
    cuda_device->host_event_pool = host_event_pool;
    cuda_device->device_event_pool = device_event_pool;
    cuda_device->timepoint_pool = timepoint_pool;
  } else {
    // Release resources we have acquired after HAL device creation.
    if (timepoint_pool) iree_hal_cuda_timepoint_pool_free(timepoint_pool);
    if (device_event_pool) iree_hal_cuda_event_pool_release(device_event_pool);
    if (host_event_pool) iree_event_pool_free(host_event_pool);
    // Release other resources via the HAL device.
    iree_hal_device_release(*out_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

CUcontext iree_hal_cuda_device_context(iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device =
      iree_hal_cuda_device_cast_unsafe(base_device);
  return device->cu_context;
}

const iree_hal_cuda_dynamic_symbols_t* iree_hal_cuda_device_dynamic_symbols(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device =
      iree_hal_cuda_device_cast_unsafe(base_device);
  return device->cuda_symbols;
}

static void iree_hal_cuda_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  const iree_hal_cuda_dynamic_symbols_t* symbols = device->cuda_symbols;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Destroy the pending workload queues (one per dispatch stream).
  for (int q = 0; q < IREE_HAL_CUDA_QUEUE_COUNT; ++q) {
    if (device->work_queues[q]) {
      iree_hal_deferred_work_queue_destroy(device->work_queues[q]);
    }
  }

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Buffers may have been retaining collective resources.
  iree_hal_channel_provider_release(device->channel_provider);

  // Destroy memory pools that hold on to reserved memory.
  iree_hal_cuda_memory_pools_deinitialize(&device->memory_pools);

  for (int q = 0; q < IREE_HAL_CUDA_QUEUE_COUNT; ++q) {
    if (device->tracing_contexts[q]) {
      iree_hal_stream_tracing_context_free(device->tracing_contexts[q]);
      device->tracing_contexts[q] = NULL;
    }
  }
  device->tracing_context = NULL;

  // Destroy various pools for synchronization.
  if (device->timepoint_pool) {
    iree_hal_cuda_timepoint_pool_free(device->timepoint_pool);
  }
  if (device->device_event_pool) {
    iree_hal_cuda_event_pool_release(device->device_event_pool);
  }
  if (device->host_event_pool) iree_event_pool_free(device->host_event_pool);

  for (int q = 0; q < IREE_HAL_CUDA_QUEUE_COUNT; ++q) {
    if (device->dispatch_cu_streams[q]) {
      IREE_CUDA_IGNORE_ERROR(
          symbols, cuStreamDestroy(device->dispatch_cu_streams[q]));
    }
  }
  if (device->foreign_bridge_cu_stream) {
    IREE_CUDA_IGNORE_ERROR(symbols,
                           cuStreamDestroy(device->foreign_bridge_cu_stream));
  }

  IREE_CUDA_IGNORE_ERROR(symbols, cuDevicePrimaryCtxRelease(device->cu_device));

  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_cuda_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_cuda_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_cuda_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_cuda_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_cuda_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_cuda_device_trim(iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  if (device->supports_memory_pools) {
    IREE_RETURN_IF_ERROR(iree_hal_cuda_memory_pools_trim(
        &device->memory_pools, &device->params.memory_pools));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_query_attribute(
    iree_hal_cuda_device_t* device, CUdevice_attribute attribute,
    int64_t* out_value) {
  int value = 0;
  IREE_CUDA_RETURN_IF_ERROR(
      device->cuda_symbols,
      cuDeviceGetAttribute(&value, attribute, device->cu_device),
      "cuDeviceGetAttribute");
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("cuda-nvptx-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("cuda.device"))) {
    if (iree_string_view_equal(key, IREE_SV("compute_capability_major"))) {
      return iree_hal_cuda_device_query_attribute(
          device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, out_value);
    } else if (iree_string_view_equal(key,
                                      IREE_SV("compute_capability_minor"))) {
      return iree_hal_cuda_device_query_attribute(
          device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, out_value);
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_cuda_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_cuda_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_cuda_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  device->topology_info = *topology_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  if (!device->nccl_symbols || !device->nccl_symbols->dylib) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "NCCL runtime library version %d.%d and greater not available; "
        "ensure installed and the shared library (nccl.dll/libnccl.so) "
        "is on your PATH/LD_LIBRARY_PATH.",
        NCCL_MAJOR, NCCL_MINOR);
  }

  // Today we only allow a single logical device per channel.
  // We could multiplex channels but it'd be better to surface that to the
  // compiler so that it can emit the right rank math.
  int requested_count = iree_math_count_ones_u64(queue_affinity);
  // TODO(#12206): properly assign affinity in the compiler.
  if (requested_count != 64 && requested_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exactly one participant is allowed in a "
                            "channel but %d were specified",
                            requested_count);
  }

  // Ask the channel provider (if configured) for the default rank and count
  // if the user did not set them.
  if (device->channel_provider &&
      (params.rank == IREE_HAL_CHANNEL_RANK_DEFAULT ||
       params.count == IREE_HAL_CHANNEL_COUNT_DEFAULT)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_channel_provider_query_default_rank_and_count(
            device->channel_provider, &params.rank, &params.count),
        "querying default collective group rank and count");
  }

  // An ID is required to initialize NCCL. On the root it'll be the local ID and
  // on all other participants it'll be the root ID.
  iree_hal_cuda_nccl_id_t id;
  memset(&id, 0, sizeof(id));
  if (iree_const_byte_span_is_empty(params.id)) {
    // User wants the default ID.
    if (!device->channel_provider) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "default collective channel ID requested but no channel provider has "
          "been set on the device to provide it");
    }
    if (params.rank == 0) {
      // Bootstrap NCCL to get the root ID.
      IREE_RETURN_IF_ERROR(
          iree_hal_cuda_nccl_get_unique_id(device->nccl_symbols, &id),
          "bootstrapping NCCL root");
    }
    // Exchange NCCL ID with all participants.
    IREE_RETURN_IF_ERROR(iree_hal_channel_provider_exchange_default_id(
                             device->channel_provider,
                             iree_make_byte_span((void*)&id, sizeof(id))),
                         "exchanging NCCL ID with other participants");
  } else if (params.id.data_length != IREE_ARRAYSIZE(id.data)) {
    // User provided something but it's not what we expect.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NCCL ID must be %zu bytes matching the "
                            "ncclUniqueId struct but caller provided %zu bytes",
                            IREE_ARRAYSIZE(id.data), sizeof(id));
  } else {
    // User provided the ID - we treat it as opaque here and let NCCL validate.
    memcpy(id.data, params.id.data, IREE_ARRAYSIZE(id.data));
  }

  if (iree_hal_cuda_nccl_id_is_empty(&id)) {
    // TODO: maybe this is ok? a localhost alias or something?
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no default NCCL ID specified (all zeros)");
  }

  // TODO: when we support multiple logical devices we'll want to pass in the
  // context of the device mapped to the queue_affinity. For now since this
  // implementation only supports one device we pass in the only one we have.
  return iree_hal_cuda_nccl_channel_create(
      device->cuda_symbols, device->nccl_symbols, &id, params.rank,
      params.count, device->host_allocator, out_channel);
}

iree_status_t iree_hal_cuda_device_create_stream_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_stream_command_buffer_create(
      iree_hal_device_allocator(base_device), device->cuda_symbols,
      device->nccl_symbols, device->tracing_context, mode, command_categories,
      binding_capacity, device->dispatch_cu_streams[0], &device->block_pool,
      device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_cuda_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);

  switch (device->params.command_buffer_mode) {
    case IREE_HAL_CUDA_COMMAND_BUFFER_MODE_GRAPH: {
      // TODO(indirect-cmd): when we can record indirect graphs we won't need
      // to use deferred command buffers - this is here to emulate indirect
      // command buffers.
      if (binding_capacity > 0) {
        return iree_hal_deferred_command_buffer_create(
            iree_hal_device_allocator(base_device), mode, command_categories,
            queue_affinity, binding_capacity, &device->block_pool,
            iree_hal_device_host_allocator(base_device), out_command_buffer);
      } else {
        return iree_hal_cuda_graph_command_buffer_create(
            iree_hal_device_allocator(base_device), device->cuda_symbols,
            device->tracing_context, device->cu_context, mode,
            command_categories, queue_affinity, binding_capacity,
            &device->block_pool, device->host_allocator, out_command_buffer);
      }
    }
    case IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM: {
      return iree_hal_deferred_command_buffer_create(
          iree_hal_device_allocator(base_device), mode, command_categories,
          queue_affinity, binding_capacity, &device->block_pool,
          iree_hal_device_host_allocator(base_device), out_command_buffer);
    }
    default: {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid command buffer mode");
    }
  }
}

static iree_status_t iree_hal_cuda_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet implemented");
}

static iree_status_t iree_hal_cuda_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_nop_executable_cache_create(
      identifier, device->cuda_symbols, device->cu_context, device->cu_device,
      device->host_allocator, out_executable_cache);
}

static iree_status_t iree_hal_cuda_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_cuda_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  (void)queue_affinity;
  // A semaphore may be waited on by actions enqueued on any DWQ — not just
  // the one matching queue_affinity at creation time (the HAL creates some
  // semaphores with IREE_HAL_QUEUE_AFFINITY_ANY, and the same semaphore is
  // then used by actions on different queues). Give the semaphore every DWQ
  // on this device so signal-on-fire wakes all of them. Skipping this makes
  // actions parked on queues other than `ctz(queue_affinity)` hang forever.
  return iree_hal_cuda_event_semaphore_create(
      initial_value, device->cuda_symbols, device->timepoint_pool,
      device->work_queues, IREE_HAL_CUDA_QUEUE_COUNT, device->host_allocator,
      out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_cuda_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement CUDA semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

//===----------------------------------------------------------------------===//
// Foreign-semaphore bridge
//===----------------------------------------------------------------------===//
//
// Resolves a non-CUDA (foreign, e.g. local-task) semaphore wait before the
// caller submits downstream GPU work.
//
// Previous implementation routed this through cuLaunchHostFunc +
// cuEventRecord on a dedicated bridge stream so the VM submit thread stayed
// unblocked. That layered a ~24 ms host-side round-trip per bridge because
// CUDA serializes all host callbacks on a single runtime thread; see
// iree-issues/2026-04-24-heterogeneous-gpu-idle-gap.md for the breakdown.
//
// Since queue_execute always submits the downstream command buffer
// immediately after this call returns, ordering is preserved by simply
// waiting synchronously on the submit thread: once this returns OK, the
// foreign sema has reached |value|, and the subsequent cuLaunchKernel /
// cuGraphLaunch on the dispatch stream is correctly sequenced behind it
// without any CUDA-side event or stream-wait. No bridge event, no callback
// hop, no bridge stream allocation per call.
//
// Tradeoff: if a single queue_execute carried many foreign waits we would
// now resolve them one-by-one on the submit thread instead of letting the
// callback thread pipeline them. In practice submissions carry at most one
// foreign wait (the cross-device join) and that was serial to begin with.
static iree_status_t iree_hal_cuda_queue_bridge_foreign_wait(
    iree_hal_cuda_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_semaphore_t* semaphore, uint64_t value) {
  (void)device;
  (void)queue_affinity;
  // Fast path: already signaled — no wait needed.
  uint64_t current_value = 0;
  iree_status_t query_status =
      iree_hal_semaphore_query(semaphore, &current_value);
  if (iree_status_is_ok(query_status) && current_value >= value) {
    return iree_ok_status();
  }
  iree_status_ignore(query_status);

  // Block the submit thread until the foreign semaphore catches up. On
  // return the caller submits its command buffer; the dispatch stream FIFO
  // handles the rest.
  return iree_hal_semaphore_wait(semaphore, value, iree_infinite_timeout(),
                                 IREE_HAL_WAIT_FLAG_DEFAULT);
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in CUDA to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_cuda_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  // fprintf(stderr, "[ALLOCA] size=%zu\n", (size_t)allocation_size);

  // Honor all wait semaphores, but avoid blocking the VM submit thread on
  // foreign (non-CUDA) ones. Instead, bridge each foreign sem into a CUDA
  // event via cuLaunchHostFunc on a dedicated stream; then serialize the
  // dispatch stream behind a cuStreamWaitEvent for the bridge. The blocking
  // wait happens on the CUDA runtime's host-callback thread, not the VM.
  //
  // CUDA semaphores still pre-wait host-side (fast, no deadlock risk) —
  // this keeps the deferred work queue's action-readiness check simple.
  IREE_RETURN_IF_ERROR(IREE_CURESULT_TO_STATUS(
      device->cuda_symbols, cuCtxSetCurrent(device->cu_context),
      "cuCtxSetCurrent"));
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    if (iree_hal_cuda_semaphore_isa(wait_semaphore_list.semaphores[i])) {
      IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(
          wait_semaphore_list.semaphores[i],
          wait_semaphore_list.payload_values[i], iree_infinite_timeout(),
          IREE_HAL_WAIT_FLAG_DEFAULT));
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_cuda_queue_bridge_foreign_wait(
          device, queue_affinity, wait_semaphore_list.semaphores[i],
          wait_semaphore_list.payload_values[i]));
    }
  }
  // Integrated GPU (Tegra/Jetson): when the allocation is potentially
  // accessed from multiple queues (other devices or any-queue), route it
  // through the sync allocator so the result is host-mappable. This enables
  // cross-device consumption (CPU dispatch reading a GPU-produced
  // intermediate) without a staging transfer.
  //
  // Single-queue allocations still go through the async pool path so we
  // keep its steady-state cheap-reuse behavior. popcount(queue_affinity) > 1
  // catches ANY (all-bits) and any explicit multi-queue mask; == 1 means a
  // single device exclusively owns this buffer and the pool is fine.
  //
  // The sync allocator's refcount-based release_callback handles dealloca
  // correctly. The VM's queue_dealloca only signals semaphores; real freeing
  // is refcount-driven for both pool and sync buffers. Verified no leak in
  // 1700+ iter stress test.
  // Force HOST_VISIBLE on integrated GPUs only when there are peer devices
  // in the topology (multi-device setups where a CPU dispatch might read a
  // GPU-produced buffer). For single-device runs, queue_affinity is 0 and
  // there are no peers — keep pool semantics to preserve its cheap-reuse.
  const bool has_peers = device->topology_info.topology != NULL &&
                         device->topology_info.topology->device_count > 1;
  if (device->is_integrated && has_peers &&
      !iree_all_bits_set(params.type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    params.type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                   IREE_HAL_MEMORY_TYPE_HOST_CACHED;
    params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING |
                    IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  }

  iree_status_t status = iree_ok_status();
  if (device->supports_memory_pools &&
      !iree_all_bits_set(params.type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    int qi = iree_hal_cuda_device_select_queue_index(queue_affinity);
    status = iree_hal_cuda_memory_pools_alloca(
        &device->memory_pools, device->dispatch_cu_streams[qi], pool, params,
        allocation_size, flags, out_buffer);
  } else {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(base_device), params, allocation_size,
        out_buffer);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  return status;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in CUDA to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_cuda_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  // fprintf(stderr, "[DEALLOCA]\n");

  IREE_RETURN_IF_ERROR(IREE_CURESULT_TO_STATUS(
      device->cuda_symbols, cuCtxSetCurrent(device->cu_context),
      "cuCtxSetCurrent"));
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    if (iree_hal_cuda_semaphore_isa(wait_semaphore_list.semaphores[i])) {
      IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(
          wait_semaphore_list.semaphores[i],
          wait_semaphore_list.payload_values[i], iree_infinite_timeout(),
          IREE_HAL_WAIT_FLAG_DEFAULT));
    }
  }
  return iree_hal_semaphore_list_signal(signal_semaphore_list);
}

static iree_status_t iree_hal_cuda_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_cuda_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return loop_status;
}

static void iree_hal_cuda_device_collect_tracing_context(void* user_data) {
  iree_hal_stream_tracing_context_collect(
      (iree_hal_stream_tracing_context_t*)user_data);
}

// Context passed to the release callback for imported (host-registered) buffers.
typedef struct iree_hal_cuda_imported_buffer_info_t {
  const iree_hal_cuda_dynamic_symbols_t* cuda_symbols;
  void* host_ptr;
} iree_hal_cuda_imported_buffer_info_t;

// Release callback that unregisters host memory from CUDA when the wrapper
// buffer is destroyed.
static void iree_hal_cuda_imported_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  iree_hal_cuda_imported_buffer_info_t* info =
      (iree_hal_cuda_imported_buffer_info_t*)user_data;
  // Skip cuMemHostUnregister — keep memory registered across iterations
  // to avoid costly re-registration (cuMemHostRegister caching fix).
  (void)info->host_ptr;
  iree_allocator_free(iree_allocator_system(), info);
}

static iree_status_t iree_hal_cuda_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Handle foreign (non-CUDA) wait semaphores by bridging them into CUDA
  // events on the dispatch stream (same strategy as queue_alloca). The
  // deferred work queue below can only process CUDA-native waits via device
  // events; leaving a foreign wait in the list would deadlock because the
  // foreign HAL wouldn't know to re-issue the DWQ on signal. Instead we:
  //   1. For each foreign wait, queue a cuLaunchHostFunc + cuEventRecord on
  //      the bridge stream and cuStreamWaitEvent on the dispatch stream so
  //      any GPU work enqueued next waits on the foreign-sem completion
  //      device-side.
  //   2. Filter the foreign waits out of the list handed to the DWQ.
  //
  // The VM submit thread is not blocked by the foreign wait; the blocking
  // happens on the CUDA runtime's host-callback thread.
  iree_hal_semaphore_list_t native_wait_list = wait_semaphore_list;
  iree_hal_semaphore_t* filtered_semaphores_storage[16] = {0};
  uint64_t filtered_values_storage[16] = {0};
  iree_hal_semaphore_t** filtered_semaphores_heap = NULL;
  uint64_t* filtered_values_heap = NULL;
  iree_host_size_t foreign_count = 0;
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    if (!iree_hal_cuda_semaphore_isa(wait_semaphore_list.semaphores[i])) {
      ++foreign_count;
    }
  }
  if (foreign_count > 0 && foreign_count < wait_semaphore_list.count) {
    // Mixed list — need to filter out the foreign entries.
    iree_hal_semaphore_t** out_sems = filtered_semaphores_storage;
    uint64_t* out_vals = filtered_values_storage;
    iree_host_size_t native_count =
        wait_semaphore_list.count - foreign_count;
    if (native_count > IREE_ARRAYSIZE(filtered_semaphores_storage)) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_allocator_malloc(
                  device->host_allocator,
                  native_count * sizeof(*filtered_semaphores_heap),
                  (void**)&filtered_semaphores_heap));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_allocator_malloc(
                  device->host_allocator,
                  native_count * sizeof(*filtered_values_heap),
                  (void**)&filtered_values_heap));
      out_sems = filtered_semaphores_heap;
      out_vals = filtered_values_heap;
    }
    iree_host_size_t w = 0;
    for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
      if (iree_hal_cuda_semaphore_isa(wait_semaphore_list.semaphores[i])) {
        out_sems[w] = wait_semaphore_list.semaphores[i];
        out_vals[w] = wait_semaphore_list.payload_values[i];
        ++w;
      }
    }
    native_wait_list.count = native_count;
    native_wait_list.semaphores = out_sems;
    native_wait_list.payload_values = out_vals;
  } else if (foreign_count == wait_semaphore_list.count) {
    // All-foreign list — DWQ will get an empty wait list.
    native_wait_list.count = 0;
    native_wait_list.semaphores = NULL;
    native_wait_list.payload_values = NULL;
  }
  // Now bridge each foreign wait.
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    if (!iree_hal_cuda_semaphore_isa(wait_semaphore_list.semaphores[i])) {
      iree_status_t status = iree_hal_cuda_queue_bridge_foreign_wait(
          device, queue_affinity, wait_semaphore_list.semaphores[i],
          wait_semaphore_list.payload_values[i]);
      if (!iree_status_is_ok(status)) {
        if (filtered_semaphores_heap) {
          iree_allocator_free(device->host_allocator,
                              filtered_semaphores_heap);
        }
        if (filtered_values_heap) {
          iree_allocator_free(device->host_allocator, filtered_values_heap);
        }
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
    }
  }

  // Ensure the correct CUDA context for host memory registration.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_CURESULT_TO_STATUS(device->cuda_symbols,
                                   cuCtxSetCurrent(device->cu_context),
                                   "cuCtxSetCurrent"));

  // Import foreign (non-CUDA) buffers in the binding table by registering
  // their host memory with CUDA via cuMemHostRegister. This gives us a valid
  // CUdeviceptr that GPU kernels can use to access the data.
  // We create a mutable copy of the binding table and replace foreign buffer
  // entries with CUDA wrapper buffers.
  iree_hal_buffer_binding_t* local_bindings = NULL;
  iree_hal_buffer_binding_table_t local_binding_table = binding_table;
  if (binding_table.count > 0) {
    iree_host_size_t bindings_size =
        binding_table.count * sizeof(iree_hal_buffer_binding_t);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(device->host_allocator, bindings_size,
                                  (void**)&local_bindings));
    memcpy(local_bindings, binding_table.bindings, bindings_size);
    local_binding_table.bindings = local_bindings;

    for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
      iree_hal_buffer_t* buffer = local_bindings[i].buffer;
      if (!buffer) continue;

      // Get the underlying allocated buffer (unwrap subspans).
      iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
      if (iree_hal_cuda_buffer_isa(allocated)) continue;

      // Unified memory (Jetson): foreign buffers are already GPU-accessible.
      // Map to get the host pointer, which on unified memory IS the device
      // pointer. No cuMemHostRegister, no wrapper buffer needed.
      iree_hal_buffer_mapping_t mapping;
      iree_status_t status = iree_hal_buffer_map_range(
          allocated, IREE_HAL_MAPPING_MODE_PERSISTENT,
          IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, 0,
          iree_hal_buffer_allocation_size(allocated), &mapping);
      if (!iree_status_is_ok(status)) {
        status = iree_hal_buffer_map_range(
            allocated, IREE_HAL_MAPPING_MODE_PERSISTENT,
            IREE_HAL_MEMORY_ACCESS_READ, 0,
            iree_hal_buffer_allocation_size(allocated), &mapping);
      }
      if (!iree_status_is_ok(status)) {
        for (iree_host_size_t j = 0; j < i; ++j) {
          if (local_bindings[j].buffer != binding_table.bindings[j].buffer) {
            iree_hal_buffer_release(local_bindings[j].buffer);
          }
        }
        iree_allocator_free(device->host_allocator, local_bindings);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }

      void* host_ptr = mapping.contents.data;
      iree_device_size_t alloc_size =
          iree_hal_buffer_allocation_size(allocated);
      iree_hal_buffer_unmap_range(&mapping);

      // On unified memory, managed pointers are already GPU-accessible.
      // Try cuMemHostGetDevicePointer first (zero-cost for managed memory).
      // Only fall back to cuMemHostRegister for plain malloc'd memory.
      CUdeviceptr device_ptr = 0;
      CUresult get_result =
          device->cuda_symbols->cuMemHostGetDevicePointer(
              &device_ptr, host_ptr, 0);
      if (get_result != CUDA_SUCCESS) {
        CUresult reg_result = device->cuda_symbols->cuMemHostRegister(
            host_ptr, (size_t)alloc_size,
            CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_PORTABLE);
        if (reg_result != CUDA_SUCCESS &&
            reg_result != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
          status = iree_make_status(
              IREE_STATUS_INTERNAL,
              "cuMemHostRegister failed with %d", (int)reg_result);
        }
        if (iree_status_is_ok(status)) {
          status = IREE_CURESULT_TO_STATUS(
              device->cuda_symbols,
              cuMemHostGetDevicePointer(&device_ptr, host_ptr, 0),
              "cuMemHostGetDevicePointer");
        }
        if (!iree_status_is_ok(status)) {
          for (iree_host_size_t j = 0; j < i; ++j) {
            if (local_bindings[j].buffer != binding_table.bindings[j].buffer) {
              iree_hal_buffer_release(local_bindings[j].buffer);
            }
          }
          iree_allocator_free(device->host_allocator, local_bindings);
          IREE_TRACE_ZONE_END(z0);
          return status;
        }
      }
      iree_hal_buffer_t* cuda_buffer = NULL;
      const iree_hal_buffer_placement_t placement = {
          .device = base_device,
          .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
          .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
      };
      iree_hal_buffer_release_callback_t release_callback = {
          .fn = NULL,
          .user_data = NULL,
      };
      status = iree_hal_cuda_buffer_wrap(
          placement,
          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
              IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          IREE_HAL_MEMORY_ACCESS_ALL, iree_hal_buffer_allowed_usage(allocated),
          alloc_size, /*byte_offset=*/0,
          /*byte_length=*/alloc_size,
          IREE_HAL_CUDA_BUFFER_TYPE_HOST_REGISTERED, device_ptr, host_ptr,
          release_callback, device->host_allocator, &cuda_buffer);
      if (!iree_status_is_ok(status)) {
        for (iree_host_size_t j = 0; j < i; ++j) {
          if (local_bindings[j].buffer != binding_table.bindings[j].buffer) {
            iree_hal_buffer_release(local_bindings[j].buffer);
          }
        }
        iree_allocator_free(device->host_allocator, local_bindings);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }

      local_bindings[i].buffer = cuda_buffer;
      // The original binding offset is relative to the original buffer which
      // may be a subspan. We need to add the subspan's byte_offset.
      local_bindings[i].offset =
          binding_table.bindings[i].offset +
          iree_hal_buffer_byte_offset(buffer);
      local_bindings[i].length = binding_table.bindings[i].length;
    }
  }

  // Stamp the last-writer fence on each CUDA buffer in the binding table so
  // the caching allocator can skip them on pool-reuse scan if not yet ready.
  // The first CUDA-native signal semaphore is what completes when the DWQ
  // action (and hence the writes to these buffers) finishes. Over-stamping
  // readers is correct: waiting on the last-reader fence before reuse is
  // strictly safe. See
  // iree-issues/2026-04-24-cuda-resource-set-bypasses-pooling-allocator.md.
  //
  // Limitation: this covers indirect-binding dispatches (binding_capacity>0).
  // Command buffers with direct bindings (binding_capacity==0) and the
  // emulated queue_copy/update/fill paths don't populate binding_table; all
  // pooled transients in our scenario flow through binding_table.
  iree_hal_semaphore_t* stamp_sema = NULL;
  uint64_t stamp_value = 0;
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    if (iree_hal_cuda_semaphore_isa(signal_semaphore_list.semaphores[i])) {
      stamp_sema = signal_semaphore_list.semaphores[i];
      stamp_value = signal_semaphore_list.payload_values[i];
      break;
    }
  }
  if (stamp_sema && local_binding_table.count > 0) {
    for (iree_host_size_t i = 0; i < local_binding_table.count; ++i) {
      iree_hal_buffer_t* buffer = local_binding_table.bindings[i].buffer;
      if (!buffer) continue;
      iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
      if (iree_hal_cuda_buffer_isa(allocated)) {
        iree_hal_cuda_buffer_stamp_last_writer(allocated, stamp_sema,
                                               stamp_value);
      }
    }
  }

  // Hand the DWQ only the native-CUDA waits; foreign waits are bridged
  // device-side via cuStreamWaitEvent above. Pick the DWQ for this
  // queue_affinity so independent queues run on independent CUDA streams.
  int exec_qi = iree_hal_cuda_device_select_queue_index(queue_affinity);
  iree_status_t status = iree_hal_deferred_work_queue_enqueue(
      device->work_queues[exec_qi],
      iree_hal_cuda_device_collect_tracing_context,
      device->tracing_contexts[exec_qi],
      native_wait_list, signal_semaphore_list,
      command_buffer ? 1 : 0, command_buffer ? &command_buffer : NULL,
      &local_binding_table);
  if (iree_status_is_ok(status)) {
    // Try to advance the deferred work queue.
    status = iree_hal_deferred_work_queue_issue(device->work_queues[exec_qi]);
  }

  // Release our references to imported buffers. The DWQ's resource_set now
  // holds references to them and will release when the action completes.
  if (local_bindings) {
    for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
      if (local_bindings[i].buffer != binding_table.bindings[i].buffer) {
        iree_hal_buffer_release(local_bindings[i].buffer);
      }
    }
    iree_allocator_free(device->host_allocator, local_bindings);
  }
  if (filtered_semaphores_heap) {
    iree_allocator_free(device->host_allocator, filtered_semaphores_heap);
  }
  if (filtered_values_heap) {
    iree_allocator_free(device->host_allocator, filtered_values_heap);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  // Advance every DWQ — flush is queue-unspecific in its common call sites
  // (e.g., VM end-of-invocation). Cheap no-op if a DWQ has nothing pending.
  iree_status_t status = iree_ok_status();
  for (int q = 0; q < IREE_HAL_CUDA_QUEUE_COUNT; ++q) {
    iree_status_t s = iree_hal_deferred_work_queue_issue(device->work_queues[q]);
    if (iree_status_is_ok(status) && !iree_status_is_ok(s)) status = s;
    else if (!iree_status_is_ok(s)) iree_status_ignore(s);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_semaphore_multi_wait(semaphore_list, wait_mode, timeout,
                                            flags, &device->block_pool);
}

static iree_status_t iree_hal_cuda_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  // We could hook in to CUPTI here or use the much simpler cuProfilerStart API.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_cuda_device_vtable = {
    .destroy = iree_hal_cuda_device_destroy,
    .id = iree_hal_cuda_device_id,
    .host_allocator = iree_hal_cuda_device_host_allocator,
    .device_allocator = iree_hal_cuda_device_allocator,
    .replace_device_allocator = iree_hal_cuda_replace_device_allocator,
    .replace_channel_provider = iree_hal_cuda_replace_channel_provider,
    .trim = iree_hal_cuda_device_trim,
    .query_i64 = iree_hal_cuda_device_query_i64,
    .query_capabilities = iree_hal_cuda_device_query_capabilities,
    .topology_info = iree_hal_cuda_device_topology_info,
    .refine_topology_edge = iree_hal_cuda_device_refine_topology_edge,
    .assign_topology_info = iree_hal_cuda_device_assign_topology_info,
    .create_channel = iree_hal_cuda_device_create_channel,
    .create_command_buffer = iree_hal_cuda_device_create_command_buffer,
    .create_event = iree_hal_cuda_device_create_event,
    .create_executable_cache = iree_hal_cuda_device_create_executable_cache,
    .import_file = iree_hal_cuda_device_import_file,
    .create_semaphore = iree_hal_cuda_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_cuda_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_cuda_device_queue_alloca,
    .queue_dealloca = iree_hal_cuda_device_queue_dealloca,
    .queue_fill = iree_hal_device_queue_emulated_fill,
    .queue_update = iree_hal_device_queue_emulated_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_cuda_device_queue_read,
    .queue_write = iree_hal_cuda_device_queue_write,
    .queue_host_call = iree_hal_device_queue_emulated_host_call,
    .queue_dispatch = iree_hal_device_queue_emulated_dispatch,
    .queue_execute = iree_hal_cuda_device_queue_execute,
    .queue_flush = iree_hal_cuda_device_queue_flush,
    .wait_semaphores = iree_hal_cuda_device_wait_semaphores,
    .profiling_begin = iree_hal_cuda_device_profiling_begin,
    .profiling_flush = iree_hal_cuda_device_profiling_flush,
    .profiling_end = iree_hal_cuda_device_profiling_end,
};

static const iree_hal_deferred_work_queue_device_interface_vtable_t
    iree_hal_cuda_deferred_work_queue_device_interface_vtable = {
        .destroy = iree_hal_cuda_deferred_work_queue_device_interface_destroy,
        .bind_to_thread =
            iree_hal_cuda_deferred_work_queue_device_interface_bind_to_thread,
        .wait_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_wait_native_event,
        .create_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_create_native_event,
        .record_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_record_native_event,
        .synchronize_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_synchronize_native_event,
        .destroy_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_destroy_native_event,
        .semaphore_acquire_timepoint_device_signal_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_semaphore_acquire_timepoint_device_signal_native_event,
        .acquire_host_wait_event =
            iree_hal_cuda_deferred_work_queue_device_interface_acquire_host_wait_event,
        .device_wait_on_host_event =
            iree_hal_cuda_deferred_work_queue_device_interface_device_wait_on_host_event,
        .release_wait_event =
            iree_hal_cuda_deferred_work_queue_device_interface_release_wait_event,
        .native_event_from_wait_event =
            iree_hal_cuda_deferred_work_queue_device_interface_native_event_from_wait_event,
        .create_stream_command_buffer =
            iree_hal_cuda_deferred_work_queue_device_interface_create_stream_command_buffer,
        .submit_command_buffer =
            iree_hal_cuda_deferred_work_queue_device_interface_submit_command_buffer,
        .async_alloc =
            iree_hal_cuda_deferred_work_queue_device_interface_async_alloc,
        .async_dealloc =
            iree_hal_cuda_deferred_work_queue_device_interface_async_dealloc,
};

static const iree_hal_stream_tracing_device_interface_vtable_t
    iree_hal_cuda_tracing_device_interface_vtable_t = {
        .destroy = iree_hal_cuda_tracing_device_interface_destroy,
        .synchronize_native_event =
            iree_hal_cuda_tracing_device_interface_synchronize_native_event,
        .create_native_event =
            iree_hal_cuda_tracing_device_interface_create_native_event,
        .query_native_event =
            iree_hal_cuda_tracing_device_interface_query_native_event,
        .event_elapsed_time =
            iree_hal_cuda_tracing_device_interface_event_elapsed_time,
        .destroy_native_event =
            iree_hal_cuda_tracing_device_interface_destroy_native_event,
        .record_native_event =
            iree_hal_cuda_tracing_device_interface_record_native_event,
        .add_graph_event_record_node =
            iree_hal_cuda_tracing_device_interface_add_graph_event_record_node,
};
