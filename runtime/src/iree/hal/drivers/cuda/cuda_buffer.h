// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_CUDA_BUFFER_H_
#define IREE_HAL_DRIVERS_CUDA_CUDA_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/cuda_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_cuda_buffer_type_e {
  // Device local buffer; allocated with cuMemAlloc/cuMemAllocManaged, freed
  // with cuMemFree.
  IREE_HAL_CUDA_BUFFER_TYPE_DEVICE = 0,
  // Host local buffer; allocated with cuMemHostAlloc, freed with cuMemFreeHost.
  IREE_HAL_CUDA_BUFFER_TYPE_HOST,
  // Host local buffer; registered with cuMemHostRegister, freed with
  // cuMemHostUnregister.
  IREE_HAL_CUDA_BUFFER_TYPE_HOST_REGISTERED,
  // Device local buffer, allocated with cuMemAllocFromPoolAsync, freed with
  // cuMemFree/cuMemFreeAsync.
  IREE_HAL_CUDA_BUFFER_TYPE_ASYNC,
  // Externally registered buffer whose providence is unknown.
  // Must be freed by the user.
  IREE_HAL_CUDA_BUFFER_TYPE_EXTERNAL,
} iree_hal_cuda_buffer_type_t;

// Wraps a CUDA allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_cuda_buffer_wrap(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_cuda_buffer_type_t buffer_type, CUdeviceptr device_ptr,
    void* host_ptr, iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is a CUDA buffer (as opposed to a foreign buffer
// from another HAL driver like local-task).
bool iree_hal_cuda_buffer_isa(const iree_hal_buffer_t* buffer);

// Returns the underlying CUDA buffer type of the given |buffer|.
iree_hal_cuda_buffer_type_t iree_hal_cuda_buffer_type(
    const iree_hal_buffer_t* buffer);

// Returns the CUDA base device pointer for the given |buffer|.
//
// Note that this is the entire allocated_buffer and must be offset by the
// buffer byte_offset and byte_length when used.
CUdeviceptr iree_hal_cuda_buffer_device_pointer(
    const iree_hal_buffer_t* buffer);

// Returns the CUDA host pointer for the given |buffer|, if available.
void* iree_hal_cuda_buffer_host_pointer(const iree_hal_buffer_t* buffer);

// Sets the device pointer on an existing buffer.
// Used for deferred async allocations where the buffer object is created first
// and the device memory is allocated later.
void iree_hal_cuda_buffer_set_device_pointer(iree_hal_buffer_t* buffer,
                                              CUdeviceptr device_ptr);

// Clears the device pointer, marking the buffer as empty/unallocated.
void iree_hal_cuda_buffer_set_allocation_empty(iree_hal_buffer_t* buffer);

// Drops the release callback so that when the buffer is destroyed no callback
// will be made. This is not thread safe but all callers are expected to be
// holding an allocation and the earliest the buffer could be destroyed is after
// this call returns and the caller has released its reference.
void iree_hal_cuda_buffer_drop_release_callback(iree_hal_buffer_t* buffer);

// Stamps |sema|/|value| as the last-writer fence for |buffer|. A subsequent
// non-blocking readiness check (iree_hal_cuda_buffer_query_ready) can tell
// whether the buffer is safe to hand out for reuse: ready iff the stamped
// fence has been signaled. Each call replaces any prior stamp. |sema| is
// retained while stamped. Passing a NULL |sema| is a no-op.
//
// Call at queue_execute time for every buffer touched by the command buffer
// (over-stamping readers is correct — waiting on the last-reader fence before
// reuse is strictly safe). Must be a CUDA-native semaphore (foreign sems are
// bridged elsewhere and do not flow through this path).
void iree_hal_cuda_buffer_stamp_last_writer(iree_hal_buffer_t* buffer,
                                            iree_hal_semaphore_t* sema,
                                            uint64_t value);

// Returns true in |*out_ready| if the last-writer fence of |buffer| has been
// signaled (or no fence was ever stamped). Non-blocking: calls
// iree_hal_semaphore_query. Safe to call with any mutex held; does not enter
// the allocator or block. Returns an error if the query itself fails; on
// success |*out_ready| is always populated.
iree_status_t iree_hal_cuda_buffer_query_ready(
    const iree_hal_buffer_t* buffer, bool* IREE_RESTRICT out_ready);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_CUDA_BUFFER_H_
