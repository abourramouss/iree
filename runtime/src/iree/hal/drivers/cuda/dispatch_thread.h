// CUDA dispatch thread for async kernel submission.
#ifndef IREE_HAL_DRIVERS_CUDA_DISPATCH_THREAD_H_
#define IREE_HAL_DRIVERS_CUDA_DISPATCH_THREAD_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

typedef struct iree_hal_cuda_dispatch_thread_t iree_hal_cuda_dispatch_thread_t;

typedef iree_status_t (*iree_hal_cuda_dispatch_callback_t)(void* user_data,
                                                           iree_status_t status);

iree_status_t iree_hal_cuda_dispatch_thread_initialize(
    iree_allocator_t host_allocator,
    iree_hal_cuda_dispatch_thread_t** out_thread);

void iree_hal_cuda_dispatch_thread_deinitialize(
    iree_hal_cuda_dispatch_thread_t* thread);

iree_status_t iree_hal_cuda_dispatch_thread_add_dispatch(
    iree_hal_cuda_dispatch_thread_t* thread,
    iree_hal_cuda_dispatch_callback_t callback, void* user_data);

iree_status_t iree_hal_cuda_dispatch_thread_add_dispatch_sync(
    iree_hal_cuda_dispatch_thread_t* thread,
    iree_hal_cuda_dispatch_callback_t callback, void* user_data);

#endif
