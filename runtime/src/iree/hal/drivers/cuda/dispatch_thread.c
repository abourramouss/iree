// CUDA dispatch thread — async kernel submission.
#include "iree/hal/drivers/cuda/dispatch_thread.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include <stdlib.h>
#include <string.h>

#define QUEUE_SIZE 256

typedef struct {
  iree_hal_cuda_dispatch_callback_t callback;
  void* user_data;
} item_t;

struct iree_hal_cuda_dispatch_thread_t {
  iree_thread_t* thread;
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;
  iree_notification_t notification;
  bool do_exit;
  iree_status_t failure_status;
  item_t items[QUEUE_SIZE];
  int head, tail, count;
};

static bool has_work(void* ud) {
  iree_hal_cuda_dispatch_thread_t* t = (iree_hal_cuda_dispatch_thread_t*)ud;
  iree_slim_mutex_lock(&t->mutex);
  bool r = (t->count > 0) || t->do_exit;
  iree_slim_mutex_unlock(&t->mutex);
  return r;
}

static int thread_main(void* param) {
  iree_hal_cuda_dispatch_thread_t* t = (iree_hal_cuda_dispatch_thread_t*)param;
  while (true) {
    iree_notification_await(&t->notification, &has_work, t, iree_infinite_timeout());
    iree_slim_mutex_lock(&t->mutex);
    bool ex = t->do_exit;
    iree_status_t st = iree_status_clone(t->failure_status);
    while (t->count > 0) {
      item_t item = t->items[t->head];
      t->head = (t->head + 1) % QUEUE_SIZE;
      t->count--;
      iree_slim_mutex_unlock(&t->mutex);
      st = item.callback(item.user_data, st);
      iree_slim_mutex_lock(&t->mutex);
      if (!iree_status_is_ok(st)) {
        iree_status_ignore(t->failure_status);
        t->failure_status = iree_status_clone(st);
      }
    }
    iree_slim_mutex_unlock(&t->mutex);
    if (!iree_status_is_ok(st) || ex) { iree_status_ignore(st); break; }
  }
  return 0;
}

iree_status_t iree_hal_cuda_dispatch_thread_initialize(
    iree_allocator_t host_allocator,
    iree_hal_cuda_dispatch_thread_t** out_thread) {
  *out_thread = NULL;
  iree_hal_cuda_dispatch_thread_t* t = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*t), (void**)&t));
  memset(t, 0, sizeof(*t));
  t->host_allocator = host_allocator;
  t->failure_status = iree_ok_status();
  iree_slim_mutex_initialize(&t->mutex);
  iree_notification_initialize(&t->notification);
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = iree_make_cstring_view("iree-hal-cuda-dispatch");
  iree_status_t status = iree_thread_create(thread_main, t, params, host_allocator, &t->thread);
  if (iree_status_is_ok(status)) { *out_thread = t; }
  else { iree_slim_mutex_deinitialize(&t->mutex); iree_allocator_free(host_allocator, t); }
  return status;
}

void iree_hal_cuda_dispatch_thread_deinitialize(iree_hal_cuda_dispatch_thread_t* t) {
  if (!t) return;
  iree_slim_mutex_lock(&t->mutex);
  t->do_exit = true;
  iree_slim_mutex_unlock(&t->mutex);
  iree_notification_post(&t->notification, IREE_ALL_WAITERS);
  iree_thread_release(t->thread);
  iree_status_ignore(t->failure_status);
  iree_slim_mutex_deinitialize(&t->mutex);
  iree_allocator_free(t->host_allocator, t);
}

iree_status_t iree_hal_cuda_dispatch_thread_add_dispatch(
    iree_hal_cuda_dispatch_thread_t* t,
    iree_hal_cuda_dispatch_callback_t callback, void* user_data) {
  iree_slim_mutex_lock(&t->mutex);
  iree_status_t status = iree_status_clone(t->failure_status);
  if (iree_status_is_ok(status)) {
    if (t->count >= QUEUE_SIZE) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "dispatch queue full");
    } else {
      t->items[t->tail].callback = callback;
      t->items[t->tail].user_data = user_data;
      t->tail = (t->tail + 1) % QUEUE_SIZE;
      t->count++;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(t->failure_status);
    t->failure_status = iree_status_clone(status);
  }
  iree_slim_mutex_unlock(&t->mutex);
  iree_notification_post(&t->notification, IREE_ALL_WAITERS);
  if (!iree_status_is_ok(status))
    iree_status_ignore(callback(user_data, iree_status_clone(status)));
  return status;
}

// Sync dispatch: push + SPIN-WAIT for completion (avoids CONFIG_HZ latency).
typedef struct {
  iree_hal_cuda_dispatch_callback_t callback;
  void* user_data;
  volatile int done;  // atomic flag: 0=pending, 1=done
  iree_status_t result;
} sync_data_t;

static iree_status_t sync_wrapper(void* ud, iree_status_t st) {
  sync_data_t* s = (sync_data_t*)ud;
  s->result = s->callback(s->user_data, st);
  __sync_synchronize();  // memory barrier
  s->done = 1;
  return s->result;
}

iree_status_t iree_hal_cuda_dispatch_thread_add_dispatch_sync(
    iree_hal_cuda_dispatch_thread_t* t,
    iree_hal_cuda_dispatch_callback_t callback, void* user_data) {
  sync_data_t s;
  s.callback = callback;
  s.user_data = user_data;
  s.done = 0;
  s.result = iree_ok_status();
  iree_status_t status = iree_hal_cuda_dispatch_thread_add_dispatch(t, sync_wrapper, &s);
  if (!iree_status_is_ok(status)) return status;
  // Spin-wait (sub-microsecond latency vs 4ms for notification)
  while (!s.done) {
    __sync_synchronize();
  }
  return s.result;
}
