// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C++ helpers for using IREE threading primitives.

#ifndef SHORTFIN_SUPPORT_IREE_THREADING_H
#define SHORTFIN_SUPPORT_IREE_THREADING_H

#include <atomic>

#include "iree/base/threading/api.h"
#include "iree/base/wait_source.h"
#include "shortfin/support/iree_helpers.h"

// Set up threading annotations.
#if defined(SHORTFIN_HAS_THREAD_SAFETY_ANNOTATIONS)
#define SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(x) __attribute__((x))
#else
#define SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(x)
#endif

#define SHORTFIN_GUARDED_BY(x) \
  SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(guarded_by(x))
#define SHORTFIN_REQUIRES_LOCK(...) \
  SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(__VA_ARGS__))

namespace shortfin::iree {

SHORTFIN_IREE_DEF_PTR(thread);

// Wraps an iree::slim_mutex as an RAII object.
class SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(capability("mutex")) slim_mutex {
 public:
  slim_mutex() { iree_slim_mutex_initialize(&mu_); }
  slim_mutex(const slim_mutex &) = delete;
  slim_mutex &operator=(const slim_mutex &) = delete;
  ~slim_mutex() { iree_slim_mutex_deinitialize(&mu_); }

  operator iree_slim_mutex_t *() { return &mu_; }

  void Lock() SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(acquire_capability()) {
    iree_slim_mutex_lock(&mu_);
  }

  void Unlock() SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(release_capability()) {
    iree_slim_mutex_unlock(&mu_);
  }

 private:
  iree_slim_mutex_t mu_;
};

// RAII slim mutex lock guard.
class SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(scoped_lockable)
    slim_mutex_lock_guard {
 public:
  slim_mutex_lock_guard(slim_mutex &mu)
      SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(acquire_capability(mu))
      : mu_(mu) {
    mu_.Lock();
  }
  ~slim_mutex_lock_guard()
      SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(release_capability()) {
    mu_.Unlock();
  }

 private:
  slim_mutex &mu_;
};

// Wrapper around iree_notification_t that provides event-like semantics.
// Note: The old iree_event_t API was removed in IREE commit d7f5aba
// ("Unify HAL semaphores on async infrastructure"). This class provides
// a compatibility layer using notification + a custom wait source.
class event {
 public:
  event(bool initial_state) : epoch_(initial_state ? 1u : 0u) {
    iree_notification_initialize(&notification_);
    if (initial_state) {
      iree_notification_post(&notification_, IREE_ALL_WAITERS);
    }
  }
  event(const event &) = delete;
  event &operator=(const event &) = delete;
  ~event() { iree_notification_deinitialize(&notification_); }

  void set() {
    epoch_.fetch_add(1, std::memory_order_release);
    iree_notification_post(&notification_, IREE_ALL_WAITERS);
  }

  void reset() {
    // Reset by advancing to next even epoch (unsignaled state)
    uint32_t current = epoch_.load(std::memory_order_acquire);
    if (current % 2 == 1) {
      epoch_.fetch_add(1, std::memory_order_release);
    }
  }

  iree_wait_source_t await() {
    return (iree_wait_source_t){
        .self = this,
        .data = epoch_.load(std::memory_order_acquire),
        .resolve = &event::resolve_wait,
    };
  }

 private:
  static iree_status_t resolve_wait(iree_wait_source_t wait_source,
                                     iree_timeout_t timeout,
                                     iree_wait_source_resolve_callback_t callback,
                                     void* user_data) {
    auto* self = static_cast<event*>(wait_source.self);
    uint32_t target_epoch = static_cast<uint32_t>(wait_source.data);

    // If callback is provided, async wait is not supported in this simple impl
    if (callback) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "async wait not supported on event wrapper");
    }

    // Synchronous wait: wait for epoch to advance past target (signaled state)
    iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
    while (self->epoch_.load(std::memory_order_acquire) <= target_epoch) {
      if (iree_time_now() >= deadline_ns &&
          !iree_timeout_is_infinite(timeout)) {
        return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
      // Prepare and commit wait on notification
      iree_wait_token_t token =
          iree_notification_prepare_wait(&self->notification_);
      // Check again before committing (avoid race)
      if (self->epoch_.load(std::memory_order_acquire) > target_epoch) {
        break;
      }
      iree_notification_commit_wait(&self->notification_, token,
                                    iree_make_timeout_ms(10),
                                    IREE_DURATION_ZERO);
    }
    return iree_ok_status();
  }

  iree_notification_t notification_;
  std::atomic<uint32_t> epoch_;
};

// An event that is ref-counted.
class shared_event : private event {
 public:
  class ref {
   public:
    ref() = default;
    explicit ref(bool initial_state) : inst_(new shared_event(initial_state)) {}
    ref(const ref &other) : inst_(other.inst_) {
      if (inst_) {
        inst_->ref_count_.fetch_add(1);
      }
    }
    ref &operator=(const ref &other) {
      if (inst_ != other.inst_) {
        reset();
        inst_ = other.inst_;
        if (inst_) {
          inst_->ref_count_.fetch_add(1);
        }
      }
      return *this;
    }
    ref(ref &&other) : inst_(other.inst_) { other.inst_ = nullptr; }
    ~ref() { reset(); }

    operator bool() const { return inst_ != nullptr; }
    iree::event *operator->() const { return inst_; }
    void reset() {
      if (inst_) {
        manual_release();
        inst_ = nullptr;
      }
    }

    int ref_count() const { return inst_->ref_count_.load(); }

    // Manually retain the event. Must be matched by a call to release().
    void manual_retain() { inst_->ref_count_.fetch_add(1); }
    void manual_release() {
      if (inst_->ref_count_.fetch_sub(1) == 1) {
        delete inst_;
      }
    }

   private:
    explicit ref(iree::shared_event *inst) : inst_(inst) {}
    shared_event *inst_ = nullptr;
    friend class iree::shared_event;
  };

  static ref create(bool initial_state) {
    return ref(new shared_event(initial_state));
  }

 private:
  using event::event;
  ~shared_event() = default;

  std::atomic<int> ref_count_{1};
};

}  // namespace shortfin::iree

#endif  // SHORTFIN_SUPPORT_IREE_THREADING_H
