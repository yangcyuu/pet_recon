#pragma once
#include <algorithm>
#include <atomic>
#include <ranges>
#include <thread>
#include <vector>

#include "../core/BasicMath.hpp"
#include "../core/Span.hpp"
namespace openpni::experimental::tools {
enum CPUThreadNum { MAX_THREAD, HALF_THREAD, QUARTER_THREAD, SINGLE_THREAD };
enum CPUScheduleType { STATIC, DYNAMIC };
struct CpuMultiThread {
  int threadNum() const { return std::max<int>(1, m_threadNum); }
  CPUScheduleType scheduleType() const { return m_scheduleType; }
  int scheduleNum() const { return std::max<int>(1, m_scheduleNum); }
  CpuMultiThread &setThreadNum(
      int num) {
    m_threadNum = std::clamp<int>(num, 1, std::thread::hardware_concurrency());
    return *this;
  }
  CpuMultiThread &setThreadNumType(
      CPUThreadNum num) {
    switch (num) {
    case CPUThreadNum::MAX_THREAD:
      return setThreadNum(std::thread::hardware_concurrency());
    case CPUThreadNum::HALF_THREAD:
      return setThreadNum(std::max<int>(1, std::thread::hardware_concurrency() / 2));
    case CPUThreadNum::QUARTER_THREAD:
      return setThreadNum(std::max<int>(1, std::thread::hardware_concurrency() / 4));
    case CPUThreadNum::SINGLE_THREAD:
      return setThreadNum(1);
    }
    return setThreadNum(1);
  }
  CpuMultiThread &setScheduleType(
      CPUScheduleType type) {
    m_scheduleType = type;
    return *this;
  }
  CpuMultiThread &setScheduleNum(
      int num) {
    m_scheduleNum = std::max(num, 1);
    return *this;
  }

private:
  int m_threadNum{1};
  int m_scheduleNum{1};
  CPUScheduleType m_scheduleType{CPUScheduleType::STATIC};
};
inline CpuMultiThread &cpu_threads() {
  thread_local static CpuMultiThread instance;
  return instance;
}
} // namespace openpni::experimental::tools
namespace openpni::experimental::tools {
namespace impl {
template <typename Func>
inline void for_each_static_impl(
    std::size_t __start, std::size_t __step, std::size_t __max, const Func &__func) {
  for (std::size_t i = __start; i < __max; i += __step)
    __func(i);
}
template <typename Func>
inline void for_each_dynamic_impl(
    std::atomic<std::size_t> &__index, std::size_t __max, std::size_t __scheduleSize, const Func &__func) {
  while (true) {
    const auto index = __index.fetch_add(__scheduleSize);
    for (std::size_t i = index; i < index + __scheduleSize; i++)
      if (i < __max)
        __func(i);
      else
        return;
  }
}
} // namespace impl

template <typename Func>
inline void parallel_for_each(
    std::size_t __max, Func &&__func) {
  if (__max <= 0)
    return;
  if (cpu_threads().threadNum() <= 1) {
    const auto iota = std::views::iota(std::size_t{0}, __max);
    std::for_each(iota.begin(), iota.end(), std::forward<Func>(__func));
    return;
  }

  if (cpu_threads().scheduleType() == CPUScheduleType::STATIC) {
    std::vector<std::jthread> threads;
    const auto realThreadNum = std::min<std::size_t>(cpu_threads().threadNum(), __max);
    for (const auto threadIndex : std::views::iota(0ull, realThreadNum))
      threads.emplace_back([&, threadIndex] { impl::for_each_static_impl(threadIndex, realThreadNum, __max, __func); });
  } else {
    std::vector<std::jthread> threads;
    const auto scheduleSize = cpu_threads().scheduleNum(); // >=1 is assumed
    const auto realThreadNum =
        std::min(core::dev_ceil<std::size_t>(__max, scheduleSize), std::size_t(cpu_threads().threadNum()));
    std::atomic<std::size_t> index{0};
    for (const auto threadIndex : std::views::iota(0ull, realThreadNum))
      threads.emplace_back([&] { impl::for_each_dynamic_impl(index, __max, scheduleSize, __func); });
  }
}
template <typename Func>
inline void parallel_for_each(
    std::size_t __begin, std::size_t __end, Func &&__func) {
  if (__end <= __begin)
    return;
  parallel_for_each(__end - __begin, [&](std::size_t index) { __func(index + __begin); });
}
} // namespace openpni::experimental::tools
// 为多种情景适配的并行for_each
namespace openpni::experimental::tools {
template <int N, typename Func>
inline void parallel_for_each(
    core::MDSpan<N> span, Func &&func) {
  parallel_for_each(span.totalSize(), [&](std::size_t index) { func(span.toIndex(index)); });
}
template <int N, typename Func>
inline void parallel_for_each(
    core::MDBeginEndSpan<N> span, Func &&func) {
  parallel_for_each(span.totalSize(), [&](std::size_t index) { func(span.toIndex(index)); });
}

} // namespace openpni::experimental::tools
