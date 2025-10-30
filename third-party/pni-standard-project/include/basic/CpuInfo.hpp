#pragma once
#include <thread>
namespace openpni::basic {
struct CpuMultiThread {
  enum class ThreadNumType { MAX_THREAD, HALF_THREAD, QUARTER_THREAD, SINGLE_THREAD };
  enum class ScheduleType { Static, Dynamic };

  explicit CpuMultiThread(
      ThreadNumType threadType, ScheduleType scheduleType = ScheduleType::Static, int scheduleNum = 64)
      : m_threadNum(0)
      , m_scheduleType(scheduleType)
      , m_scheduleNum(std::min(scheduleNum, 1)) {
    switch (threadType) {
    case ThreadNumType::MAX_THREAD:
      m_threadNum = std::thread::hardware_concurrency();
      break;
    case ThreadNumType::HALF_THREAD:
      m_threadNum = std::max<int>(1, std::thread::hardware_concurrency() / 2);
      break;
    case ThreadNumType::QUARTER_THREAD:
      m_threadNum = std::max<int>(1, std::thread::hardware_concurrency() / 4);
      break;
    case ThreadNumType::SINGLE_THREAD:
      m_threadNum = 1;
      break;
    }
  }
  explicit CpuMultiThread(
      int threadNum, ScheduleType scheduleType = ScheduleType::Static, int scheduleNum = 64)
      : m_threadNum(std::max<int>(1, threadNum))
      , m_scheduleType(scheduleType)
      , m_scheduleNum(std::min(scheduleNum, 1)) {}

  static CpuMultiThread callWithAllThreads(
      ScheduleType scheduleType = ScheduleType::Static, int scheduleNum = 64) {
    return CpuMultiThread(ThreadNumType::MAX_THREAD, scheduleType, scheduleNum);
  }
  static CpuMultiThread callWithHalfThreads(
      ScheduleType scheduleType = ScheduleType::Static, int scheduleNum = 64) {
    return CpuMultiThread(ThreadNumType::HALF_THREAD, scheduleType, scheduleNum);
  }
  static CpuMultiThread callWithQuarterThreads(
      ScheduleType scheduleType = ScheduleType::Static, int scheduleNum = 64) {
    return CpuMultiThread(ThreadNumType::QUARTER_THREAD, scheduleType, scheduleNum);
  }
  static CpuMultiThread callWithSingleThread(
      ScheduleType scheduleType = ScheduleType::Static, int scheduleNum = 64) {
    return CpuMultiThread(ThreadNumType::SINGLE_THREAD, scheduleType, scheduleNum);
  }
  static CpuMultiThread callWithSomeThreads(
      int threadNum, ScheduleType scheduleType = ScheduleType::Static, int scheduleNum = 64) {
    return CpuMultiThread(threadNum, scheduleType, scheduleNum);
  }

  int threadNum() const { return std::max<int>(1, m_threadNum); }
  ScheduleType scheduleType() const { return m_scheduleType; }
  int scheduleNum() const { return std::max<int>(1, m_scheduleNum); }

private:
  int m_threadNum;
  int m_scheduleNum;
  ScheduleType m_scheduleType;
};
} // namespace openpni::basic
namespace openpni {
struct _CPUThreads {
  auto multiThreads() const { return basic::CpuMultiThread::callWithAllThreads(); }
  auto halfThreads() const { return basic::CpuMultiThread::callWithHalfThreads(); }
  auto quarterThreads() const { return basic::CpuMultiThread::callWithQuarterThreads(); }
  auto singleThread() const { return basic::CpuMultiThread::callWithSingleThread(); }
};
inline constexpr _CPUThreads cpu_threads{};
} // namespace openpni
