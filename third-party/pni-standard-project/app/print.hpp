#pragma once

#if defined(_WIN32) || defined(_WIN64)
#define NO_ANSI_ESCAPE_SEQUENCES
#endif

#if defined(__CYGWIN__)
#undef __STRICT_ANSI__
#include <iostream>
#include <sstream>
#define __STRICT_ANSI__
#else
#include <iostream>
#include <sstream>
#endif
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <termcolor/termcolor.hpp>

enum LogLevel { DEBUG, INFO, WARNING, ERROR, CRITICAL, __SILENT };

namespace pni_log {

// 输出从程序启动后的时间戳
inline std::string time(
    bool realOut = true) {
  using namespace std::chrono;
  static const auto clockStart = steady_clock::now();
  const auto clockNow = steady_clock::now();
  const auto timeDuration = clockNow - clockStart;
  std::stringstream ss;
  if (realOut) {
    ss << "[" << std::setw(3) << std::setfill('0') << duration_cast<seconds>(timeDuration).count() << "."
       << std::setw(3) << std::setfill('0') << (duration_cast<milliseconds>(timeDuration).count() % 1000) << "]";
  }
  return ss.str();
}

// 日志级别定义

// 从字符串到日志级别的映射
inline const std::map<std::string, LogLevel> mapLogLevels = {
    {"DEBUG", LogLevel::DEBUG}, {"INFO", LogLevel::INFO},         {"WARNING", LogLevel::WARNING},
    {"ERROR", LogLevel::ERROR}, {"CRITICAL", LogLevel::CRITICAL}, {"SILENT", LogLevel::__SILENT}};

template <typename Stream>
class PrintHandler {
  struct _Helper {
    _Helper(
        PrintHandler &handler)
        : m_handler(handler)
        , m_lock(handler.m_mutex) {
      m_handler.m_stream << time();
    }
    ~_Helper() { m_handler.m_stream << termcolor::reset; }
    _Helper &say(
        std::string msg) {
      if (!m_blockOnce) {
        m_handler.m_stream << msg;
      }
      m_blockOnce = false;
      return *this;
    }
    _Helper &endline() {
      if (!m_blockOnce) {
        m_handler.m_stream << std::endl;
      }
      m_blockOnce = false;
      return *this;
    }
    _Helper &error() {
      if (!m_blockOnce) {
        m_handler.m_stream << termcolor::red << termcolor::bold;
      }
      m_blockOnce = false;
      return *this;
    }
    _Helper &warning() {
      if (!m_blockOnce) {
        m_handler.m_stream << termcolor::yellow;
      }
      m_blockOnce = false;
      return *this;
    }
    _Helper &success() {
      if (!m_blockOnce) {
        m_handler.m_stream << termcolor::green;
      }
      m_blockOnce = false;
      return *this;
    }
    _Helper &reset() {
      if (!m_blockOnce) {
        m_handler.m_stream << termcolor::reset;
      }
      m_blockOnce = false;
      return *this;
    }
    _Helper &level(
        LogLevel l) {
      if (!m_blockOnce) {
        switch (l) {
        case LogLevel::DEBUG:
          m_handler.m_stream << "[" << termcolor::grey << "DEBUG" << termcolor::reset << "]";
          break;
        case LogLevel::INFO:
          m_handler.m_stream << "[INFO]";
          break;
        case LogLevel::WARNING:
          m_handler.m_stream << "[" << termcolor::yellow << "WARNING" << termcolor::reset << "]";
          break;
        case LogLevel::ERROR:
          m_handler.m_stream << "[" << termcolor::red << "ERROR" << termcolor::reset << "]";
          break;
        case LogLevel::CRITICAL:
          m_handler.m_stream << "[" << termcolor::red << termcolor::bold << "CRITICAL" << termcolor::reset << "]";
          break;
        default:
          break;
        }
      }
      m_blockOnce = true;
      return *this;
    }
    _Helper &block(
        bool doBlock) {
      m_blockOnce = doBlock;
      return *this;
    }

  private:
    PrintHandler &m_handler;
    std::lock_guard<std::mutex> m_lock;
    bool m_blockOnce = false; // Set false every use.
  };

public:
  PrintHandler(
      Stream &stream)
      : m_stream(stream) {}

  _Helper print() { return _Helper(*this); }

private:
  Stream &m_stream;
  std::mutex m_mutex;
};

} // namespace pni_log