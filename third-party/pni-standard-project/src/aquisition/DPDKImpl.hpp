#pragma once
#ifndef _PNI_STD_AQUISITION_IMPL_DPDK_HPP_
#define _PNI_STD_AQUISITION_IMPL_DPDK_HPP_
#include "include/PnI-Config.hpp"
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
#include <map>
#include <mutex>
#include <thread>

#include "DPDKFunctions.hpp"
#include "IPSetting.hpp"
#include "TimedDataBuffer.hpp"
#include "include/process/Acquisition.hpp"

namespace openpni::process::dpdk {
class DPDK::DPDKImpl {
public:
  DPDKImpl(AcquisitionInfo, process::FuncLogger &&);
  virtual ~DPDKImpl() noexcept;
  DPDKImpl(const DPDKImpl &) = delete;
  DPDKImpl &operator=(const DPDKImpl &) = delete;

public:
  Status status() noexcept;
  bool start();
  bool stop() noexcept;
  bool isFinished() noexcept;
  std::optional<RawDataView> read() noexcept;

private:
  bool allocate();

private:
  bool writeToBuffer(WriteParam *);
  void removePacket(WriteParam *param);
  void threadMark();

private:
  struct AcquisitionLoop {
    bool r_dpdkSeized{false};
    bool r_finished{false};
    uint64_t r_lastReadingIndex{0};
    std::optional<RawDataView> r_lastReadResult;
    uint64_t r_lastReadoutCount{0};
    std::jthread r_threadMark;
    bool r_stopped{false};
    uint64_t r_unknownPkts{0};
  };

private:
  const AcquisitionInfo mc_param;
  std::unique_ptr<timed_buffer::Groups> m_buffer;
  bool m_status{true};
  process::FuncLogger m_logger;
  FuncWriteToMemory m_funcWriteToMemory;
  std::unique_ptr<AcquisitionLoop> m_runtime;

  std::function<uint16_t(IP, uint8_t *, uint16_t)> m_ipMapper;
};
} // namespace openpni::process::dpdk
#endif
#endif // !_PNI_STD_AQUISITION_IMPL_DPDK_HPP_
