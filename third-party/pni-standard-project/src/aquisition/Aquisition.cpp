#include "SocketImpl.hpp"
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
#include "DPDKImpl.hpp"
#endif
#include <cstring>

#include "include/process/Acquisition.hpp"
namespace openpni::process {

namespace socket {
Socket::Socket(
    AcquisitionInfo p, process::FuncLogger &&f) noexcept {
  impl = std::make_unique<SocketImpl>(p, std::move(f));
}

Socket::~Socket() noexcept {}

Status Socket::status() noexcept {
  return impl->status();
}

bool Socket::start() {
  return impl->start();
}

bool Socket::stop() noexcept {
  return impl->stop();
}

bool Socket::isFinished() noexcept {
  return impl->isFinished();
}

std::optional<RawDataView> Socket::read() noexcept {
  return impl->read();
}

} // namespace socket
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
namespace dpdk {
void InitDPDK() {};
DPDK::DPDK(
    AcquisitionInfo p, process::FuncLogger &&f) noexcept {
  impl = std::make_unique<DPDKImpl>(p, std::move(f));
}
DPDK::~DPDK() noexcept {}
Status DPDK::status() noexcept {
  return impl->status();
}
bool DPDK::start() {
  return impl->start();
}
bool DPDK::stop() noexcept {
  return impl->stop();
}
bool DPDK::isFinished() noexcept {
  return impl->isFinished();
}
std::optional<RawDataView> DPDK::read() noexcept {
  return impl->read();
}
} // namespace dpdk
#endif
} // namespace openpni::process
