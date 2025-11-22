#pragma once
#include "../interface/SingleGenerator.hpp"
#include "../../basic/CudaPtr.hpp"
namespace openpni::experimental::node {
struct DPackets {
  cuda_sync_ptr<uint8_t> raw;
  cuda_sync_ptr<uint64_t> offset;
  cuda_sync_ptr<uint16_t> length;
  cuda_sync_ptr<uint16_t> channel;
  uint64_t count;
  static DPackets fromHost(uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length,
                           uint16_t const *h_channel, uint64_t h_count);
  static DPackets fromHost(interface::PacketsInfo h_packets);
};

class ConvergedR2S_impl;
class ConvergedR2S {
public:
  ConvergedR2S();
  ConvergedR2S(std::unique_ptr<ConvergedR2S_impl> impl);
  ~ConvergedR2S();

public:
  using R2SResult = std::vector<std::span<interface::LocalSingle const>>;

public:
  void setChannels(std::vector<interface::SingleGenerator *> const &channels);
  R2SResult r2s_cpu(uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length, uint16_t const *h_channel,
                    uint64_t h_count) const noexcept;
  R2SResult 
  r2s_cuda(uint8_t const *d_raw, uint64_t const *d_offset, uint16_t const *d_length, uint16_t const *d_channel,
           uint64_t d_count) const noexcept;

private:
  std::unique_ptr<ConvergedR2S_impl> m_impl;
};
} // namespace openpni::experimental::node
