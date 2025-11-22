#include "include/experimental/node/ConvergedR2S.hpp"

#include <basic/CudaPtr.hpp>
#include <iostream>

#include "include/Exceptions.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/node/BDM2R2S.hpp"
using R2SResult = openpni::experimental::node::ConvergedR2S::R2SResult;
namespace openpni::experimental::node {
template <typename Tuple>
struct make_unique_ptr_tuple;

template <typename... Ts>
struct make_unique_ptr_tuple<std::tuple<Ts...>> {
  using type = std::tuple<std::unique_ptr<Ts>...>;
  static auto create() { return std::tuple<std::unique_ptr<Ts>...>(std::make_unique<Ts>()...); }
};

template <typename Tuple>
using make_unique_ptr_tuple_t = typename make_unique_ptr_tuple<Tuple>::type;

using supported_generator_array_types =
    std::tuple<openpni::experimental::node::BDM2R2SArray>; // 如果有更多类型的generator，添加到这里即可
using supported_generator_array_ptr_types = make_unique_ptr_tuple_t<supported_generator_array_types>;

auto make_supported_generator_array_ptr_types() {
  return make_unique_ptr_tuple<supported_generator_array_types>::create();
}

void insert_generator(
    interface::SingleGenerator *generator, supported_generator_array_ptr_types &generatorArrays) {
  // If the tuple is empty, throw an error
  if constexpr (std::tuple_size_v<supported_generator_array_ptr_types> == 0)
    throw exceptions::algorithm_unexpected_condition("ConvergedR2S: Not implemented.");

  bool inserted = false;
  std::apply(
      [&](auto &&...args) noexcept {
        (
            [&] {
              if (inserted)
                return;
              if (args->addSingleGenerator(generator))
                inserted = true;
            }(),
            ...);
      },
      generatorArrays);
  if (!inserted)
    throw exceptions::algorithm_unexpected_condition("ConvergedR2S: There is no suitable generator.");
}
void clear_generator_arrays(
    supported_generator_array_ptr_types &generatorArrays) noexcept {
  std::apply([&](auto &&...args) noexcept { ([&] { args->clearSingleGenerators(); }(), ...); }, generatorArrays);
}

class ConvergedR2S_impl {
public:
  ConvergedR2S_impl()
      : m_generatorArrays(make_supported_generator_array_ptr_types()) {}
  ~ConvergedR2S_impl() = default;

public:
  void setChannels(std::vector<interface::SingleGenerator *> const &channels);
  R2SResult r2s_cpu(uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length, uint16_t const *h_channel,
                    uint64_t h_count) const noexcept;
  R2SResult r2s_cuda(uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length,
                     uint16_t const *h_channel, uint64_t h_count) const noexcept;

private:
  supported_generator_array_ptr_types m_generatorArrays;
};

void ConvergedR2S_impl::setChannels(
    std::vector<interface::SingleGenerator *> const &channels) {
  clear_generator_arrays(m_generatorArrays);
  for (auto *generator : channels)
    insert_generator(generator, m_generatorArrays);
}
R2SResult ConvergedR2S_impl::r2s_cpu(
    uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length, uint16_t const *h_channel,
    uint64_t h_count) const noexcept {
  R2SResult results;
  interface::PacketsInfo h_packets{h_raw, h_offset, h_length, h_channel, h_count};
  std::apply([&](auto &&...args) noexcept { ([&] { results.push_back(args->r2s_cpu(h_packets)); }(), ...); },
             m_generatorArrays);
  return results;
}
R2SResult ConvergedR2S_impl::r2s_cuda(
    uint8_t const *d_raw, uint64_t const *d_offset, uint16_t const *d_length, uint16_t const *d_channel,
    uint64_t count) const noexcept {
  R2SResult results;
  if (count == 0)
    return results;

  // 在设备上构造一个 PacketsInfo，其指针字段指向上面拷贝到设备的缓冲区
  interface::PacketsInfo d_packets{d_raw, d_offset, d_length, d_channel, count};
  std::apply([&](auto const &...args) noexcept { ([&] { results.push_back(args->r2s_cuda(d_packets)); }(), ...); },
             m_generatorArrays);
  return results;
}

ConvergedR2S::ConvergedR2S(
    std::unique_ptr<ConvergedR2S_impl> impl)
    : m_impl(std::move(impl)) {}
ConvergedR2S::ConvergedR2S()
    : m_impl(std::make_unique<ConvergedR2S_impl>()) {}
ConvergedR2S::~ConvergedR2S() {}
void ConvergedR2S::setChannels(
    std::vector<interface::SingleGenerator *> const &channels) {
  m_impl->setChannels(channels);
}
ConvergedR2S::R2SResult ConvergedR2S::r2s_cpu(
    uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length, uint16_t const *h_channel,
    uint64_t h_count) const noexcept {
  return m_impl->r2s_cpu(h_raw, h_offset, h_length, h_channel, h_count);
}
ConvergedR2S::R2SResult ConvergedR2S::r2s_cuda(
    uint8_t const *d_raw, uint64_t const *d_offset, uint16_t const *d_length, uint16_t const *d_channel,
    uint64_t count) const noexcept {
  return m_impl->r2s_cuda(d_raw, d_offset, d_length, d_channel, count);
}

DPackets DPackets::fromHost(
    uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length, uint16_t const *h_channel,
    uint64_t h_count) {
  DPackets d_packets;
  d_packets.count = h_count;
  d_packets.raw = openpni::make_cuda_sync_ptr_from_hcopy(
      std::span<const uint8_t>(h_raw, static_cast<size_t>(h_offset[h_count - 1] + h_length[h_count - 1] - h_offset[0])),
      "DPackets_raw");
  d_packets.offset = openpni::make_cuda_sync_ptr_from_hcopy(
      std::span<const uint64_t>(h_offset, static_cast<size_t>(h_count)), "DPackets_offset");
  if (h_offset[0] != 0)
    example::d_parallel_sub(d_packets.offset, h_offset[0], d_packets.offset, d_packets.offset.elements());
  d_packets.length = openpni::make_cuda_sync_ptr_from_hcopy(
      std::span<const uint16_t>(h_length, static_cast<size_t>(h_count)), "DPackets_length");
  d_packets.channel = openpni::make_cuda_sync_ptr_from_hcopy(
      std::span<const uint16_t>(h_channel, static_cast<size_t>(h_count)), "DPackets_channel");
  return d_packets;
}

DPackets DPackets::fromHost(
    interface::PacketsInfo h_packets) {
  return fromHost(h_packets.raw, h_packets.offset, h_packets.length, h_packets.channel, h_packets.count);
}

} // namespace openpni::experimental::node
