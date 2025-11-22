#include "include/experimental/node/BDM2R2S.hpp"

#include <format>
#include <iostream>
#include <optional>

#include "impl/BDM2.hpp"
#include "include/Exceptions.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "src/common/Debug.h"

using namespace openpni::device::bdm2;
namespace openpni::experimental::node {
using impl::energy_coef_span;
using impl::position_table_span;

class BDM2R2S_impl {
public:
  BDM2R2S_impl();
  ~BDM2R2S_impl();

public:
  void setChannelIndex(uint16_t channelIndex) noexcept;
  void loadCalibration(std::string filePath);
  bool isCalibrationLoaded() const noexcept { return m_calibrationLoaded; }

public:
  std::span<interface::LocalSingle const> r2s_cpu(interface::PacketsInfo h_packets);
  std::span<interface::LocalSingle const> r2s_cuda(interface::PacketsInfo d_packets);
  impl::BDM2ConvertSingleContext DContext();
  impl::BDM2ConvertSingleContext HContext();
  uint16_t getChannelIndex() const {
    if (!m_channelIndex)
      throw exceptions::algorithm_unexpected_condition("BDM2R2S: Channel index is not set.");
    return *m_channelIndex;
  }

private:
  void checkHTable();
  void checkDTable();

private:
  std::optional<uint16_t> m_channelIndex;
  bool m_calibrationLoaded{false};
  bool m_calibrationInCUDA{false};
  std::vector<float> mh_energyCoef;
  std::vector<uint8_t> mh_positionTable;
  cuda_sync_ptr<float> md_energyCoef{"BDM2R2S_energyCoef"};
  cuda_sync_ptr<uint8_t> md_positionTable{"BDM2R2S_positionTable"};

  std::vector<uint32_t> mh_packetInclusiveSum;
  std::vector<interface::LocalSingle> mh_outputBuffer;
  cuda_sync_ptr<uint32_t> md_packetInclusiveSum{"BDM2R2S_packetInclusiveSum"};
  cuda_sync_ptr<interface::LocalSingle> md_outputBuffer{"BDM2R2S_outputBuffer"};
};

BDM2R2S_impl::BDM2R2S_impl() {}
BDM2R2S_impl::~BDM2R2S_impl() {}
void BDM2R2S_impl::setChannelIndex(
    uint16_t channelIndex) noexcept {
  m_channelIndex = channelIndex;
}
void BDM2R2S_impl::loadCalibration(
    std::string filePath) {
  auto &&[e, p] = impl::read_bdm2_calibration_table(filePath);
  mh_energyCoef.resize(energy_coef_span().totalSize());
  mh_positionTable.resize(position_table_span().totalSize());
  for (const auto [crystal, block] : energy_coef_span())
    mh_energyCoef[energy_coef_span()(crystal, block)] = (*e)[block][crystal];
  for (const auto [posx, posy, block] : position_table_span())
    mh_positionTable[position_table_span()(posx, posy, block)] = (*p)[block][posx + CRYSTAL_RAW_POSITION_RANGE * posy];
  m_calibrationLoaded = true;
}
std::span<interface::LocalSingle const> BDM2R2S_impl::r2s_cpu(
    interface::PacketsInfo h_packets) {
  if (h_packets.count == 0)
    return {};
  if (!m_channelIndex)
    throw exceptions::algorithm_unexpected_condition("BDM2R2S: Channel index is not set.");
  checkHTable();
  mh_packetInclusiveSum.resize(h_packets.count);
  example::h_inclusive_sum_equals(h_packets.channel, *m_channelIndex, mh_packetInclusiveSum.data(), h_packets.count);

  auto totalPackets = mh_packetInclusiveSum.empty() ? 0 : mh_packetInclusiveSum.back();
  if (mh_outputBuffer.size() < totalPackets * SINGLE_NUM_PER_PACKET)
    mh_outputBuffer.resize(totalPackets * SINGLE_NUM_PER_PACKET);
  impl::h_r2s(h_packets.raw, h_packets.offset, h_packets.length, *m_channelIndex, HContext(), mh_outputBuffer.data(),
              h_packets.count, mh_packetInclusiveSum.data());
  return md_outputBuffer.cspan(totalPackets * SINGLE_NUM_PER_PACKET);
}

std::span<interface::LocalSingle const> BDM2R2S_impl::r2s_cuda(
    interface::PacketsInfo d_packets) {
  if (d_packets.count == 0)
    return {};
  if (!m_channelIndex)
    throw exceptions::algorithm_unexpected_condition("BDM2R2S: Channel index is not set.");
  checkDTable();
  md_packetInclusiveSum.reserve(d_packets.count);
  example::d_inclusive_sum_equals(d_packets.channel, *m_channelIndex, md_packetInclusiveSum.get(), d_packets.count);

  auto totalPackets = md_packetInclusiveSum.at(d_packets.count - 1);
  md_outputBuffer.reserve(totalPackets * SINGLE_NUM_PER_PACKET);
  impl::d_r2s(d_packets.raw, d_packets.offset, d_packets.length, *m_channelIndex, DContext(), md_outputBuffer.get(),
              d_packets.count, md_packetInclusiveSum.get());
  return md_outputBuffer.cspan(totalPackets * SINGLE_NUM_PER_PACKET);
}

void BDM2R2S_impl::checkHTable() {
  if (m_calibrationLoaded)
    return;
  throw exceptions::algorithm_unexpected_condition("BDM2R2S: Calibration table is not loaded in host.");
}

void BDM2R2S_impl::checkDTable() {
  checkHTable();
  if (m_calibrationInCUDA)
    return;
  md_energyCoef = openpni::make_cuda_sync_ptr_from_hcopy(mh_energyCoef, "BDM2R2S_energyCoef");
  md_positionTable = openpni::make_cuda_sync_ptr_from_hcopy(mh_positionTable, "BDM2R2S_positionTable");
  m_calibrationInCUDA = true;
}

impl::BDM2ConvertSingleContext BDM2R2S_impl::DContext() {
  checkDTable();
  return impl::BDM2ConvertSingleContext{md_energyCoef.get(), md_positionTable.get()};
}

impl::BDM2ConvertSingleContext BDM2R2S_impl::HContext() {
  checkHTable();
  return impl::BDM2ConvertSingleContext{mh_energyCoef.data(), mh_positionTable.data()};
}

BDM2R2S::BDM2R2S()
    : m_impl(std::make_unique<BDM2R2S_impl>()) {}

BDM2R2S::BDM2R2S(
    std::unique_ptr<BDM2R2S_impl> impl)
    : m_impl(std::move(impl)) {}

BDM2R2S::~BDM2R2S() {}
void BDM2R2S::setChannelIndex(
    uint16_t channelIndex) noexcept {
  m_impl->setChannelIndex(channelIndex);
}
void BDM2R2S::loadCalibration(
    std::string filePath) {
  m_impl->loadCalibration(filePath);
}
std::span<interface::LocalSingle const> BDM2R2S::r2s_cpu(
    interface::PacketsInfo h_packets) const {
  return m_impl->r2s_cpu(h_packets);
}
std::span<interface::LocalSingle const> BDM2R2S::r2s_cuda(
    interface::PacketsInfo d_packets) const {
  return m_impl->r2s_cuda(d_packets);
}

} // namespace openpni::experimental::node
namespace openpni::experimental::node {
class BDM2R2SArray_impl {
public:
  BDM2R2SArray_impl();
  ~BDM2R2SArray_impl();

public:
  bool addSingleGenerator(interface::SingleGenerator *generator) noexcept;
  void clearSingleGenerators() noexcept;

public:
  std::span<interface::LocalSingle const> r2s_cpu(interface::PacketsInfo h_packets);
  std::span<interface::LocalSingle const> r2s_cuda(interface::PacketsInfo d_packets);

private:
  void checkHConvergence();
  void checkDConvergence();

private:
  std::vector<BDM2R2S *> m_generators;
  std::vector<impl::BDM2ConvertSingleContext> m_hContexts;
  cuda_sync_ptr<impl::BDM2ConvertSingleContext> m_dContexts;
  std::vector<uint16_t> mh_channelContextMap;
  std::vector<uint16_t> mh_channelIndicesValue;
  cuda_sync_ptr<uint16_t> md_channelContextmap{"BDM2R2SArray_channelContextmap"};
  cuda_sync_ptr<uint16_t> md_channelIndicesValue{"BDM2R2SArray_channelIndicesValue"};
  bool m_convergedInCPU{false};
  bool m_convergedInCUDA{false};

  std::vector<uint32_t> mh_packetInclusiveSum;
  std::vector<interface::LocalSingle> mh_outputBuffer;
  cuda_sync_ptr<uint32_t> md_packetInclusiveSum{"BDM2R2SArray_packetInclusiveSum"};
  cuda_sync_ptr<interface::LocalSingle> md_outputBuffer{"BDM2R2SArray_outputBuffer"};
};
BDM2R2SArray_impl::BDM2R2SArray_impl() {}
BDM2R2SArray_impl::~BDM2R2SArray_impl() {}
bool BDM2R2SArray_impl::addSingleGenerator(
    interface::SingleGenerator *generator) noexcept {
  auto *bdm2r2s = dynamic_cast<BDM2R2S *>(generator);
  if (!bdm2r2s)
    return false;
  m_generators.push_back(bdm2r2s);
  m_convergedInCPU = false;
  m_convergedInCUDA = false;
  return true;
}
void BDM2R2SArray_impl::clearSingleGenerators() noexcept {
  m_generators.clear();
}
void BDM2R2SArray_impl::checkHConvergence() {
  if (m_convergedInCPU)
    return;
  m_hContexts.clear();
  for (auto *gen : m_generators)
    m_hContexts.push_back(gen->m_impl->HContext());
  mh_channelContextMap.clear();
  mh_channelContextMap.resize(1 << 16, 0xFFFF);
  for (int i = 0; i < m_generators.size(); ++i)
    mh_channelContextMap[m_generators[i]->m_impl->getChannelIndex()] = i;
  mh_channelIndicesValue.clear();
  for (int i = 0; i < m_generators.size(); ++i)
    mh_channelIndicesValue.push_back(m_generators[i]->m_impl->getChannelIndex());
  m_convergedInCPU = true;
}
void BDM2R2SArray_impl::checkDConvergence() {
  checkHConvergence();
  if (m_convergedInCUDA)
    return;
  m_dContexts.clear();
  std::vector<impl::BDM2ConvertSingleContext> tempContexts;
  for (auto *gen : m_generators)
    tempContexts.push_back(gen->m_impl->DContext());
  md_channelContextmap =
      openpni::make_cuda_sync_ptr_from_hcopy<uint16_t>(mh_channelContextMap, "BDM2R2SArray_channelContextMap");
  md_channelIndicesValue =
      openpni::make_cuda_sync_ptr_from_hcopy<uint16_t>(mh_channelIndicesValue, "BDM2R2SArray_channelIndicesValue");
  m_dContexts =
      openpni::make_cuda_sync_ptr_from_hcopy<impl::BDM2ConvertSingleContext>(tempContexts, "BDM2R2SArray_contexts");
  m_convergedInCUDA = true;
}

std::span<interface::LocalSingle const> BDM2R2SArray_impl::r2s_cpu(
    interface::PacketsInfo h_packets) {
  if (h_packets.count == 0)
    return {};
  if (m_generators.empty())
    return {};
  checkHConvergence();
  mh_packetInclusiveSum.resize(h_packets.count);
  example::h_inclusive_sum_any_of(h_packets.channel, mh_channelIndicesValue.data(), mh_channelIndicesValue.size(),
                                  mh_packetInclusiveSum.data(), h_packets.count);
  auto totalPackets = mh_packetInclusiveSum.empty() ? 0 : mh_packetInclusiveSum.back();
  if (totalPackets == 0)
    return {};
  if (mh_outputBuffer.size() < totalPackets * SINGLE_NUM_PER_PACKET)
    mh_outputBuffer.resize(totalPackets * SINGLE_NUM_PER_PACKET);
  impl::h_r2s_converged(h_packets.raw, h_packets.offset, h_packets.length, h_packets.channel, m_hContexts.data(),
                        mh_channelContextMap.data(), mh_outputBuffer.data(), h_packets.count,
                        mh_packetInclusiveSum.data());
  return std::span<interface::LocalSingle const>(mh_outputBuffer.data(), totalPackets * SINGLE_NUM_PER_PACKET);
}
std::span<interface::LocalSingle const> BDM2R2SArray_impl::r2s_cuda(
    interface::PacketsInfo d_packets) {
  if (d_packets.count == 0)
    return {};
  if (m_generators.empty())
    return {};
  checkDConvergence();
  PNI_DEBUG(std::format("BDM2R2SArray: total count: {}\n", d_packets.count));
  md_packetInclusiveSum.reserve(d_packets.count);
  example::d_inclusive_sum_any_of(d_packets.channel, md_channelIndicesValue.get(), md_channelIndicesValue.elements(),
                                  md_packetInclusiveSum.get(), d_packets.count);
  PNI_DEBUG(std::format("BDM2R2SArray: channelIndices size {}, packetInclusiveSum: {}\n",
                        md_channelContextmap.elements(), md_packetInclusiveSum.elements()));
  auto totalPackets = md_packetInclusiveSum.at(d_packets.count - 1);
  // debug::d_print_value_at_index(md_packetInclusiveSum.get(), 0,std::min(1000ul ,md_packetInclusiveSum.elements()));
  if (totalPackets == 0)
    return {};
  PNI_DEBUG(std::format("BDM2R2SArray: {}, total packets {}\n", __LINE__, totalPackets));
  md_outputBuffer.reserve(totalPackets * SINGLE_NUM_PER_PACKET);
  impl::d_r2s_converged(d_packets.raw, d_packets.offset, d_packets.length, d_packets.channel, m_dContexts.get(),
                        md_channelContextmap.get(), md_outputBuffer.get(), d_packets.count,
                        md_packetInclusiveSum.get());
  return md_outputBuffer.cspan(totalPackets * SINGLE_NUM_PER_PACKET);
}
} // namespace openpni::experimental::node

namespace openpni::experimental::node {
BDM2R2SArray::BDM2R2SArray()
    : m_impl(std::make_unique<BDM2R2SArray_impl>()) {}
BDM2R2SArray::~BDM2R2SArray() {}
bool BDM2R2SArray::addSingleGenerator(
    interface::SingleGenerator *generator) noexcept {
  return m_impl->addSingleGenerator(generator);
}
void BDM2R2SArray::clearSingleGenerators() noexcept {
  m_impl->clearSingleGenerators();
}
std::span<interface::LocalSingle const> BDM2R2SArray::r2s_cpu(
    interface::PacketsInfo h_packets) const {
  return m_impl->r2s_cpu(h_packets);
}
std::span<interface::LocalSingle const> BDM2R2SArray::r2s_cuda(
    interface::PacketsInfo d_packets) const {
  return m_impl->r2s_cuda(d_packets);
}

core::DetectorGeom BDM2R2S::geom() {
  core::DetectorGeom result;
  result.blockNumU = BLOCK_NUM;
  result.blockNumV = 1;
  result.blockSizeU = BLOCK_PITCH;
  result.blockSizeV = BLOCK_PITCH;
  result.crystalNumU = CRYSTAL_LINE;
  result.crystalNumV = CRYSTAL_LINE;
  result.crystalSizeU = CRYSTAL_PITCH;
  result.crystalSizeV = CRYSTAL_PITCH;
  return result;
}
core::Vector<uint16_t, 2> BDM2R2S::udpPacketLengthRange() {
  return core::Vector<uint16_t, 2>{MIN_UDP_PACKET_SIZE, MAX_UDP_PACKET_SIZE};
}
bool BDM2R2S::isCalibrationLoaded() const noexcept {
  return m_impl->isCalibrationLoaded();
}
} // namespace openpni::experimental::node