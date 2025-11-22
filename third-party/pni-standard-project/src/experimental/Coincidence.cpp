#include "include/experimental/node/Coincidence.hpp"

#include <numeric>

#include "impl/Coin.h"
#include "include/basic/CudaPtr.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "src/common/Debug.h"

namespace openpni::experimental::node {
class Coincidence_impl {
public:
  Coincidence_impl() = default;
  ~Coincidence_impl() = default;

  void setCrystalNumOfEachChannel(std::vector<uint32_t> const &crystalNumPerChannel);
  Coincidence::CoinResult getDListmode(std::vector<std::span<interface::LocalSingle const>> d_coinListmodeLists,
                                       CoincidenceProtocol protocol);
  std::vector<unsigned> dumpCrystalCountMap() const { return make_vector_from_cuda_sync_ptr(md_crystalCountMap); }
  void clearCrystalCountMap() { md_crystalCountMap.clear(); }

private:
  void checkD();
  void checkH();

private:
  std::vector<unsigned> mh_crystalNumPerChannel;
  std::vector<unsigned> mh_crystalNumInclusiveSum;

  cuda_sync_ptr<unsigned> md_crystalNumInclusiveSum{"Coincidence_crystalNumInclusiveSum"};
  cuda_sync_ptr<unsigned> md_crystalCountMap{"Coincidence_crystalCountMap"};

  cuda_sync_ptr<interface::LocalSingle> md_localSingleBuffer{"Coincidence_localSingleBuffer"};
  cuda_sync_ptr<uint8_t> md_promptCount{"Coincidence_promptCount"};
  cuda_sync_ptr<uint8_t> md_delayCount{"Coincidence_delayCount"};
  cuda_sync_ptr<uint32_t> md_promptInclusiveSum{"Coincidence_promptInclusiveSum"};
  cuda_sync_ptr<uint32_t> md_delayInclusiveSum{"Coincidence_delayInclusiveSum"};
  cuda_sync_ptr<LocalListmode> md_promptListmode{"Coincidence_promptListmode"};
  cuda_sync_ptr<LocalListmode> md_delayListmode{"Coincidence_delayListmode"};
};
void Coincidence_impl::setCrystalNumOfEachChannel(
    std::vector<uint32_t> const &crystalNumPerChannel) {
  mh_crystalNumPerChannel = crystalNumPerChannel;
  mh_crystalNumInclusiveSum = {};
  md_crystalNumInclusiveSum.clear();
}

void Coincidence_impl::checkH() {
  if (mh_crystalNumInclusiveSum.size())
    return;

  if (mh_crystalNumPerChannel.empty())
    throw exceptions::algorithm_unexpected_condition("The crystal num of each channel is not specified.");

  mh_crystalNumInclusiveSum = mh_crystalNumPerChannel;
  std::inclusive_scan(mh_crystalNumPerChannel.begin(), mh_crystalNumPerChannel.end(),
                      mh_crystalNumInclusiveSum.begin());
}
void Coincidence_impl::checkD() {
  if (md_crystalNumInclusiveSum)
    return;

  checkH();
  md_crystalNumInclusiveSum = make_cuda_sync_ptr_from_hcopy(mh_crystalNumInclusiveSum, "Copy_inclusiveSum");
}
Coincidence::CoinResult Coincidence_impl::getDListmode(
    std::vector<std::span<interface::LocalSingle const>> d_coinListmodeLists, CoincidenceProtocol protocol) {
  checkD();
  // Join local single vectors into one global vector
  uint64_t totalSingleNum =
      std::accumulate(d_coinListmodeLists.begin(), d_coinListmodeLists.end(), 0ULL,
                      [](uint64_t sum, auto const &localSingle) { return sum + localSingle.size(); });
  PNI_DEBUG(std::format("There are {} singles in total.\n", totalSingleNum));
  if (totalSingleNum == 0)
    return {};

  if (md_crystalCountMap.elements() < mh_crystalNumInclusiveSum.back()) {
    md_crystalCountMap.reserve(mh_crystalNumInclusiveSum.back() * 4);
    md_crystalCountMap.allocator().memset(0, md_crystalCountMap.span());
  }

  md_localSingleBuffer.reserve(totalSingleNum * 2);
  for (uint64_t copiedSingleNum{0}; const auto localSingleGroup : d_coinListmodeLists) {
    impl::d_copy(md_localSingleBuffer.data() + copiedSingleNum, localSingleGroup.data(), localSingleGroup.size());
    copiedSingleNum += localSingleGroup.size();
  }

  // Count the number of singles in each channel [ Layer 1 ]
  impl::d_countCrystal(md_crystalCountMap.data(), md_localSingleBuffer.data(), totalSingleNum,
                       md_crystalNumInclusiveSum.data(), static_cast<unsigned>(mh_crystalNumPerChannel.size()));
#if PNI_STANDARD_CONFIG_ENABLE_DEBUG_THREAD_TIME
  PNI_DEBUG("-- After d_countCrystal --\n");
#endif

  // Do energy coincidence
  auto totalEnergyEvents = impl::d_energyFilter(md_localSingleBuffer.data(), totalSingleNum, protocol.energyLower_eV,
                                                protocol.energyUpper_eV);
#if PNI_STANDARD_CONFIG_ENABLE_DEBUG_THREAD_TIME
  PNI_DEBUG("-- After d_energyFilter --\n");
#endif
  PNI_DEBUG(std::format("There are {} singles after energy filter, energy ratio: {}%\n", totalEnergyEvents,
                        static_cast<float>(totalEnergyEvents) / static_cast<float>(totalSingleNum) * 100));

  // Count the number of selected singles after energy filter [ Layer 2 ]
  impl::d_countCrystal(md_crystalCountMap.data() + mh_crystalNumInclusiveSum.back(), md_localSingleBuffer.data(),
                       totalEnergyEvents, md_crystalNumInclusiveSum.data(),
                       static_cast<unsigned>(mh_crystalNumPerChannel.size()));

  // Double the buffer for coincidence output
  impl::d_doubleSinglesAndShiftTime(md_localSingleBuffer.data(), totalEnergyEvents,
                                    static_cast<uint64_t>(protocol.delayTime_ps));
#if PNI_STANDARD_CONFIG_ENABLE_DEBUG_THREAD_TIME
  PNI_DEBUG("-- After d_doubleSinglesAndShiftTime --\n");
#endif

  // Sort.
  impl::d_sortSinglesByTime(md_localSingleBuffer.data(), totalEnergyEvents * 2);
#if PNI_STANDARD_CONFIG_ENABLE_DEBUG_THREAD_TIME
  PNI_DEBUG("-- After d_sortSinglesByTime --\n");
#endif

  // Count prompt & delay coincidences in one pass
  md_promptCount.reserve(totalEnergyEvents * 2);
  md_delayCount.reserve(totalEnergyEvents * 2);
  md_promptInclusiveSum.reserve(totalEnergyEvents * 2);
  md_delayInclusiveSum.reserve(totalEnergyEvents * 2);
  impl::d_countCoincidences(md_localSingleBuffer.data(), totalEnergyEvents * 2, md_promptCount.data(),
                            md_delayCount.data(), protocol.timeWindow_ps);
#if PNI_STANDARD_CONFIG_ENABLE_DEBUG_THREAD_TIME
  PNI_DEBUG("-- After d_countCoincidences --\n");
#endif
  impl::d_inclusiveSumUint8(md_promptCount.data(), totalEnergyEvents * 2, md_promptInclusiveSum.data());
  impl::d_inclusiveSumUint8(md_delayCount.data(), totalEnergyEvents * 2, md_delayInclusiveSum.data());
#if PNI_STANDARD_CONFIG_ENABLE_DEBUG_THREAD_TIME
  PNI_DEBUG("-- After d_inclusiveSumUint8 --\n");
#endif
  const auto promptNum = std::min<unsigned>(md_promptInclusiveSum.at(totalEnergyEvents * 2 - 1), totalEnergyEvents);
  const auto delayNum = std::min<unsigned>(md_delayInclusiveSum.at(totalEnergyEvents * 2 - 1), totalEnergyEvents);
  PNI_DEBUG(std::format("There are {} prompt coincidences and {} delay coincidences, rate: {}%/{}%.\n", promptNum,
                        delayNum, double(100) * promptNum / totalEnergyEvents,
                        double(100) * delayNum / totalEnergyEvents));

  // Extract coincidence listmode
  md_promptListmode.reserve(promptNum);
  md_delayListmode.reserve(delayNum);
  impl::d_extractCoincidenceListmode(md_localSingleBuffer.data(), totalEnergyEvents * 2, md_promptInclusiveSum.data(),
                                     md_delayInclusiveSum.data(), md_promptListmode.data(), md_delayListmode.data(),
                                     totalEnergyEvents);
#if PNI_STANDARD_CONFIG_ENABLE_DEBUG_THREAD_TIME
  PNI_DEBUG("-- After d_extractCoincidenceListmode --\n");
#endif

  // Count the number of crystals [ Layer 3 & 4 ]
  impl::d_countCrystal(md_crystalCountMap.data() + mh_crystalNumInclusiveSum.back() * 2, md_promptListmode.data(),
                       promptNum, md_crystalNumInclusiveSum.data(),
                       static_cast<unsigned>(mh_crystalNumPerChannel.size()));
  impl::d_countCrystal(md_crystalCountMap.data() + mh_crystalNumInclusiveSum.back() * 3, md_delayListmode.data(),
                       delayNum, md_crystalNumInclusiveSum.data(),
                       static_cast<unsigned>(mh_crystalNumPerChannel.size()));
  return Coincidence::CoinResult{md_promptListmode.cspan(promptNum), md_delayListmode.cspan(delayNum)};
}

Coincidence::Coincidence()
    : m_impl(std::make_unique<Coincidence_impl>()) {}
Coincidence::~Coincidence() {};
Coincidence::CoinResult Coincidence::getDListmode(
    std::vector<std::span<interface::LocalSingle const>> d_coinListmodeLists, CoincidenceProtocol protocol) const {
  return m_impl->getDListmode(d_coinListmodeLists, protocol);
}
std::vector<unsigned> Coincidence::dumpCrystalCountMap() const {
  return m_impl->dumpCrystalCountMap();
}
void Coincidence::setTotalCrystalNumOfEachChannel(
    std::vector<uint32_t> const &crystalNumPerChannel) const {
  m_impl->setCrystalNumOfEachChannel(crystalNumPerChannel);
}
void Coincidence::clearCrystalCountMap() const {
  m_impl->clearCrystalCountMap();
}

} // namespace openpni::experimental::node
