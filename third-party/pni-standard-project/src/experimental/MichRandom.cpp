#include "include/experimental/node/MichRandom.hpp"

#include <mutex>
#include <variant>

#include "impl/MichCrystal.h"
#include "impl/MichFactors.hpp"
#include "impl/Random.h"
#include "impl/Share.hpp"
#include "include/IO.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/tools/Copy.hpp"
#include "include/experimental/tools/Parallel.hpp"

namespace openpni::experimental::node {
template <typename T>
inline void move_or_add(
    std::vector<T> &target, std::vector<T> &&source) {
  if (target.size() != source.size())
    target = std::move(source);
  else
    for (size_t i = 0; i < target.size(); ++i)
      target[i] += source[i];
}
class MichRandom_impl : public tools::DisableCopy {
public:
  MichRandom_impl(
      const core::MichDefine &mich)
      : m_mich(mich) {}
  ~MichRandom_impl() = default;

public:
  // void setRandomRatio(
  //     float randomRatio) {
  //   m_randomRatio = std::clamp(randomRatio, 0.f, 1.f);
  // }
  // test
  void setCountRatio(
      float countRatio) {
    m_countRatio = std::clamp(countRatio, 0.f, 1.f);
  }
  void setTimeBinRatio(
      float timeBinRatio) {
    m_timeBinRatio = std::clamp(timeBinRatio, 0.f, 1.f);
  }
  float getCountRatio() { return m_countRatio; }

  float getTimeBinRatio() { return m_timeBinRatio; }
  //
  void setMinSectorDifference(
      int minSectorDifference) {
    m_minSectorDifference = std::max(0, minSectorDifference);
    m_minSectorDifference = std::min<int>(MichInfoHub(m_mich).getPanelNum() / 2, m_minSectorDifference);
    reset_factors(); // clear: means not calculated, should be recalculated later.
  }
  void setRadialModuleNumS(
      int radialModuleNumS) {
    m_radialModuleNumS = std::max(1, radialModuleNumS);
    reset_factors(); // clear: means not calculated, should be recalculated later.
  }
  void setBadChannelThreshold(
      float badChannelThreshold) {
    m_badChannelThreshold = std::clamp(badChannelThreshold, 0.f, 1.f);
    reset_factors(); // clear: means not calculated, should be recalculated later.
  }
  void setDelayMich(
      float *delayMich) {
    auto [block, fan] = impl::calBlockLORAndFanLOR(m_mich, delayMich);
    mh_blockLOR = std::move(block);
    mh_fanLOR = std::move(fan);
    reset_factors(); // clear: means not calculated, should be recalculated later.
  }
  void addDelayMich(
      float *delayMich) {
    auto [block, fan] = impl::calBlockLORAndFanLOR(m_mich, delayMich);
    move_or_add(mh_blockLOR, std::move(block));
    move_or_add(mh_fanLOR, std::move(fan));
    reset_factors(); // clear: means not calculated, should be recalculated later.
  }
  void addDelayListmodes(
      std::span<basic::Listmode_t const> listmodes) {
    auto [block, fan] = impl::calBlockLORAndFanLOR(m_mich, listmodes);
    move_or_add(mh_blockLOR, std::move(block));
    move_or_add(mh_fanLOR, std::move(fan));
    PNI_DEBUG(std::format("Added {} delay listmodes.\n", listmodes.size()));
    PNI_DEBUG(std::format("Block LOR[0,0]: {}, Fan LOR[0,0]: {}\n", mh_blockLOR.empty() ? 0.f : mh_blockLOR[0],
                          mh_fanLOR.empty() ? 0.f : mh_fanLOR[0]));
    reset_factors(); // clear: means not calculated, should be recalculated later.
  }

  const std::vector<float> &getFactors() {
    check_hfactors();
    return mh_factors;
  }
  std::unique_ptr<float[]> dumpFactorsAsHMich() {
    check_hfactors();
    auto result = std::make_unique_for_overwrite<float[]>(MichInfoHub(m_mich).getMichSize());
    if (mh_factors.empty())
      std::fill_n(result.get(), MichInfoHub(m_mich).getMichSize(), 0.f);
    else {
      for (const auto [lor, rid1, rid2] : RangeGenerator(m_mich).allLORAndRectangleCrystals()) {
        result[lor] = impl::get_factor(rid1, rid2, mh_factors.data(), m_mich, m_minSectorDifference);
      }
    }
    return result;
  }
  cuda_sync_ptr<float> dumpFactorsAsDMich() {
    check_dfactors();
    auto result = make_cuda_sync_ptr<float>(MichInfoHub(m_mich).getMichSize(), "MichRandom_dumpFactorsAsDMich");
    impl::dumpFactorsAsDMich(md_factors.data(), m_mich, result.begin(), m_minSectorDifference);
    return result;
  }
  float const *getHRandomFactors(
      std::span<core::MichStandardEvent const> h_events) {
    check_hfactors();
    if (h_events.size() > mh_bufferedValues.size())
      mh_bufferedValues.resize(h_events.size());
    tools::parallel_for_each(h_events.size(), [&](std::size_t i) {
      const auto &event = h_events[i];
      mh_bufferedValues[i] = impl::get_factor(event, mh_factors.data(), m_mich, m_minSectorDifference) *
                             m_countRatio; // 20251111lgx:m_randomRatio may not apply here!
    });
    return mh_bufferedValues.data();
  }
  float const *getDRandomFactors(
      std::span<core::MichStandardEvent const> d_events) {
    check_dfactors();
    if (d_events.size() > md_bufferedValues.elements())
      md_bufferedValues.reserve(d_events.size());
    impl::getDRandomFactors(d_events, md_factors.data(), m_mich, m_minSectorDifference, md_bufferedValues.data());
    example::d_parallel_mul(md_bufferedValues.data(), m_countRatio, md_bufferedValues.data(),
                            md_bufferedValues.elements());
    return md_bufferedValues.data();
  }
  float const *getDRandomFactorsBatch(
      std::span<std::size_t const> lorIndices) {
    tl_mich_standard_events().reserve(lorIndices.size());
    impl::d_fill_standard_events_ids_from_lor_ids(tl_mich_standard_events().data(), lorIndices, m_mich);
    return getDRandomFactors(tl_mich_standard_events().cspan(lorIndices.size()));
  }
  float const *getHRandomFactorsBatch(
      std::span<std::size_t const> lorIndices) {
    if (lorIndices.size() > mh_bufferedEvents.size())
      mh_bufferedEvents.resize(lorIndices.size());
    impl::h_fill_crystal_ids(mh_bufferedEvents.data(), lorIndices.data(), lorIndices.size(), m_mich);
    return getHRandomFactors(std::span<core::MichStandardEvent const>(mh_bufferedEvents.data(), lorIndices.size()));
  }

  std::unique_ptr<MichRandom_impl> copy() {
    std::lock_guard lock(m_mutex); // ensure only one thread calculate factors.
    auto result = std::make_unique<MichRandom_impl>(m_mich);
    result->mh_blockLOR = mh_blockLOR;
    result->mh_fanLOR = mh_fanLOR;
    // result->m_randomRatio = m_randomRatio;
    result->m_countRatio = m_countRatio;
    result->m_timeBinRatio = m_timeBinRatio;
    result->m_minSectorDifference = m_minSectorDifference;
    result->m_radialModuleNumS = m_radialModuleNumS;
    result->m_badChannelThreshold = m_badChannelThreshold;
    result->mh_factors = mh_factors; // copy the factors
    result->md_factors.clear();      // do not copy device factors, should copy from host later.
    result->mh_bufferedValues = decltype(mh_bufferedValues)(); // do not copy buffered values.
    result->md_bufferedValues.clear();                         // do not copy buffered values.
    return result;
  }

private:
  void check_hfactors() {
    std::lock_guard lock(m_mutex); // ensure only one thread calculate factors.
    if (mh_factors.size() != MichInfoHub(m_mich).getTotalCrystalNum())
      calculate_hFactors();
  }
  void check_dfactors() {
    std::lock_guard lock(m_mutex); // ensure only one thread calculate factors.
    check_hfactors();
    if (!md_factors || md_factors.getDeviceIndex() != cuda_get_device_index_exept())
      md_factors = make_cuda_sync_ptr_from_hcopy(std::span<float const>(mh_factors), "MichRandom_factors_fromHost");
  }
  void reset_factors() {
    std::lock_guard lock(m_mutex); // ensure only one thread calculate factors.
    mh_factors.clear();
    md_factors.clear();
  }
  void calculate_hFactors() {
    std::lock_guard lock(m_mutex); // ensure only one thread calculate factors.
    mh_factors.clear();
    mh_factors.resize(MichInfoHub(m_mich).getTotalCrystalNum(), 0);
    md_factors.clear(); // clear device factors because host factors changed.
    if (mh_blockLOR.empty() || mh_fanLOR.empty())
      throw exceptions::algorithm_unexpected_condition(
          "MichRandom: delay source is not set or invalid, cannot calculate random factors.");
    mh_factors = impl::calCryFctByFanSum(m_mich, mh_fanLOR, mh_blockLOR, m_radialModuleNumS, m_badChannelThreshold);
    PNI_DEBUG("MichRandom: Random factors calculation done.\n");
  }

private:
  int m_minSectorDifference = 0;
  int m_radialModuleNumS = 0;
  float m_badChannelThreshold = 0;
  // float m_randomRatio = 1.0f;
  // test
  float m_timeBinRatio = 1.0f;
  float m_countRatio = 1.0f;
  //

  std::vector<float> mh_factors;                         // 每个晶体的随机因子，Rectangle Layout
  cuda_sync_ptr<float> md_factors{"MichRandom_factors"}; // 每个晶体的随机因子，Rectangle Layout
  std::vector<float> mh_bufferedValues;
  cuda_sync_ptr<float> md_bufferedValues{"MichRandom_bufferedValues"};
  std::vector<core::MichStandardEvent> mh_bufferedEvents;
  std::recursive_mutex m_mutex;
  std::vector<float> mh_blockLOR;
  const core::MichDefine m_mich;
  std::vector<float> mh_fanLOR;
};

MichRandom::MichRandom(
    const core::MichDefine &mich)
    : m_impl(std::make_unique<MichRandom_impl>(mich)) {}
MichRandom::MichRandom(
    std::unique_ptr<MichRandom_impl> impl)
    : m_impl(std::move(impl)) {}
MichRandom::MichRandom(MichRandom&&) noexcept = default;
MichRandom &MichRandom::operator=(MichRandom&&) noexcept = default;
MichRandom::~MichRandom() {};
MichRandom MichRandom::copy() const {
  return MichRandom(m_impl->copy());
}
std::unique_ptr<MichRandom> MichRandom::copyPtr() {
  return std::make_unique<MichRandom>(m_impl->copy());
}

void MichRandom::setMinSectorDifference(
    int minSectorDifference) {
  m_impl->setMinSectorDifference(minSectorDifference);
}
void MichRandom::setRadialModuleNumS(
    int radialModuleNumS) {
  m_impl->setRadialModuleNumS(radialModuleNumS);
}
void MichRandom::setBadChannelThreshold(
    float badChannelThreshold) {
  m_impl->setBadChannelThreshold(badChannelThreshold);
}
void MichRandom::setDelayMich(
    float *delayMich) {
  m_impl->setDelayMich(delayMich);
}
void MichRandom::addDelayListmodes(
    std::span<basic::Listmode_t const> listmodes) {
  m_impl->addDelayListmodes(listmodes);
}
std::vector<float> const &MichRandom::getFactors() {
  return m_impl->getFactors();
}
std::unique_ptr<float[]> MichRandom::dumpFactorsAsHMich() {
  return m_impl->dumpFactorsAsHMich();
}
cuda_sync_ptr<float> MichRandom::dumpFactorsAsDMich() {
  return m_impl->dumpFactorsAsDMich();
}

float const *MichRandom::getHRandomFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  return m_impl->getHRandomFactors(events);
}
float const *MichRandom::getHRandomFactorsBatch(
    std::span<std::size_t const> lorIndices) {
  return m_impl->getHRandomFactorsBatch(lorIndices);
}
float const *MichRandom::getDRandomFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  return m_impl->getDRandomFactors(events);
}
float const *MichRandom::getDRandomFactorsBatch(
    std::span<std::size_t const> lorIndices) {
  return m_impl->getDRandomFactorsBatch(lorIndices);
}
// void MichRandom::setRandomRatio(
//     float randomRatio) {
//   m_impl->setRandomRatio(randomRatio);
// }
// lgxtest
void MichRandom::setCountRatio(
    float countRatio) {
  m_impl->setCountRatio(countRatio);
}
void MichRandom::setTimeBinRatio(
    float timeBinRatio) {
  m_impl->setTimeBinRatio(timeBinRatio);
}
float MichRandom::getCountRatio() {
  return m_impl->getCountRatio();
}
float MichRandom::getTimeBinRatio() {
  return m_impl->getTimeBinRatio();
}
} // namespace openpni::experimental::node