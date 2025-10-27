#include "Copy.h"
#include "MichCrystal.h"
#include "Projection.h"
#include "Share.hpp"
#include "include/basic/CudaPtr.hpp"
#include "include/experimental/core/BasicMath.hpp"
#include "include/experimental/core/Image.hpp"
#include "include/experimental/core/Random.hpp"
#include "include/experimental/node/MichAttn.hpp"
#include "include/experimental/node/MichCrystal.hpp"
#include "include/experimental/tools/Parallel.hpp"

namespace openpni::experimental::node::impl {
void h_HUMapToAttnMap(
    float *out_attnmap, float const *in_humap, core::Grids<3> map) {
  tools::parallel_for_each(map.totalSize(), [&](std::size_t idx) {
    constexpr float WATER_ATTENUATION_MM_511 = 0.009687;
    out_attnmap[idx] = core::max<float>(0.f, in_humap[idx] * 1e-3 + 1) * WATER_ATTENUATION_MM_511;
  });
}

} // namespace openpni::experimental::node::impl
namespace openpni::experimental::node {
class MichAttn_impl {
public:
  explicit MichAttn_impl(
      core::MichDefine __mich)
      : m_michDefine(__mich)
      , m_michCrystal(m_michDefine) {}
  ~MichAttn_impl() = default;

public:
  std::unique_ptr<MichAttn_impl> copy();
  auto mich() const -> core::MichDefine { return m_michDefine; }
  void setPreferredSource(MichAttn::AttnFactorSource source);
  void setFetchMode(MichAttn::AttnFactorFetchMode mode);
  void setMapSize(core::Grids<3> map);
  void bindHHUMap(float const *h_data);
  void bindHAttnMap(float const *h_data);
  std::unique_ptr<float[]> dumpHAttnMich();

  // // temp, for test
  // void test_bindHExistingAttn(float *h_attnCoff, float *h_attnMap, core::Grids<3> map);

public:
  float const *getHAttnFactorsBatch(std::span<core::MichStandardEvent const> events);
  float const *getHAttnFactorsBatch(std::span<std::size_t const> lorIndices);
  float const *getDAttnFactorsBatch(std::span<core::MichStandardEvent const> events);
  float const *getDAttnFactorsBatch(std::span<std::size_t const> lorIndices);

  // temp
  float const *h_getAttnMap();
  float const *d_getAttnMap();
  float const *h_getAttnMich();
  float const *d_getAttnMich();
  core::Grids<3> getMapGrids() const;

private:
  void checkHAttnMich();
  void checkDAttnMich();
  void checkOrThrowGenerateFlags();
  void generateAttnMich();
  std::unique_ptr<float[]> h_generateAttnMich();
  cuda_sync_ptr<float> d_generateAttnMich();
  // float const *h_getAttnMap();
  // float const *d_getAttnMap();
  // float const *h_getAttnMich();
  // float const *d_getAttnMich();
  float const *h_calculateSomeAttnFactors(std::span<core::MichStandardEvent const> events);
  float const *d_calculateSomeAttnFactors(std::span<core::MichStandardEvent const> events);
  float const *h_getSomeAttnFactors(std::span<core::MichStandardEvent const> events);
  float const *d_getSomeAttnFactors(std::span<core::MichStandardEvent const> events);

private:
  const core::MichDefine m_michDefine;
  MichCrystal m_michCrystal;
  std::optional<core::Grids<3>> m_grid;

  /**
   * 这几个指针的唯一流转路径是：
   * 1. mh_HUMap_ -> mh_AttnMap
   * 2. mh_AttnMap_ -> md_AttnMap
   * 3. mh_AttnMap -> md_AttnMap
   */

  float const *mh_HUMap_;   // Bound from outside
  float const *mh_AttnMap_; // Bound from outside
  std::unique_ptr<float[]> mh_AttnMap;
  cuda_sync_ptr<float> md_AttnMap{"MichAttn_attnMap"};
  bool m_attnMichGenerated = false;

  MichAttn::AttnFactorSource m_attnFactorSource{MichAttn::Attn_GPU};
  MichAttn::AttnFactorFetchMode m_attnFetchMode{MichAttn::FromPreBaked};
  std::unique_ptr<float[]> mh_attnMich;
  cuda_sync_ptr<float> md_attnMich{"MichAttn_attnMich"};
  std::unique_ptr<float[]> mh_tempAttnFactors;
  cuda_sync_ptr<float> md_tempAttnFactors{"MichAttn_tempFactors"};

  std::vector<core::MichStandardEvent> mh_tempEvents;
  std::vector<float> mh_tempFactors;
  cuda_sync_ptr<float> md_tempFactors{"MichAttn_tempFactors"};
};
void MichAttn_impl::setPreferredSource(
    MichAttn::AttnFactorSource source) {
  m_attnFactorSource = source;
  m_attnMichGenerated = false;
}
void MichAttn_impl::setFetchMode(
    MichAttn::AttnFactorFetchMode mode) {
  m_attnFetchMode = mode;
  m_attnMichGenerated = false;
}
void MichAttn_impl::setMapSize(
    core::Grids<3> map) {
  m_grid = map;
  m_attnMichGenerated = false;
}
void MichAttn_impl::bindHHUMap(
    float const *h_data) {
  mh_HUMap_ = h_data;
  m_attnMichGenerated = false;
}
void MichAttn_impl::bindHAttnMap(
    float const *h_data) {
  mh_AttnMap_ = h_data;
  m_attnMichGenerated = false;
}
std::unique_ptr<float[]> MichAttn_impl::dumpHAttnMich() {
  generateAttnMich();
  checkOrThrowGenerateFlags();
  if (mh_attnMich)
    return host_deep_copy(mh_attnMich.get(), core::MichInfoHub::create(m_michDefine).getMichSize());
  else if (md_attnMich) {
    return device_deep_copy_to_host(md_attnMich);
  }
  throw exceptions::algorithm_unexpected_condition("No attenuation mich generated.");
}

float const *MichAttn_impl::getDAttnFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  if (m_attnFetchMode == MichAttn::FromInstant)
    return d_calculateSomeAttnFactors(events);
  else
    return d_getSomeAttnFactors(events);
}
float const *MichAttn_impl::getDAttnFactorsBatch(
    std::span<std::size_t const> lorIndices) {
  tl_mich_standard_events().reserve(lorIndices.size());
  impl::d_fill_crystal_ids(tl_mich_standard_events().get(), lorIndices.data(), lorIndices.size(), m_michDefine);
  m_michCrystal.fillDCrystalsBatch(tl_mich_standard_events().span(lorIndices.size()));
  return getDAttnFactorsBatch(tl_mich_standard_events().cspan(lorIndices.size()));
}
float const *MichAttn_impl::getHAttnFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  if (m_attnFetchMode == MichAttn::FromInstant)
    return h_calculateSomeAttnFactors(events);
  else
    return h_getSomeAttnFactors(events);
}
float const *MichAttn_impl::getHAttnFactorsBatch(
    std::span<std::size_t const> lorIndices) {
  if (mh_tempEvents.size() < lorIndices.size())
    mh_tempEvents.resize(lorIndices.size());
  impl::h_fill_crystal_ids(mh_tempEvents.data(), lorIndices.data(), lorIndices.size(), m_michDefine);
  return getHAttnFactorsBatch(std::span<core::MichStandardEvent const>(mh_tempEvents.data(), lorIndices.size()));
}
float const *MichAttn_impl::h_calculateSomeAttnFactors(
    std::span<core::MichStandardEvent const> events) {
  auto *h_attnMap = h_getAttnMap();
  if (!h_attnMap)
    throw exceptions::algorithm_unexpected_condition("No attenuation map bound or generated.");
  if (mh_tempFactors.size() < events.size())
    mh_tempFactors.resize(events.size());
  impl::h_gen_some_attn_factors(mh_tempFactors.data(),
                                std::span<core::MichStandardEvent const>(events.data(), events.size()), h_attnMap,
                                *m_grid);
  return mh_tempFactors.data();
}
float const *MichAttn_impl::d_calculateSomeAttnFactors(
    std::span<core::MichStandardEvent const> events) {
  auto *d_attnMap = d_getAttnMap();
  if (!d_attnMap)
    throw exceptions::algorithm_unexpected_condition("No attenuation map bound or generated.");
  md_tempFactors.reserve(events.size());
  impl::d_gen_some_attn_factors(md_tempFactors, std::span<core::MichStandardEvent const>(events), d_attnMap, *m_grid);
  return md_tempFactors.get();
}
float const *MichAttn_impl::h_getSomeAttnFactors(
    std::span<core::MichStandardEvent const> events) {
  auto *h_attnMich = h_getAttnMich(); // If no attn mich, will generate
  if (mh_tempFactors.size() < events.size())
    mh_tempFactors.resize(events.size());
  tools::parallel_for_each(events.size(), [&](std::size_t index) {
    auto lorIndex = core::IndexConverter::create(m_michDefine)
                        .getLORIDFromRectangleID(events[index].crystal1, events[index].crystal2);
    mh_tempFactors[index] = h_attnMich[lorIndex];
  });
  return mh_tempFactors.data();
}
float const *MichAttn_impl::d_getSomeAttnFactors(
    std::span<core::MichStandardEvent const> events) {
  auto *d_attnMich = d_getAttnMich(); // If no attn mich, will generate
  md_tempFactors.reserve(events.size());
  impl::d_fill_factors_from_mich(events.size(), events.data(), md_tempFactors.get(), d_attnMich, m_michDefine);
  return md_tempFactors.get();
}

float const *MichAttn_impl::h_getAttnMap() {
  if (mh_AttnMap_)
    return mh_AttnMap_;
  if (mh_AttnMap)
    return mh_AttnMap.get();
  if (mh_HUMap_ && m_grid) {
    mh_AttnMap = std::make_unique<float[]>(m_grid->totalSize());
    impl::h_HUMapToAttnMap(mh_AttnMap.get(), mh_HUMap_, *m_grid);
    return mh_AttnMap.get();
  }
  return nullptr;
}
float const *MichAttn_impl::d_getAttnMap() {
  if (md_AttnMap)
    return md_AttnMap.get();
  md_AttnMap = make_cuda_sync_ptr_from_hcopy(std::span<float const>(h_getAttnMap(), m_grid->totalSize()),
                                             "MichAttn_attnMap_fromHost");
  return md_AttnMap.get();
}
float const *MichAttn_impl::h_getAttnMich() {
  generateAttnMich();
  checkHAttnMich();
  return mh_attnMich.get();
}
float const *MichAttn_impl::d_getAttnMich() {
  generateAttnMich();
  checkDAttnMich();
  return md_attnMich.get();
}

void MichAttn_impl::checkHAttnMich() {
  if (mh_attnMich)
    return;
  if (md_attnMich) {
    mh_attnMich = device_deep_copy_to_host(md_attnMich);
    return;
  }
  // No attn mich, need to generate
  generateAttnMich();
  checkHAttnMich();
}
void MichAttn_impl::checkDAttnMich() {
  if (md_attnMich)
    return;
  if (mh_attnMich) {
    md_attnMich = make_cuda_sync_ptr_from_hcopy(
        std::span<float const>(mh_attnMich.get(), core::MichInfoHub::create(m_michDefine).getMichSize()));
    return;
  }
  // No attn mich, need to generate
  generateAttnMich();
  checkDAttnMich();
}
void MichAttn_impl::generateAttnMich() {
  if (!m_attnMichGenerated) {
    if (m_attnFactorSource == MichAttn::Attn_CPU) {
      mh_attnMich = h_generateAttnMich();
      m_attnMichGenerated = true;
    } else {
      md_attnMich = d_generateAttnMich();
      m_attnMichGenerated = true;
    }
  }
}
std::unique_ptr<float[]> MichAttn_impl::h_generateAttnMich() {
  auto *attnMap = h_getAttnMap();
  if (!attnMap)
    throw exceptions::algorithm_unexpected_condition("No attenuation map bound or generated.");
  PNI_DEBUG("Generating attenuation mich on CPU...\n");
  auto result = std::make_unique<float[]>(core::MichInfoHub::create(m_michDefine).getMichSize());
  LORBatch lorBatch(m_michDefine);
  lorBatch.setSubsetNum(1).setCurrentSubset(0);
  for (auto lors = lorBatch.nextHBatch(); !lors.empty(); lors = lorBatch.nextHBatch()) {
    if (mh_tempEvents.size() < lors.size())
      mh_tempEvents.resize(lors.size());
    impl::h_fill_crystal_ids(mh_tempEvents.data(), lors.data(), lors.size(), m_michDefine);
    m_michCrystal.fillHCrystalsBatch(std::span<core::MichStandardEvent>(mh_tempEvents.data(), lors.size()));
    if (mh_tempFactors.size() < lors.size())
      mh_tempFactors.resize(lors.size());
    impl::h_gen_some_attn_factors(
        mh_tempFactors.data(), std::span<core::MichStandardEvent>(mh_tempEvents.data(), lors.size()), attnMap, *m_grid);
    tools::parallel_for_each(lors.size(), [&](std::size_t index) { result[lors[index]] = mh_tempFactors[index]; });
  }
  return result;
}
cuda_sync_ptr<float> MichAttn_impl::d_generateAttnMich() {
  auto *attnMap = d_getAttnMap();
  if (!attnMap)
    throw exceptions::algorithm_unexpected_condition("No attenuation map bound or generated.");
  PNI_DEBUG("Generating attenuation mich on GPU...\n");
  auto result = make_cuda_sync_ptr<float>(core::MichInfoHub::create(m_michDefine).getMichSize(),
                                          "MichAttn_attnMichGenerateResult");
  LORBatch lorBatch(m_michDefine);
  lorBatch.setSubsetNum(1).setCurrentSubset(0);
  for (auto lors = lorBatch.nextDBatch(); !lors.empty(); lors = lorBatch.nextDBatch()) {
    tl_mich_standard_events().reserve(lors.size());
    impl::d_fill_crystal_ids(tl_mich_standard_events(), lors.data(), lors.size(), m_michDefine);
    m_michCrystal.fillDCrystalsBatch(tl_mich_standard_events().span(lors.size()));
    md_tempFactors.reserve(lors.size());
    impl::d_gen_some_attn_factors(md_tempFactors.get(), tl_mich_standard_events().span(lors.size()), attnMap, *m_grid);
    impl::d_redirect_to_mich(lors.size(), lors.data(), md_tempFactors.get(), result.get());
  }
  PNI_DEBUG("Attenuation mich generation completed.\n");
  return result;
}
void MichAttn_impl::checkOrThrowGenerateFlags() {
  if (!m_attnMichGenerated)
    throw exceptions::algorithm_unexpected_condition("No attenuation mich generated.");
}

core::Grids<3> MichAttn_impl::getMapGrids() const {
  if (m_grid)
    return *m_grid;
  throw exceptions::algorithm_unexpected_condition("No map grid set.");
}

std::unique_ptr<MichAttn_impl> MichAttn_impl::copy() {
  auto new_impl = std::make_unique<MichAttn_impl>(m_michDefine);
  new_impl->m_grid = m_grid;
  new_impl->mh_HUMap_ = mh_HUMap_;
  new_impl->mh_AttnMap_ = mh_AttnMap_;
  return new_impl;
}

// // temp, for test
// void MichAttn_impl::test_bindHExistingAttn(
//     float *h_attnCoff, float *h_attnMap, core::Grids<3> map) {
//   if (mh_AttnMap_)
//     throw exceptions::algorithm_unexpected_condition("Attn map already bound.");
//   if (mh_attnMich)
//     throw exceptions::algorithm_unexpected_condition("Attn coff already exists.");
//   if (mh_HUMap_)
//     throw exceptions::algorithm_unexpected_condition("HU map already bound.");
//   mh_attnMich = host_deep_copy(h_attnCoff, core::MichInfoHub::create(m_michDefine).getMichSize());
//   setMapSize(map);
//   bindHAttnMap(h_attnMap);
//   m_attnMichGenerated = true;
// }

} // namespace openpni::experimental::node
