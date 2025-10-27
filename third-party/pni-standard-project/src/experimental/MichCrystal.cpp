#include "include/experimental/node/MichCrystal.hpp"

#include <mutex>
#include <vector>

#include "impl/MichCrystal.h"
#include "include/basic/CudaPtr.hpp"
#include "include/experimental/core/Span.hpp"
#include "include/experimental/tools/Copy.hpp"
#include "include/experimental/tools/Parallel.hpp"
#define MichInfoHub(m) core::MichInfoHub::create(m)
#define IndexConverter(m) core::IndexConverter::create(m)
#define RangeGenerator(m) core::RangeGenerator::create(m)
namespace openpni::experimental::node::impl {
inline std::vector<core::CrystalGeom> createUniformCrystalLayout(
    core::MichDefine __mich) {
  std::vector<core::CrystalGeom> result(MichInfoHub(__mich).getTotalCrystalNum());
  std::vector<std::optional<core::DetectorCoord>> detectorCoordCache;
  core::MDSpan<3> resultSpan = core::MDSpan<3>::create(
      __mich.polygon.getTotalDetectorNum(), __mich.detector.getTotalCrystalU(), __mich.detector.getTotalCrystalV());
  detectorCoordCache.resize(__mich.polygon.getTotalDetectorNum());
  for (const auto &idx : resultSpan) {
    // auto &detIdx{idx[0]}, &uCryIdx{idx[1]}, &vCryIdx{idx[2]};
    auto &[detIdx, uCryIdx, vCryIdx] = idx;
    if (!detectorCoordCache[detIdx])
      detectorCoordCache[detIdx] = __mich.polygon.getDetectorGlobalCoor(detIdx);
    result[resultSpan[idx]] =
        core::calculateCrystalGeometry(*detectorCoordCache[detIdx], __mich.detector, uCryIdx, vCryIdx);
  }
  return result;
}
inline std::vector<core::CrystalGeom> createRectangleCrystalLayout(
    core::MichDefine __mich) {
  std::vector<core::CrystalGeom> result(MichInfoHub(__mich).getTotalCrystalNum());
  std::vector<std::optional<core::DetectorCoord>> detectorCoordCache;
  core::MDSpan<2> resultSpan =
      core::MDSpan<2>::create(MichInfoHub(__mich).getCrystalNumOneRing(), MichInfoHub(__mich).getRingNum());
  detectorCoordCache.resize(__mich.polygon.getTotalDetectorNum());
  for (const auto &idx : resultSpan) {
    core::RectangleID rid;
    rid.ringID = idx[1];
    rid.idInRing = idx[0];
    const auto [detIdx, uCryIdx, vCryIdx] = IndexConverter(__mich).getUniformIDFromRectangleID(rid);
    if (!detectorCoordCache[detIdx])
      detectorCoordCache[detIdx] = __mich.polygon.getDetectorGlobalCoor(detIdx);
    result[resultSpan[idx]] =
        core::calculateCrystalGeometry(*detectorCoordCache[detIdx], __mich.detector, uCryIdx, vCryIdx);
  }
  return result;
}
} // namespace openpni::experimental::node::impl
namespace openpni::experimental::node {
class MichCrystal_impl : public tools::DisableCopy {
public:
  MichCrystal_impl(
      const core::MichDefine &__mich)
      : m_mich(__mich) {}
  ~MichCrystal_impl() = default;

public:
  std::unique_ptr<MichCrystal_impl> copy() {
    auto result = std::make_unique<MichCrystal_impl>(m_mich);
    this->check_rd_layout();
    this->check_ud_layout();
    result->mh_uniformLayout = mh_uniformLayout;
    result->mh_rectangleLayout = mh_rectangleLayout;
    return result;
  }

public:
  core::CrystalGeom const *getHCrystals(
      std::span<std::size_t const> __lors) {
    check_rh_layout();
    mh_bufferedCrystals.resize(__lors.size() * 2);
    tools::parallel_for_each(__lors.size(), [&](std::size_t i) {
      const auto [rid1, rid2] = IndexConverter(m_mich).getCrystalIDFromLORID(__lors[i]);
      mh_bufferedCrystals[i * 2] = mh_rectangleLayout[IndexConverter(m_mich).getFlatIdFromRectangleId(rid1)];
      mh_bufferedCrystals[i * 2 + 1] = mh_rectangleLayout[IndexConverter(m_mich).getFlatIdFromRectangleId(rid2)];
    });
    return mh_bufferedCrystals.data();
  }
  core::CrystalGeom const *getHCrystals(
      std::span<core::UniformID const> __uids) {
    check_uh_layout();
    mh_bufferedCrystals.resize(__uids.size());
    tools::parallel_for_each(__uids.size(), [&](std::size_t i) {
      mh_bufferedCrystals[i] = mh_uniformLayout[IndexConverter(m_mich).getFlatIdFromUniformID(__uids[i])];
    });
    return mh_bufferedCrystals.data();
  }
  core::CrystalGeom const *getHCrystals(
      std::span<core::RectangleID const> __rids) {
    check_rh_layout();
    mh_bufferedCrystals.resize(__rids.size());
    tools::parallel_for_each(__rids.size(), [&](std::size_t i) {
      mh_bufferedCrystals[i] = mh_rectangleLayout[IndexConverter(m_mich).getFlatIdFromRectangleId(__rids[i])];
    });
    return mh_bufferedCrystals.data();
  }
  core::CrystalGeom const *getDCrystals(
      std::span<std::size_t const> __lors) {
    check_rd_layout();
    if (__lors.size() * 2 > md_bufferedCrystals.elements())
      md_bufferedCrystals = make_cuda_sync_ptr<core::CrystalGeom>(__lors.size() * 2, "MichCrystal_rd_buffer");
    impl::d_fill_crystal_geoms(md_bufferedCrystals.get(), m_mich, md_rectangleLayout.get(), __lors);
    return md_bufferedCrystals.get();
  }
  core::CrystalGeom const *getDCrystals(
      std::span<core::RectangleID const> __rids) {
    check_rd_layout();
    if (__rids.size() > md_bufferedCrystals.elements())
      md_bufferedCrystals = make_cuda_sync_ptr<core::CrystalGeom>(__rids.size(), "MichCrystal_rd_buffer");
    impl::d_fill_crystal_geoms(md_bufferedCrystals.get(), m_mich, md_rectangleLayout.get(), __rids);
    return md_bufferedCrystals.get();
  }
  core::CrystalGeom const *getDCrystals(
      std::span<core::UniformID const> __uids) {
    check_ud_layout();
    if (__uids.size() > md_bufferedCrystals.elements())
      md_bufferedCrystals = make_cuda_sync_ptr<core::CrystalGeom>(__uids.size(), "MichCrystal_ud_buffer");
    impl::d_fill_crystal_geoms(md_bufferedCrystals.get(), m_mich, md_uniformLayout.get(), __uids);
    return md_bufferedCrystals.get();
  }
  core::MichDefine mich() const { return m_mich; }
  std::vector<core::CrystalGeom> dumpCrystalsUniformLayout() {
    check_uh_layout();
    return mh_uniformLayout;
  }
  std::vector<core::CrystalGeom> dumpCrystalsRectangleLayout() {
    check_rh_layout();
    return mh_rectangleLayout;
  }
  void fillHCrystalsBatch(
      std::span<core::MichStandardEvent> __events) {
    check_rh_layout();
    tools::parallel_for_each(__events.size(), [&](std::size_t i) {
      __events[i].geo1 = mh_rectangleLayout[IndexConverter(m_mich).getFlatIdFromRectangleId(__events[i].crystal1)];
      __events[i].geo2 = mh_rectangleLayout[IndexConverter(m_mich).getFlatIdFromRectangleId(__events[i].crystal2)];
    });
  }
  void fillDCrystalsBatch(
      std::span<core::MichStandardEvent> __events) {
    check_rd_layout();
    impl::d_fill_crystal_geoms(__events, m_mich, md_rectangleLayout.get());
  }

private:
  void check_uh_layout() {
    std::lock_guard lock(m_mutex);
    if (mh_uniformLayout.empty())
      mh_uniformLayout = impl::createUniformCrystalLayout(m_mich);
  }
  void check_rh_layout() {
    std::lock_guard lock(m_mutex);
    if (mh_rectangleLayout.empty())
      mh_rectangleLayout = impl::createRectangleCrystalLayout(m_mich);
  }
  void check_ud_layout() {
    std::lock_guard lock(m_mutex);
    check_uh_layout();
    if (!md_uniformLayout || md_uniformLayout.getDeviceIndex() != cuda_get_device_index_exept())
      md_uniformLayout = make_cuda_sync_ptr_from_hcopy(mh_uniformLayout, "MichCrystal_ud_layout");
  }
  void check_rd_layout() {
    std::lock_guard lock(m_mutex);
    check_rh_layout();
    if (!md_rectangleLayout || md_rectangleLayout.getDeviceIndex() != cuda_get_device_index_exept())
      md_rectangleLayout = make_cuda_sync_ptr_from_hcopy(mh_rectangleLayout, "MichCrystal_rd_layout");
  }

private:
  const core::MichDefine m_mich;
  std::vector<core::CrystalGeom> mh_uniformLayout;
  std::vector<core::CrystalGeom> mh_rectangleLayout;
  cuda_sync_ptr<core::CrystalGeom> md_uniformLayout;
  cuda_sync_ptr<core::CrystalGeom> md_rectangleLayout;
  std::vector<core::CrystalGeom> mh_bufferedCrystals;
  cuda_sync_ptr<core::CrystalGeom> md_bufferedCrystals;
  std::recursive_mutex m_mutex;
};

MichCrystal::MichCrystal(
    core::MichDefine __mich)
    : m_impl(std::make_unique<MichCrystal_impl>(__mich)) {}
MichCrystal::~MichCrystal() {}
core::CrystalGeom const *MichCrystal::getHCrystalsBatch(
    std::span<std::size_t const> __lors) const {
  return m_impl->getHCrystals(__lors);
}

core::CrystalGeom const *MichCrystal::getHCrystalsBatch(
    std::span<core::UniformID const> __uids) const {
  return m_impl->getHCrystals(__uids);
}
core::CrystalGeom const *MichCrystal::getHCrystalsBatch(
    std::span<core::RectangleID const> __rids) const {
  return m_impl->getHCrystals(__rids);
}

core::CrystalGeom const *MichCrystal::getDCrystalsBatch(
    std::span<core::UniformID const> __uids) const {
  return m_impl->getDCrystals(__uids);
}

core::CrystalGeom const *MichCrystal::getDCrystalsBatch(
    std::span<core::RectangleID const> __rids) const {
  return m_impl->getDCrystals(__rids);
}
core::MichDefine MichCrystal::mich() const {
  return m_impl->mich();
}
std::vector<core::CrystalGeom> MichCrystal::dumpCrystalsUniformLayout() const {
  return m_impl->dumpCrystalsUniformLayout();
}
std::vector<core::CrystalGeom> MichCrystal::dumpCrystalsRectangleLayout() const {
  return m_impl->dumpCrystalsRectangleLayout();
}
void MichCrystal::fillHCrystalsBatch(
    std::span<core::MichStandardEvent> __events) const {
  m_impl->fillHCrystalsBatch(__events);
}
void MichCrystal::fillDCrystalsBatch(
    std::span<core::MichStandardEvent> __events) const {
  m_impl->fillDCrystalsBatch(__events);
}
} // namespace openpni::experimental::node
