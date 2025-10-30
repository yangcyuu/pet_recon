#include "include/experimental/node/LORBatch.hpp"

#include <mutex>

#include "impl/LORBatch.h"
#include "impl/Share.hpp"
#include "include/basic/CudaPtr.hpp"
#include "include/experimental/core/Span.hpp"
#include "include/experimental/tools/Loop.hpp"
#include "include/experimental/tools/Parallel.hpp"
#ifndef MichInfoHub
#define MichInfoHub(m) core::MichInfoHub::create(m)
#endif
#ifndef IndexConverter
#define IndexConverter(m) core::IndexConverter::create(m)
#endif
#ifndef RangeGenerator
#define RangeGenerator(m) core::RangeGenerator::create(m)
#endif
namespace openpni::experimental::node {
class LORBatch_impl {
public:
  LORBatch_impl(
      core::MichDefine __mich)
      : m_mich(__mich) {
    m_validRingPair =
        tools::continuous_range_generator.by_max_difference_ordered(0, MichInfoHub(m_mich).getRingNum(), m_maxRingDiff);
  }

  void setSubsetNum(
      int num) {
    if (num <= 0)
      throw exceptions::algorithm_unexpected_condition("LORRange: subsetNum must be positive");
    m_subsetNum = std::max<int>(1, num);
  }
  void setBinCut(
      int cut) {
    m_binCut = std::clamp<int>(cut, 0, MichInfoHub(m_mich).getBinNum() / 2 - 1);
  }
  void setMaxRingDiff(
      int diff) {
    m_maxRingDiff = diff < 0 ? MichInfoHub(m_mich).getRingNum() : diff;
    m_validRingPair =
        tools::continuous_range_generator.by_max_difference_ordered(0, MichInfoHub(m_mich).getRingNum(), m_maxRingDiff);
    md_validRingPair = {};
  }

  void setCurrentSubset(
      int subset) {
    if (subset >= m_subsetNum || subset < 0)
      throw exceptions::algorithm_unexpected_condition("LORRange: subset out of range");
    m_currentSubset = subset;
    start();
  }
  void start() {
    m_currentSubsetProcessedLORs = 0;
    m_currentSubsetTotalLORs =
        (MichInfoHub(m_mich).getBinNum() - 2 * m_binCut) *
        core::mich::calViewNumInSubset(m_mich.polygon, m_mich.detector, m_subsetNum, m_currentSubset) *
        m_validRingPair.size();
  }
  std::span<const std::size_t> nextHBatch() {
    if (!hasNextBatch())
      return {};
    check_hBatch();
    fill_hBatch();
    return std::span<const std::size_t>(mh_currentBatch.data(), m_currentBatchSize);
  }
  std::span<const std::size_t> nextDBatch() {
    if (!hasNextBatch())
      return {};
    check_dBatch();
    fill_dBatch();
    return tl_lorBatch_indices().cspan(m_currentBatchSize);
  }

private:
  bool hasNextBatch() const { return m_currentSubsetProcessedLORs < m_currentSubsetTotalLORs; }
  void check_hBatch() {
    if (mh_currentBatch.size() != m_batchSize)
      mh_currentBatch.resize(m_batchSize);
  }
  void fill_hBatch() {
    std::size_t begin = m_currentSubsetProcessedLORs;
    std::size_t end = std::min(m_currentSubsetProcessedLORs + m_batchSize, m_currentSubsetTotalLORs);
    if (begin == end) {
      m_currentBatchSize = 0;
      return;
    }
    core::MDSpan<3> span = core::MDSpan<3>::create(
        (MichInfoHub(m_mich).getBinNum() - 2 * m_binCut),
        core::mich::calViewNumInSubset(m_mich.polygon, m_mich.detector, m_subsetNum, m_currentSubset),
        m_validRingPair.size());
    tools::parallel_for_each(begin, end, [&](std::size_t index) {
      const auto &[binCutted, viewInSubset, ringPairIndex] = span.toIndex(index);
      const int binIndex = binCutted + m_binCut;
      const int viewIndex = viewInSubset * m_subsetNum + m_currentSubset;
      const auto &[ring1, ring2] = m_validRingPair[ringPairIndex];
      const int sliceIndex = IndexConverter(m_mich).getSliceFromRing1Ring2(ring1, ring2);
      const std::size_t lorId = IndexConverter(m_mich).getLORFromBVS(binIndex, viewIndex, sliceIndex);
      mh_currentBatch[index - begin] = lorId;
    });
    m_currentSubsetProcessedLORs += (end - begin);
    m_currentBatchSize = end - begin;
  }
  void check_dBatch() {
    tl_lorBatch_indices().reserve(m_batchSize);
    if (md_validRingPair.elements() != m_validRingPair.size() ||
        md_validRingPair.getDeviceIndex() != cuda_get_device_index_exept())
      md_validRingPair =
          make_cuda_sync_ptr_from_hcopy<core::Vector<int64_t, 2>>(m_validRingPair, "LORBatch_validRingPair");
  }
  void fill_dBatch() {
    std::size_t begin = m_currentSubsetProcessedLORs;
    std::size_t end = std::min(m_currentSubsetProcessedLORs + m_batchSize, m_currentSubsetTotalLORs);
    if (begin == end) {
      m_currentBatchSize = 0;
      return;
    }

    impl::d_fillLORBatch(begin, end, m_binCut, m_subsetNum, m_currentSubset, md_validRingPair.cspan(), m_mich,
                         tl_lorBatch_indices().data());
    m_currentSubsetProcessedLORs += (end - begin);
    m_currentBatchSize = end - begin;
  }

private:
  const core::MichDefine m_mich;
  int m_subsetNum{1};
  int m_binCut{0};
  int m_maxRingDiff = 0x3f3f3f3f;
  std::vector<core::Vector<int64_t, 2>> m_validRingPair;
  cuda_sync_ptr<core::Vector<int64_t, 2>> md_validRingPair;
  int m_currentSubset = 0;
  std::size_t m_currentSubsetProcessedLORs = 0;
  std::size_t m_currentSubsetTotalLORs = 0;
  std::size_t m_currentBatchSize = 0;
  constexpr static std::size_t m_batchSize = 1024 * 1024 * 5; // 5M
  std::vector<std::size_t> mh_currentBatch;
};

LORBatch::LORBatch(
    core::MichDefine __mich)
    : m_impl(std::make_unique<LORBatch_impl>(__mich)) {}

LORBatch::~LORBatch() {}

LORBatch &LORBatch::setSubsetNum(
    int num) {
  m_impl->setSubsetNum(num);
  return *this;
}
LORBatch &LORBatch::setBinCut(
    int binCut) {
  m_impl->setBinCut(binCut);
  return *this;
}
LORBatch &LORBatch::setMaxRingDiff(
    int maxRingDiff) {
  m_impl->setMaxRingDiff(maxRingDiff);
  return *this;
}

LORBatch &LORBatch::setCurrentSubset(
    int subset) {
  m_impl->setCurrentSubset(subset);
  return *this;
}
std::span<const std::size_t> LORBatch::nextHBatch() {
  return m_impl->nextHBatch();
}
std::span<const std::size_t> LORBatch::nextDBatch() {
  return m_impl->nextDBatch();
}

} // namespace openpni::experimental::node
