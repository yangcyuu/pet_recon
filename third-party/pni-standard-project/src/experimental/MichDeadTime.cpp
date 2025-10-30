#include "include/experimental/node/MichDeadTime.hpp"

#include "impl/DeadTimeFile.hpp"
#include "include/experimental/algorithms/EasyMath.hpp"
#include "include/experimental/core/Span.hpp"
#include "src/autogen/autogen_xml.hpp"
namespace openpni::experimental::node {

void save_dt_to_file(
    std::string const &filename, std::vector<float> const &cfdt, std::vector<float> const &rt) {
  impl::MichDeadTimeFile file(impl::MichDeadTimeFile::OpenMode::Write);
  file.setCFDTTable(cfdt);
  file.setRTTable(rt);
  file.open(filename);
}
void load_dt_from_file(
    std::string const &filename, std::vector<float> &cfdt, std::vector<float> &rt) {
  impl::MichDeadTimeFile file(impl::MichDeadTimeFile::OpenMode::Read);
  file.open(filename);
  cfdt = file.getCFDTTable();
  rt = file.getRTTable();
}

static std::vector<float> down_sampling_mich_by_block_slice(
    float const *__mich, core::MichDefine const &__michDef) {
  auto range = core::RangeGenerator::create(__michDef).allLORAndBinViewSlices();
  auto michInfo = core::MichInfoHub::create(__michDef);
  auto indexConverter = core::IndexConverter::create(__michDef);
  auto blockRingNum = michInfo.getBlockRingNum();

  std::vector<float> out_downSampleMich(blockRingNum * blockRingNum, 0.0f);

  for (const auto [lorID, binID, viewID, sliceID] : range) {
    auto [ring1, ring2] = indexConverter.getRing1Ring2FromSlice(sliceID);
    int blockRing1 = ring1 / michInfo.getCrystalNumZInBlock();
    int blockRing2 = ring2 / michInfo.getCrystalNumZInBlock();
    int dsSliceID = blockRing1 * blockRingNum + blockRing2;
    out_downSampleMich[dsSliceID] += __mich[lorID];
  }
  return out_downSampleMich;
}
class MichDeadTime_impl {
public:
  MichDeadTime_impl(
      core::MichDefine mich): m_michDefine{mich} {}
  ~MichDeadTime_impl() {}

public:
  void appendAcquisition(
      float const *prompMich, float const *delayMich, float scanTime, float activity) {
    m_dataItems.push_back({});
    m_dataItems.back().scanTime = scanTime;
    m_dataItems.back().activity = activity;
    m_dataItems.back().michP = down_sampling_mich_by_block_slice(prompMich, m_michDefine);
    m_dataItems.back().michR = down_sampling_mich_by_block_slice(delayMich, m_michDefine);

    m_cfdtTable.clear();
    m_rtTable.clear();
  }
  std::vector<float> dumpCFDTTable() {
    checkTable();
    return m_cfdtTable;
  }
  std::vector<float> dumpRTTable() {
    checkTable();
    return m_rtTable;
  }
  void dumpToFile(
      std::string const &filename) {
    checkTable();
  }
  void recoverFromFile(
      std::string const &filename) {
    std::vector<float> cfdt;
    std::vector<float> rt;
    load_dt_from_file(filename, cfdt, rt);

    const auto blockRingNum = core::MichInfoHub::create(m_michDefine).getBlockRingNum();
    const auto blockSliceNum = blockRingNum * blockRingNum;
#define ASSIGN_IF_SIZE_CORRECT(member, expectedSizeMultiple)                                                           \
  if (m_##member.size() % expectedSizeMultiple != 0)                                                                   \
    throw exceptions::algorithm_unexpected_condition(                                                                  \
        std::format("MichNormalizationFile " #member " size mismatch, expected to be divided by {}, got {}",           \
                    expectedSizeMultiple, m_##member.size()));

    ASSIGN_IF_SIZE_CORRECT(cfdtTable, blockSliceNum)
    ASSIGN_IF_SIZE_CORRECT(rtTable, blockSliceNum)
#undef ASSIGN_IF_SIZE_CORRECT
    if (cfdt.empty() || rt.empty())
      throw exceptions::algorithm_unexpected_condition("MichDeadTimeFile: empty table loaded.");
    if (cfdt.size() != rt.size())
      throw exceptions::algorithm_unexpected_condition("MichDeadTimeFile: table size mismatch.");
    m_cfdtTable = std::move(cfdt);
    m_rtTable = std::move(rt);
  }

private:
  void checkTable() {
    if (m_cfdtTable.empty() || m_rtTable.empty())
      calculateTables();
    if (m_cfdtTable.empty() || m_rtTable.empty())
      throw std::runtime_error("Failed to calculate dead time correction tables.");
  }
  void calculateTables() {
    const auto blockRingNum = core::MichInfoHub::create(m_michDefine).getBlockRingNum();
    const auto blockSliceNum = blockRingNum * blockRingNum;
    if (m_dataItems.empty())
      return;
    m_cfdtTable.resize(blockSliceNum * m_dataItems.size(), 0.0f);
    m_rtTable.resize(blockSliceNum * m_dataItems.size(), 0.0f);

    std::sort(m_dataItems.begin(), m_dataItems.end(), // 按照活度从大到小排序
              [](auto const &a, auto const &b) { return a.activity > b.activity; });
    const auto actualRate = calActualRate();
    const auto cfdt = calCFDT(actualRate);

    m_cfdtTable = std::move(cfdt);
    m_rtTable = std::move(actualRate);
  }

  std::vector<float> calActualRate() const {
    const auto blockRingNum = core::MichInfoHub::create(m_michDefine).getBlockRingNum();
    const auto blockSliceNum = blockRingNum * blockRingNum;
    auto span2 = core::MDSpan<2>::create(m_dataItems.size(), blockSliceNum);
    std::vector<float> actualRate(span2.totalSize(), 0.0f);
    for (const auto [acqIdx, sliceNow] : span2) {
      auto rate_p = m_dataItems[acqIdx].michP[sliceNow] / m_dataItems[acqIdx].scanTime;
      auto rate_d = m_dataItems[acqIdx].michR[sliceNow] / m_dataItems[acqIdx].scanTime;
      actualRate[span2(acqIdx, sliceNow)] = rate_p + rate_d;
    }
    return actualRate;
  }

  std::vector<float> calCFDT(
      std::vector<float> const &__actualRt) const {
    const auto blockRingNum = core::MichInfoHub::create(m_michDefine).getBlockRingNum();
    const auto blockSliceNum = blockRingNum * blockRingNum;

    std::vector<float> out_CFDT(blockSliceNum * m_dataItems.size(), 0.0f);
    for (const auto dsslIdx : std::views::iota(0u, blockSliceNum)) {
      algorithms::LinearFittingHelper<float, algorithms::LinearFitting_WithBias> linearFit;
      // linearFit to cal CFDT
      const auto acquisitionNum = m_dataItems.size();
      for (auto p : // 选择最后几个点（活度最低的几个扫描）进行线性拟合
           std::views::iota(0ull, acquisitionNum) | std::views::reverse | std::views::take(m_maxPointsForFitting)) {
        linearFit.add(m_dataItems[p].activity, __actualRt[dsslIdx * acquisitionNum + p]);
      }
      // cal CFDT by LinearFitRate / actualRate
      for (auto acq : std::views::iota(0ull, acquisitionNum)) {
        double idealRt = linearFit.predict(m_dataItems[acq].activity);
        out_CFDT[dsslIdx * acquisitionNum + acq] = idealRt / __actualRt[dsslIdx * acquisitionNum + acq];
      }
    }
    return out_CFDT;
  }

public:
  struct DataItem {
    std::vector<float> michP;
    std::vector<float> michR;
    float scanTime;
    float activity;
  };

private:
  core::MichDefine m_michDefine;
  std::vector<DataItem> m_dataItems;

  int m_maxPointsForFitting = 10;

  std::vector<float> m_cfdtTable;
  std::vector<float> m_rtTable;
};

MichDeadTime::MichDeadTime(
    core::MichDefine mich)
    : m_impl(std::make_unique<MichDeadTime_impl>(mich)) {}
MichDeadTime::~MichDeadTime() {}
void MichDeadTime::appendAcquisition(
    float const *prompMich, float const *delayMich, float scanTime, float activity) {
  m_impl->appendAcquisition(prompMich, delayMich, scanTime, activity);
}
std::vector<float> MichDeadTime::dumpCFDTTable() {
  return m_impl->dumpCFDTTable();
}
std::vector<float> MichDeadTime::dumpRTTable() {
  return m_impl->dumpRTTable();
}
void MichDeadTime::dumpToFile(
    std::string const &filename) {
  m_impl->dumpToFile(filename);
}
void MichDeadTime::recoverFromFile(
    std::string const &filename) {
  m_impl->recoverFromFile(filename);
}

} // namespace openpni::experimental::node
