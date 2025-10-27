#pragma once
#include "Copy.h"
#include "MichFactors.hpp"
#include "Projection.h"
#include "include/Exceptions.hpp"
#include "include/basic/CudaPtr.hpp"
#include "include/basic/PetDataType.h"
#include "include/experimental/core/Image.hpp"
#include "include/experimental/core/Vector.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/node/MichAttn.hpp"
#include "include/experimental/node/MichCrystal.hpp"
#include "include/experimental/tools/Parallel.cuh"
#include "include/experimental/tools/Parallel.hpp"
enum WhereIsData { HOST, DEVICE };
template <typename T>
struct RawInputPtr {
  T *raw_ptr = nullptr;
  WhereIsData where = HOST;
};
template <typename T>
struct RawInputItem {
  RawInputItem(
      std::string __name = "")
      : owned_host_data(nullptr)
      , owned_device_data(__name) {}
  RawInputPtr<T> raw_data;
  std::size_t expected_elements;
  std::unique_ptr<std::remove_const_t<T>[]> owned_host_data;
  openpni::cuda_sync_ptr<std::remove_const_t<T>> owned_device_data;
};
template <typename T>
inline T *expected_host_pointer_nullable(
    RawInputItem<T> &item) {
  if (item.raw_data.where == HOST && item.raw_data.raw_ptr) {
    return item.raw_data.raw_ptr;
  }
  if (item.owned_host_data) {
    return item.owned_host_data.get();
  }

  if (item.raw_data.where == DEVICE && item.raw_data.raw_ptr) {
    item.owned_host_data = device_deep_copy_to_host(std::span<T const>(item.raw_data.raw_ptr, item.expected_elements));
    return item.owned_host_data.get();
  }
  if (item.owned_device_data && item.owned_device_data.elements() == item.expected_elements) {
    item.owned_host_data = device_deep_copy_to_host(item.owned_device_data);
    return item.owned_host_data.get();
  }

  return nullptr;
}
template <typename T>
inline T *expected_host_pointer(
    RawInputItem<T> &item) {
  T *ptr = expected_host_pointer_nullable(item);
  if (ptr == nullptr)
    throw openpni::exceptions::algorithm_unexpected_condition("MichScatter: expected_host_pointer: no valid data");
  return ptr;
}

template <typename T>
inline T *expected_device_pointer_nullable(
    RawInputItem<T> &item) {
  if (item.raw_data.where == DEVICE && item.raw_data.raw_ptr) {
    return item.raw_data.raw_ptr;
  }
  if (item.owned_device_data && item.owned_device_data.elements() == item.expected_elements) {
    return item.owned_device_data.get();
  }

  if (item.raw_data.where == HOST && item.raw_data.raw_ptr) {
    item.owned_device_data =
        openpni::make_cuda_sync_ptr_from_hcopy(std::span<T const>(item.raw_data.raw_ptr, item.expected_elements),
                                               item.owned_device_data.getName() + "_fromHost");
    return item.owned_device_data.get();
  }
  if (item.owned_host_data && item.owned_host_data.get()) {
    item.owned_device_data =
        openpni::make_cuda_sync_ptr_from_hcopy(std::span<T const>(item.owned_host_data.get(), item.expected_elements),
                                               item.owned_device_data.getName() + "_fromHost");
    return item.owned_device_data.get();
  }

  return nullptr;
}
template <typename T>
inline T *expected_device_pointer(
    RawInputItem<T> &item) {
  T *ptr = expected_device_pointer_nullable(item);
  if (ptr == nullptr)
    throw openpni::exceptions::algorithm_unexpected_condition("MichScatter: expected_device_pointer: no valid data");
  return ptr;
}
namespace openpni::experimental::node {
//======= temp
enum DataFormat { MICH, LISTMODE };
struct sssBatchLORGetor {
  const float *md_michData;
  std::span<openpni::basic::Listmode_t const> md_listmodeData;
  DataFormat m_dataFormat;

  void bindDMichData(const float *d_michData);
  void bindDListmodeData(std::span<openpni::basic::Listmode_t const> d_listmodeData);
  float const *getLORBatch(std::size_t __sliceBegin, std::size_t __sliceEnd, core::MichDefine __michDefine);

private:
  openpni::cuda_sync_ptr<float> m_tempValues{"sssBatchLORGetor_tempValues"};
};

inline void sssBatchLORGetor::bindDMichData(
    const float *d_michData) {
  md_michData = d_michData;
  md_listmodeData = std::span<openpni::basic::Listmode_t const>();
  m_dataFormat = MICH;
}
inline void sssBatchLORGetor::bindDListmodeData(
    std::span<openpni::basic::Listmode_t const> d_listmodeData) {
  md_listmodeData = d_listmodeData;
  md_michData = nullptr;
  m_dataFormat = LISTMODE;
}
inline float const *sssBatchLORGetor::getLORBatch(
    std::size_t __sliceBegin, std::size_t __sliceEnd, core::MichDefine __michDefine) {
  if (__sliceBegin >= __sliceEnd)
    throw openpni::exceptions::algorithm_unexpected_condition("sssBatchLORGetor::getLORBatch: invalid slice");
  m_tempValues.reserve(MichInfoHub(__michDefine).getBinNum() * MichInfoHub(__michDefine).getViewNum() *
                       (__sliceEnd - __sliceBegin));
  example::d_parralel_fill(m_tempValues.get(), 0.0f, m_tempValues.elements());
  if (m_dataFormat == DataFormat::MICH) {
    impl::d_redirect_from_mich_from_slice_range(__sliceBegin, __sliceEnd, md_michData, m_tempValues.get(),
                                                __michDefine);
  } else if (m_dataFormat == DataFormat::LISTMODE) {
    impl::d_redirect_from_mich_from_slice_range(__sliceBegin, __sliceEnd, md_listmodeData, m_tempValues.get(),
                                                __michDefine);
  } else {
    throw openpni::exceptions::algorithm_unexpected_condition("sssBatchLORGetor::getLORBatch: no valid data source");
  }
  return m_tempValues.get();
}
//========
} // namespace openpni::experimental::node
namespace openpni::experimental::node {
struct ScatterPoint {
  openpni::experimental::core::Vector<float, 3> sssPosition; // scatter point position
  float mu;                                                  // linear attenuation factor at scatter point
};
struct sssDataView {
  float *__out_d_scatterValue;
  const float *__sssAttnCoff;
  const float *__sssEmissionCoff;
  const float *__scannerEffTable;
  const core::CrystalGeom *__d_crystalGeometry;
  const ScatterPoint *__scatterPoints;

  const core::MichDefine __michDefine;
  const core::Vector<float, 3> m_scatterEffTableEnergy; // low, high, interval

  int __minSectorDifference;
  size_t __countScatter;
  double __commonfactor;
};
struct sssTOFDataView {
  float *__out_d_dsTOFBinSSSValue;
  const float *__eMap;
  const float *__sssAttnCoff;
  const float *__scannerEffTable;
  const float *__gaussBlurCoff;
  const core::CrystalGeom *__d_dsCrystalGeometry;
  const ScatterPoint *__scatterPoints;

  const core::MichDefine __dsmichDefine;
  const core::Vector<float, 3> m_scatterEffTableEnergy; // low, high, interval
  const core::Grids<3> __emapGrids;

  int __TOFBinNum;
  int __gaussSize;
  std::size_t __countScatter;
  float __TOFBinWidth; // mm
  float __crystalArea; // notice here crystal still mean origin crystal, not ds crystal
  double __commonfactor;
};

struct scatterTOFParams {
  float m_timeBinWidth;
  float m_timeBinStart;
  float m_timeBinEnd;
  float m_systemTimeRes_ns;
  int m_tofBinNum = 0;

  scatterTOFParams() {};
  scatterTOFParams(
      float timeBinWidth, float timeBinStart, float timeBinEnd, float systemTimeRes_ns)
      : m_timeBinWidth(timeBinWidth)
      , m_timeBinStart(timeBinStart)
      , m_timeBinEnd(timeBinEnd)
      , m_systemTimeRes_ns(systemTimeRes_ns) {
    calTofBinNum();
  }

  void calTofBinNum() {
    m_tofBinNum = ceil((m_timeBinEnd - m_timeBinStart) / m_timeBinWidth);
    m_tofBinNum -= 1 - (m_tofBinNum % 2); // 让 bin 数为奇数
  }
};
class MichScatter_impl {
public:
  explicit MichScatter_impl(
      core::MichDefine __mich)
      : m_michDefine(__mich)
      , m_michCrystal(__mich) {}
  ~MichScatter_impl() {}
  //====== params
  void setScatterPointsThreshold(double v);
  void setTailFittingThreshold(double v);
  void setScatterEnergyWindow(core::Vector<double, 3> windows);
  void setScatterEnergyWindow(double low, double high, double resolution);
  void setScatterEffTableEnergy(core::Vector<double, 3> energies);
  void setScatterEffTableEnergy(double low, double high, double interval);
  void setMinSectorDifference(int v);
  void setScatterPointGrid(core::Grids<3> grid);
  void setTOFParams(double timeBinWidth, double timeBinStart, double timeBinEnd, double systemTimeRes_ns);
  //====== inputs
  // The input pointers below only hold the access to the data, no copy inside.
  // If emission h_data or d_data is nullptr, it means a blank emission map (all zeros), then function
  // will ignore the map and set all scatter factors to zero.
  void bindNorm(MichNormalization *norm);
  void bindAttnCoff(MichAttn *h_data);
  void bindRandom(MichRandom *random);
  void bindHEmissionMap(core::Grids<3, float> emap, float const *h_data);
  void bindDEmissionMap(core::Grids<3, float> emap, float const *d_data);
  // test
  void bindHPromptMich(float const *h_promptMich);
  void bindDPromptMich(float const *d_promptMich);
  void bindDListmode(std::span<basic::Listmode_t const> listmodeFiles);

  float const *getHScatterFactorsBatch(std::span<core::MichStandardEvent const> events);
  float const *getDScatterFactorsBatch(std::span<core::MichStandardEvent const> events);
  std::unique_ptr<float[]> dumpHScatterMich();
  std::unique_ptr<MichScatter_impl> copy();

private:
  // check if params are given
  void checkHScatterMich();
  void checkDScatterMich();
  void checkPreGenerateFlags();
  void checkTOFFlags();
  void checkOrThrowGenerateFlags();
  void generateScatterMich();

private:
  float const *h_getEmissionMap();
  float const *d_getEmissionMap();
  float const *h_getScatterMich();
  float const *d_getScatterMich();

private: // sss main
  void sssPreGenerate();
  cuda_sync_ptr<float> d_generateScatterMich();
  cuda_sync_ptr<float> d_generateScatterTableTOF();

private:
  const core::MichDefine m_michDefine;
  MichCrystal m_michCrystal;
  scatterTOFParams m_sssTOFParams;

  core::Vector<float, 3> m_scatterEnergyWindow;   // low, high, resolution
  core::Vector<float, 3> m_scatterEffTableEnergy; // low, high, interval
  int m_minSectorDifference;
  double m_tailFittingThreshold;
  double m_scatterPointsThreshold;

  bool m_preDataGenerated = false;
  bool m_scatterMichGenerated = false;
  bool m_TOFModel = false;

private:
  std::unique_ptr<float[]> mh_scatterMich;
  cuda_sync_ptr<float> md_scatterMich{"MichScatter_scatterMich"};

  std::vector<float> mh_tempScatFactors;
  cuda_sync_ptr<float> md_tempScatFactors{"MichScatter_tempScatFactors"};

  MichNormalization *m_norm;
  MichRandom *m_random;
  MichAttn *m_AttnCoff;
  // RawInputItem<float const> m_promptMich{"MichScatter_promptMich"};
  //  test
  sssBatchLORGetor m_lorGetor;

  RawInputItem<float const> m_emissionMap{"MichScatter_emissionMap"};
  core::Grids<3> m_emissionMapGrids;
  std::optional<core::Grids<3>> m_scatterPointGrid;

  // pre-generate data
  int m_gaussSize;
  double m_commonFactor;
  std::size_t m_scatterCount;
  core::MichDefine m_dsmichDefine;
  cuda_sync_ptr<ScatterPoint> md_scatterPoints{"MichScatter_scatterPoints"};
  cuda_sync_ptr<float> md_scannerEffTable{"MichScatter_scannerEffTable"};
  cuda_sync_ptr<float> md_gaussBlurCoff{"MichScatter_gaussBlurCoff"};
  cuda_sync_ptr<float> md_sssAttnCoff{"MichScatter_sssAttnCoff"};
  cuda_sync_ptr<float> md_sssEmissionCoff{"MichScatter_sssEmissionCoff"};
  cuda_sync_ptr<float> md_trueValueBySlice{"MichScatter_trueValueBySlice"};
};

inline void MichScatter_impl::setScatterPointsThreshold(
    double v) {
  m_scatterPointsThreshold = v;
  m_preDataGenerated = false;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::setTailFittingThreshold(
    double v) {
  m_tailFittingThreshold = v;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::setScatterEnergyWindow(
    core::Vector<double, 3> windows) {
  m_scatterEnergyWindow = windows.template to<float>();
  m_preDataGenerated = false;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::setScatterEnergyWindow(
    double low, double high, double resolution) {
  m_scatterEnergyWindow = core::Vector<double, 3>::create(low, high, resolution).template to<float>();
  m_preDataGenerated = false;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::setScatterEffTableEnergy(
    core::Vector<double, 3> energies) {
  m_scatterEffTableEnergy = energies.template to<float>();
  m_preDataGenerated = false;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::setMinSectorDifference(
    int v) {
  m_minSectorDifference = v;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::setScatterEffTableEnergy(
    double low, double high, double interval) {
  m_scatterEffTableEnergy = core::Vector<double, 3>::create(low, high, interval).template to<float>();
  m_preDataGenerated = false;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::setTOFParams(
    double timeBinWidth, double timeBinStart, double timeBinEnd, double systemTimeRes_ns) {
  m_sssTOFParams = scatterTOFParams(timeBinWidth, timeBinStart, timeBinEnd, systemTimeRes_ns);
  checkTOFFlags();
  m_preDataGenerated = false;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::bindAttnCoff(
    MichAttn *h_data) {
  m_AttnCoff = h_data;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::bindNorm(
    MichNormalization *norm) {
  m_norm = norm;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::bindRandom(
    MichRandom *h_data) {
  m_random = h_data;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::bindHEmissionMap(
    core::Grids<3, float> emap, float const *h_data) {
  m_emissionMap.raw_data = RawInputPtr<float const>{h_data, HOST};
  m_emissionMap.expected_elements = emap.totalSize();
  m_emissionMap.owned_host_data.reset();
  m_emissionMap.owned_device_data.clear();
  m_emissionMapGrids = emap;
  m_scatterMichGenerated = false;
}
inline void MichScatter_impl::bindDEmissionMap(
    core::Grids<3, float> emap, float const *d_data) {
  m_emissionMap.raw_data = RawInputPtr<float const>{d_data, DEVICE};
  m_emissionMap.expected_elements = emap.totalSize();
  m_emissionMap.owned_host_data.reset();
  m_emissionMap.owned_device_data.clear();
  m_emissionMapGrids = emap;
  m_scatterMichGenerated = false;
}
// inline void MichScatter_impl::bindHPromptMich(
//     float const *h_promptMich) {
//   m_promptMich.raw_data = RawInputPtr<float const>{h_promptMich, HOST};
//   m_promptMich.expected_elements = core::MichInfoHub::create(m_michDefine).getMichSize();
//   m_promptMich.owned_host_data.reset();
//   m_promptMich.owned_device_data.clear();
//   m_scatterMichGenerated = false;
// }
// inline void MichScatter_impl::bindDPromptMich(
//     float const *d_promptMich) {
//   m_promptMich.raw_data = RawInputPtr<float const>{d_promptMich, DEVICE};
//   m_promptMich.expected_elements = core::MichInfoHub::create(m_michDefine).getMichSize();
//   m_promptMich.owned_host_data.reset();
//   m_promptMich.owned_device_data.clear();
//   m_scatterMichGenerated = false;
// }
inline void MichScatter_impl::bindHPromptMich(
    float const *h_promptMich) {
  m_lorGetor.bindDMichData(h_promptMich);
}
inline void MichScatter_impl::bindDPromptMich(
    float const *d_promptMich) {
  m_lorGetor.bindDMichData(d_promptMich);
}
inline void MichScatter_impl::bindDListmode(
    std::span<basic::Listmode_t const> listmodeFiles) {
  m_lorGetor.bindDListmodeData(listmodeFiles);
}
inline std::unique_ptr<float[]> MichScatter_impl::dumpHScatterMich() {
  generateScatterMich();
  checkOrThrowGenerateFlags();
  if (mh_scatterMich)
    return host_deep_copy(mh_scatterMich.get(), core::MichInfoHub::create(m_michDefine).getMichSize());
  else if (md_scatterMich)
    return device_deep_copy_to_host(md_scatterMich);
  else
    throw openpni::exceptions::algorithm_unexpected_condition("MichScatter: dumpHScatterMich: no valid data");
}
inline float const *MichScatter_impl::h_getEmissionMap() {
  return expected_host_pointer(m_emissionMap);
}
inline float const *MichScatter_impl::d_getEmissionMap() {
  return expected_device_pointer_nullable(m_emissionMap);
}
inline float const *MichScatter_impl::h_getScatterMich() {
  checkHScatterMich();
  return mh_scatterMich.get();
}
inline float const *MichScatter_impl::d_getScatterMich() {
  checkDScatterMich();
  return md_scatterMich.get();
}
inline const float *MichScatter_impl::getHScatterFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  generateScatterMich();
  checkOrThrowGenerateFlags();
  if (mh_tempScatFactors.size() < events.size()) {
    mh_tempScatFactors.resize(events.size());
  }
  checkHScatterMich();
  tools::parallel_for_each(events.size(), [&](std::size_t index) {
    auto lorIndex = core::IndexConverter::create(m_michDefine)
                        .getLORIDFromRectangleID(events[index].crystal1, events[index].crystal2);
    mh_tempScatFactors[index] = mh_scatterMich[lorIndex];
  });
  return mh_tempScatFactors.data();
}
inline const float *MichScatter_impl::getDScatterFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  generateScatterMich();
  checkOrThrowGenerateFlags();
  md_tempScatFactors.reserve(events.size());
  checkDScatterMich();
  impl::d_getDScatterFactorsBatch(md_tempScatFactors.get(), md_scatterMich.get(), events, m_michDefine);
  return md_tempScatFactors.get();
}

inline void MichScatter_impl::checkHScatterMich() {
  if (mh_scatterMich)
    return;
  if (md_scatterMich) {
    mh_scatterMich = device_deep_copy_to_host(md_scatterMich);
    return;
  }
  // not generated yet
  generateScatterMich();
  checkHScatterMich();
}
inline void MichScatter_impl::checkDScatterMich() {
  if (md_scatterMich)
    return;
  if (mh_scatterMich) {
    md_scatterMich = openpni::make_cuda_sync_ptr_from_hcopy(
        std::span<float const>(mh_scatterMich.get(), core::MichInfoHub::create(m_michDefine).getMichSize()),
        "MichScatter_scatterMich_fromHost");
    return;
  }
  // not generated yet
  generateScatterMich();
  checkDScatterMich();
}
inline void MichScatter_impl::checkPreGenerateFlags() {
  if (m_preDataGenerated)
    return;
  else {
    sssPreGenerate();
    checkPreGenerateFlags();
  }
}

inline void MichScatter_impl::checkTOFFlags() {
  if (m_TOFModel)
    return;
  else {
    if (m_sssTOFParams.m_tofBinNum > 0) {
      auto dsPolygon = m_michDefine.polygon;
      auto dsDetector = m_michDefine.detector;
      dsDetector.crystalNumU = 1;
      dsDetector.crystalNumV = 1;
      m_dsmichDefine = core::MichDefine{dsPolygon, dsDetector};
      m_TOFModel = true;
      checkTOFFlags();
    }
  }
}
inline void MichScatter_impl::checkOrThrowGenerateFlags() {
  if (!m_scatterMichGenerated)
    throw openpni::exceptions::algorithm_unexpected_condition("No scatter mich generated");
}
inline void MichScatter_impl::generateScatterMich() {
  if (m_scatterMichGenerated)
    return;
  checkTOFFlags();
  if (m_TOFModel) {
    auto dsmichInfo = core::MichInfoHub::create(m_dsmichDefine);
    auto michInfo = core::MichInfoHub::create(m_michDefine);
    md_scatterMich = make_cuda_sync_ptr<float>(dsmichInfo.getBinNum() * dsmichInfo.getViewNum() *
                                                   michInfo.getSliceNum() * m_sssTOFParams.m_tofBinNum,
                                               "MichScatter_scatterMichs_TOF");
    md_scatterMich.memset(0);
    if (m_emissionMap.raw_data.raw_ptr) {
      checkPreGenerateFlags();
      md_scatterMich = d_generateScatterTableTOF();
    } else {
      PNI_DEBUG("MichScatter: No emission map bound, emission map skipped.\n");
    }
  } else {
    md_scatterMich =
        make_cuda_sync_ptr<float>(core::MichInfoHub::create(m_michDefine).getMichSize(), "MichScatter_scatterMich");
    md_scatterMich.memset(0);

    if (m_emissionMap.raw_data.raw_ptr) {
      checkPreGenerateFlags();
      md_scatterMich = d_generateScatterMich();
    } else {
      PNI_DEBUG("MichScatter: No emission map bound, emission map skipped.\n");
    }
  }
  m_scatterMichGenerated = true;
}
std::unique_ptr<MichScatter_impl> MichScatter_impl::copy() {
  auto newImpl = std::make_unique<MichScatter_impl>(m_michDefine);
  newImpl->m_sssTOFParams = m_sssTOFParams;
  newImpl->m_scatterEnergyWindow = m_scatterEnergyWindow;
  newImpl->m_scatterEffTableEnergy = m_scatterEffTableEnergy;
  newImpl->m_minSectorDifference = m_minSectorDifference;
  newImpl->m_tailFittingThreshold = m_tailFittingThreshold;
  newImpl->m_scatterPointsThreshold = m_scatterPointsThreshold;

  newImpl->m_preDataGenerated = m_preDataGenerated;
  newImpl->m_scatterMichGenerated = m_scatterMichGenerated;
  newImpl->m_TOFModel = m_TOFModel;

  return newImpl;
}
void MichScatter_impl::setScatterPointGrid(
    core::Grids<3> grid) {
  m_scatterPointGrid = grid;
  m_preDataGenerated = false;
  m_scatterMichGenerated = false;
}
} // namespace openpni::experimental::node
