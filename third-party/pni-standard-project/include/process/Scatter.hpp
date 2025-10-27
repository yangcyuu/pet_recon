#pragma once
#include <random>
#include <ranges>

#include "../basic/DataView.hpp"
#include "../basic/Image.hpp"
#include "../example/PolygonalSystem.hpp"
#include "../math/EMStep.hpp"
#include "../math/Geometry.hpp"
#include "../process/EM.hpp"
#include "Attenuation.hpp"
#include "Foreach.hpp"
namespace openpni::process {
struct ScatterPoint {
  basic::CrystalGeometry sssPosition; // scatter point position
  float mu;                           // linear attenuation factor at scatter point
  float volume;                       // 单位立方毫米
};

template <typename ImageValueType>
struct _SSSDataView {
  ImageValueType *out_scatterValue;

  ImageValueType *in_sssAttnCoff;
  ImageValueType *in_sssEmission;
  ImageValueType *in_attnCutBedCoff;
  ImageValueType *in_promptMich;
  ImageValueType *in_randMich; // default : 0,must have in v3,需要smooth预处理
  ImageValueType *in_normMich; // default : 1，only use norm mich with: float* blockProfA, float* blockProfT,
                               // float*crystalFct，它们是生成归一化mich是伴随生成的部分组件
  basic::CrystalGeometry *in_crystalGeometry;

  Image3DSpan<const ImageValueType> in_d_attnMapImage3dSpan; // attnuation map and attnCoff without cutbed
  Image3DSpan<const ImageValueType> in_d_osemImage3dSpan;    // prompt mich and its reconImg without SSS

  float *scannerEffTable;
  ScatterPoint *scatterPoints;
  basic::Vec3<double> scatterEnergyWindow;   // 扫描器效率表能量范围,x is low, y is high, zis resolution
  basic::Vec3<double> scannerEffTableEnergy; // x is low,y is high,z is interval
  basic::Vec3<float> sssGridSize;            // x,y,z方向上格点间距,单位毫米
  std::size_t countScatter;
  int minSectorDifference;
  double scatterTailFittingThreshold;
  example::PolygonalSystem polygon;
  basic::DetectorGeometry detectorGeometry; // 探测器几何信息
};

template <typename ImageValueType, typename ProjectionMethod>
struct _SSSTOFDataView {
  ImageValueType *out_dsSSSTOFfullSliceWithTOFBin; // size = sliceNum * dsBinNum * dsViewNum * tofBinNum

  ImageValueType *in_sssAttnCoff;
  ImageValueType *in_attnCutBedCoff;
  ImageValueType *in_promptMich;
  ImageValueType *in_randMich; // default : 0,must have in v3,需要smooth预处理
  ImageValueType *in_normMich; // default : 1，only use norm mich with: float* blockProfA, float* blockProfT,
                               // float*crystalFct，它们是生成归一化mich是伴随生成的部分组件

  ScatterPoint *scatterPoints;
  float *in_scannerEffTable;
  double *in_guassBlur;
  basic::CrystalGeometry *in_crystalGeometry;
  basic::CrystalGeometry *in_dsCrystalGeometry;

  ProjectionMethod projector;
  Image3DSpan<const ImageValueType> in_d_attnMapImage3dSpan; // attnuation map and attnCoff without cutbed
  Image3DSpan<const ImageValueType> in_d_emap3dSpan;         // prompt mich and its reconImg without SSS

  basic::Vec3<double> scatterEnergyWindow;   // 扫描器效率表能量范围,x is low, y is high, zis resolution
  basic::Vec3<double> scannerEffTableEnergy; // x is low,y is high,z is interval
  basic::Vec3<float> sssGridSize;            // x,y,z方向上格点间距,单位毫米
  std::size_t countScatter;
  int minSectorDifference;
  double scatterTailFittingThreshold;
  float tofBinWidth;
  int tofBinNum;
  int gaussSize;
  example::PolygonalSystem polygon;
  basic::DetectorGeometry detectorGeometry;   // 探测器几何信息
  example::PolygonalSystem dsPolygon;         // down-sampled后 多边形系统信息
  basic::DetectorGeometry dsDetectorGeometry; // down-sampled后 探测器几何信息
};

namespace scatter {
using CrystalInfo = basic::CrystalInfo;
template <typename PairDataView>
struct ViewForPointsCartesianProduct // 对于任意给定的Position组
                                     // 两两计算正投影
{
  using _Event = basic::Event<float>;
  const ScatterPoint *pointsScatter;
  PairDataView pairDataView;
  std::size_t countScatter;

  __PNI_CUDA_MACRO__ std::size_t size() const { return pairDataView.crystalNum() * countScatter; }
  __PNI_CUDA_MACRO__ _Event at(
      std::size_t __dataIndex) const // 从两个晶体位置到 event
  {
    _Event result;
    const auto indexPairDataView = __dataIndex / countScatter;
    const auto indexScatter = __dataIndex % countScatter;
    result.crystal1 = pairDataView.crystal(indexPairDataView);
    result.crystal2.geometry = &(pointsScatter[indexScatter].sssPosition);
    result.crystal1.tof_deviation = nullptr; // ViewForPointsCartesianProduct
                                             // 数据集不包含TOF信息
    result.crystal2.tof_deviation = nullptr; // ViewForPointsCartesianProduct
                                             // 数据集不包含TOF信息
    result.crystal1.tof_mean = nullptr;      // ViewForPointsCartesianProduct
                                             // 数据集不包含TOF信息
    result.crystal2.tof_mean = nullptr;      // ViewForPointsCartesianProduct
                                             // 数据集不包含TOF信息
    result.time1_2 = nullptr;                // ViewForPointsCartesianProduct
                                             // 数据集不包含时间信息
    result.gain = nullptr;                   // ViewForPointsCartesianProduct
                                             // 数据集不包含增益信息
    result.eventIndex = __dataIndex;
    return result;
  }
};

__PNI_CUDA_MACRO__ inline double calTotalComptonCrossSection511KeVRelative(
    double scatterEnergy) {
  const double a = scatterEnergy / 511.0;
  // Klein-Nishina formula for a=1 & devided with
  // 0.75 == (40 - 27*log(3)) / 9
  const double prefactor = 9.0 / (-40 + 27 * log(3.));

  return // checked this in Mathematica
      prefactor *
      (((-4 - a * (16 + a * (18 + 2 * a))) / ((1 + 2 * a) * (1 + 2 * a)) + ((2 + (2 - a) * a) * log(1 + 2 * a)) / a) /
       (a * a));
}

__PNI_CUDA_MACRO__ inline double calTotalComptonCrossSection(
    double energy) {
  constexpr double PI = 3.1415926535898;
  constexpr double Re = 2.818E-13; // cm
  double k = energy / 511.f;
  double a = log(1 + 2 * k);
  double prefactor = 2 * PI * Re * Re;
  return prefactor * ((1.0 + k) / (k * k) * (2.0 * (1.0 + k) / (1.0 + 2.0 * k) - a / k) + a / (2.0 * k) -
                      (1.0 + 3.0 * k) / (1.0 + 2.0 * k) / (1.0 + 2.0 * k));
}

__PNI_CUDA_MACRO__ inline double calTotalAttenInScatterEnergy(
    double totalAtten511, double scatterEnergy) {
  return pow(totalAtten511, calTotalComptonCrossSection511KeVRelative(scatterEnergy));
}

__PNI_CUDA_MACRO__ inline double calDiffCrossSection(
    double scatCosTheta) {
  // Kelin-Nishina formula. re is classical electron radius
  const double Re = 2.818E-13;               // cm
  double waveRatio = 1 / (2 - scatCosTheta); //  lamda/lamda'
  return 0.5 * Re * Re * waveRatio * waveRatio * (waveRatio + 1 / waveRatio + scatCosTheta * scatCosTheta - 1);
}

struct _ScatterPointsGenerateRules {
  std::vector<p3df> center(
      basic::Image3DGeometry _gridGeometry) const {
    return _gridGeometry.voxel_centers();
  }
  std::vector<p3df> centerRandom(
      basic::Image3DGeometry _gridGeometry) const {
    auto points = _gridGeometry.voxel_centers();
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::mt19937 rng(std::random_device{}());
    auto nextRandomPoint = [&]() { return basic::make_vec3<float>(dist(rng), dist(rng), dist(rng)); };
    std::transform(points.begin(), points.end(), points.begin(),
                   [&](const p3df &p) { return p + nextRandomPoint() * _gridGeometry.voxelSize; });
    return points;
  }

  std::vector<ScatterPoint> initScatterPoint(
      float *attnMap, const openpni::basic::Image3DGeometry attnMapSize, basic::Vec3<float> sssGrid,
      double scatterPointThreshold) {
    std::vector<ScatterPoint> result;
    basic::Vec3<float> gridNum;
    gridNum = attnMapSize.voxelNum * attnMapSize.voxelSize / sssGrid;
    int totalNum = (int)ceil(gridNum.x) * (int)ceil(gridNum.y) * (int)ceil(gridNum.z);
    result.reserve(totalNum);
    std::cout << result.size() << " scatter points reserved." << std::endl;
    std::cout << "Notice: We will select scatter point randomly. " << std::endl;
    srand(0);

    // current grid minimum border
    basic::Vec3<float> Lim = {0.0, 0.0, 0.0};
    basic::Vec3<float> upperLim;
    basic::Vec3<float> pointRand;
    basic::Vec3<int> voxelIndex;
    ScatterPoint scatterPointTemp;
    while (Lim.x < attnMapSize.voxelSize.x * attnMapSize.voxelNum.x) {
      upperLim.x = std::min(Lim.x + sssGrid.x, (float)(attnMapSize.voxelSize.x * attnMapSize.voxelNum.x));
      Lim.y = 0;
      while (Lim.y < attnMapSize.voxelSize.y * attnMapSize.voxelNum.y) {
        upperLim.y = std::min(Lim.y + sssGrid.y, (float)(attnMapSize.voxelSize.y * attnMapSize.voxelNum.y));
        Lim.z = 0;
        while (Lim.z < attnMapSize.voxelSize.z * attnMapSize.voxelNum.z) {
          upperLim.z = std::min(Lim.z + sssGrid.z, (float)(attnMapSize.voxelSize.z * attnMapSize.voxelNum.z));
          pointRand = (upperLim - Lim) * 1.f * rand() / (float)RAND_MAX;
          pointRand = pointRand + Lim;
          voxelIndex = basic::make_vec3<int>(pointRand / attnMapSize.voxelSize);
          if (attnMap[voxelIndex.z * attnMapSize.voxelNum.x * attnMapSize.voxelNum.y +
                      voxelIndex.y * attnMapSize.voxelNum.x + voxelIndex.x] >= scatterPointThreshold) {
            scatterPointTemp.sssPosition.position = basic::make_vec3<float>(
                pointRand - attnMapSize.voxelSize * attnMapSize.voxelNum * 0.5 + attnMapSize.centre());
            // cm-1
            scatterPointTemp.mu = attnMap[voxelIndex.z * attnMapSize.voxelNum.x * attnMapSize.voxelNum.y +
                                          voxelIndex.y * attnMapSize.voxelNum.x + voxelIndex.x] *
                                  10;
            result.push_back(scatterPointTemp);
          }
          Lim.z += sssGrid.z;
        }
        Lim.y += sssGrid.y;
      }
      Lim.x += sssGrid.x;
    }
    result.shrink_to_fit();
    std::cout << result.size() << " scatter points generated." << std::endl;
    if (result.size() <= 0) {
      std::cout << "no good scatter point. At initScatterPoint." << std::endl;
      return std::vector<ScatterPoint>{};
    }
    return result;
  }
};
inline constexpr _ScatterPointsGenerateRules ScatterPointsGenerateRules{};
inline std::vector<ScatterPoint> isScatterPoint(
    const float volume, // which is gridGeometry's voxel volume
    const std::vector<p3df> &_allRandomPoints, Image3DInputSpan<float> __in_attnMapImage3dSpan,
    double scatterPointThreshold) {
  std::vector<ScatterPoint> result;
  for_each(_allRandomPoints.size(), [&](std::size_t idx) {
    auto randomPoint = _allRandomPoints[idx];

    int x = floor((randomPoint.x + 80) / 0.5);
    int y = floor((randomPoint.y + 80) / 0.5);
    int z = floor((randomPoint.z + 80) / 0.5);

    auto voxelIndex = basic::make_vec3<int>(x, y, z);

    // auto voxelIndex = basic::make_vec3<int>(randomPoint / __in_attnMapImage3dSpan.geometry.voxelSize);
    auto index = __in_attnMapImage3dSpan.geometry.at(voxelIndex);
    basic::CrystalGeometry geo;
    geo.position = randomPoint;
    if (__in_attnMapImage3dSpan.ptr[index] >= scatterPointThreshold) {
      result.push_back(
          {geo, __in_attnMapImage3dSpan.ptr[index] * 10, volume}); // float mu = __in_attnMapImage3dSpan.ptr[index]*10?
    }
  });
  return result;
}

__PNI_CUDA_MACRO__ inline double calScannerEFFWithScatterEnergy(
    float energy, basic::Vec3<double> scatterEnergyWindow) {
  double sigmaTimesSqrt2 = scatterEnergyWindow.z * 511 / 2.35482 * 1.41421;
  return basic::FMath<double>::gauss_integral(scatterEnergyWindow.x, scatterEnergyWindow.y, energy, sigmaTimesSqrt2);
}

std::vector<float> inline generateScatterEffTable(
    basic::Vec3<double> scannerEffTableEnergy, basic::Vec3<double> scatterEnergyWindow) {
  int energyBinNum = int((scannerEffTableEnergy.y - scannerEffTableEnergy.x) / scannerEffTableEnergy.z) + 1;
  std::vector<float> scannerEffTable(energyBinNum);
  for_each(
      energyBinNum,
      [&](int binIdx) {
        double energyNow = scannerEffTableEnergy.x + binIdx * scannerEffTableEnergy.z;
        scannerEffTable[binIdx] = calScannerEFFWithScatterEnergy(energyNow, scatterEnergyWindow);
      },
      basic::CpuMultiThread::callWithAllThreads());
  return scannerEffTable;
}

template <typename ImageValueType, typename PairDataView, typename EMCalculationMethod>
inline void generateSSS_AttnCoff(
    Image3DInputSpan<ImageValueType> __in_attnMapImage3dSpan, ViewForPointsCartesianProduct<PairDataView> __viewAttn,
    ImageValueType *__ptrAttnEMSum, EMCalculationMethod __emMethod, basic::CpuMultiThread __cpuThread) {
  process::EMSum(__in_attnMapImage3dSpan, __in_attnMapImage3dSpan.geometry.roi(), __ptrAttnEMSum, __viewAttn,
                 __emMethod, __cpuThread);
}

template <typename ImageValueType, typename PairDataView, typename EMCalculationMethod>
inline void generateSSS_Emission(
    Image3DInputSpan<ImageValueType> __in_osemImage3dSpan, ViewForPointsCartesianProduct<PairDataView> __viewAttn,
    ImageValueType *__ptrEmissionEMSum, EMCalculationMethod __emMethod, basic::CpuMultiThread __cpuThread) {
  process::EMSum(__in_osemImage3dSpan, __in_osemImage3dSpan.geometry.roi(), __ptrEmissionEMSum, __viewAttn, __emMethod,
                 __cpuThread);
}
inline std::vector<double> generateGaussianBlurKernel(
    double __systemTimeRes_ns, double __TOFBinWidth) {
  double sigma = __systemTimeRes_ns / 2.355; // unit: ns
  int validTofBinNumHalf = int(3.0 * sigma / __TOFBinWidth);
  std::vector<double> gauss(2 * validTofBinNumHalf + 1, 0);
  for (auto idx : std::views::iota(0, validTofBinNumHalf)) {
    int idp = validTofBinNumHalf + idx;
    int idn = validTofBinNumHalf - idx;
    gauss[idp] = exp(-(idx * __TOFBinWidth) * (idx * __TOFBinWidth) / (2 * sigma * sigma));
    gauss[idn] = exp(-(idx * __TOFBinWidth) * (idx * __TOFBinWidth) / (2 * sigma * sigma));
  }
  return gauss;
}

template <typename ProjectionMethod>
struct _calTOFActivityIntergral {
  Image3DInputSpan<float> __in_emap_3d;
  p3df m_cry1Position;
  p3df m_cry2Position;
  p3df m_scatterPosition;
  ProjectionMethod m_projector;
  float m_bin_size_mm;

  __PNI_CUDA_MACRO__ basic::Event<float> initialEvent(
      basic::CrystalGeometry *pos1, basic::CrystalGeometry *pos2) {
    basic::Event<float> tempEvent;
    tempEvent.crystal1.geometry = pos1;
    tempEvent.crystal2.geometry = pos2;
    tempEvent.crystal1.tof_deviation = nullptr;
    tempEvent.crystal2.tof_deviation = nullptr;
    tempEvent.crystal1.tof_mean = nullptr;
    tempEvent.crystal2.tof_mean = nullptr;
    tempEvent.eventIndex = -1;
    tempEvent.gain = nullptr;
    tempEvent.time1_2 = nullptr;
    return tempEvent;
  }

  __PNI_CUDA_MACRO__ double intergralBySampling(
      const basic::Event<float> &event) {
    auto &sampler = m_projector.sampler;
    auto roi_geometry = __in_emap_3d.geometry.roi();
    if (!sampler.setInfo(event, &(__in_emap_3d.geometry), &roi_geometry))
      return 0;
    sampler.reset();
    double sum = 0;
    while (!sampler.isEnd()) {
      const auto samplePoint = sampler.next();
      const auto stepSize = sampler.getlastStepSize();
      if (stepSize == 0)
        continue; // 跳过无效采样点
      sum += m_projector.interpolator(samplePoint, __in_emap_3d.ptr, __in_emap_3d.geometry) * stepSize;
    }
    return sum;
  }

  __PNI_CUDA_MACRO__ basic::Vec2<float> operator()(
      std::size_t tofBinIdx) {
    auto cry1_cry2 = m_cry2Position - m_cry1Position;
    auto cry1_cry2_len = cry1_cry2.l2();
    double line_binstart = tofBinIdx * m_bin_size_mm; // bin goes from cry1 to cry2
    double line_binend = basic::FMath<float>::min((tofBinIdx + 1) * m_bin_size_mm, cry1_cry2_len);
    // when P is binStart
    float distanceP1A = line_binstart;
    float distanceP1B = cry1_cry2_len - line_binstart;
    float distanceP1A_P1B = distanceP1A - distanceP1B;
    auto polylineStart = basic::polylinePosition(m_cry1Position, m_scatterPosition, m_cry2Position, distanceP1A_P1B);
    // when P is binEnd
    float distanceP2A = line_binend;
    float distanceP2B = cry1_cry2_len - line_binend;
    float distanceP2A_P2B = distanceP2A - distanceP2B;
    auto polylineEnd = basic::polylinePosition(m_cry1Position, m_scatterPosition, m_cry2Position, distanceP2A_P2B);
    // SA_SB
    float distanceSA_SB = (m_scatterPosition - m_cry1Position).l2() - (m_scatterPosition - m_cry2Position).l2();
    auto deltaLen1 = distanceP1A_P1B - distanceSA_SB;
    auto deltaLen2 = distanceP2A_P2B - distanceSA_SB;
    float TOFIntergral1;
    float TOFIntergral2;

    if (deltaLen1 < 0 && deltaLen2 < 0) // case1 binstart binend all on sa
    {
      TOFIntergral2 = 0;
      basic::CrystalGeometry geo1, geo2;
      geo1.position = polylineStart;
      geo2.position = polylineEnd;
      auto event = initialEvent(&geo1, &geo2);
      TOFIntergral1 = intergralBySampling(event);

    } else if (deltaLen1 > 0 && deltaLen2 > 0) // case2 binstart binend all on sb
    {
      TOFIntergral1 = 0;
      basic::CrystalGeometry geo1, geo2;
      geo1.position = polylineStart;
      geo2.position = polylineEnd;
      auto event = initialEvent(&geo1, &geo2);
      TOFIntergral2 = intergralBySampling(event);
    } else if (deltaLen1 * deltaLen2 < 0) {
      basic::CrystalGeometry geo1, geoS, geo2;
      geo1.position = polylineStart;
      geoS.position = m_scatterPosition;
      geo2.position = polylineEnd;
      auto eventA = initialEvent(&geo1, &geoS);
      TOFIntergral1 = intergralBySampling(eventA);
      auto eventB = initialEvent(&geoS, &geo2);
      TOFIntergral2 = intergralBySampling(eventB);
    }
    return basic::make_vec2<float>(TOFIntergral1, TOFIntergral2);
  }
};

template <typename ProjectionMethod>
struct _singleScatterSimulationTOF {

  float *m_out_tofBinSSS; // size is dsLORNum * TOFBinNum
  float *m_scatOneLOR;
  float *m_scatOneLORBlur;
  float *m_in_sssAttnCoff;
  double *m_guassBlur;
  basic::CrystalGeometry *m_dsCrystalGeometry;
  ScatterPoint *m_scatterPoints;
  float *m_scannerEffTable;

  example::PolygonalSystem __dsPolygon;
  basic::DetectorGeometry __dsDetectorGeometry; // 探测器几何信息

  ProjectionMethod m_projector;
  Image3DInputSpan<float> m_in_emap_3d;
  basic::Vec3<double> m_scannerEffTableEnergy;

  std::size_t m_scatterPointNum;
  float m_crystalArea;
  float m_tofBinWidth; // mm
  double m_commonfactor;
  int m_tofBinNum;
  int m_gaussSize;

  __PNI_CUDA_MACRO__ void operator()(
      std::size_t dsLORidx) {
    auto [block1IndexR, block2IndexR] =
        example::polygon::calRectangleFlatCrystalIDFromLORID(__dsPolygon, __dsDetectorGeometry, dsLORidx);
    auto block1IndexUni =
        example::polygon::getUniformIDFromRectangleID(__dsPolygon, __dsDetectorGeometry, block1IndexR);
    auto block2IndexUni =
        example::polygon::getUniformIDFromRectangleID(__dsPolygon, __dsDetectorGeometry, block2IndexR);

    for (int sctPointIdx = 0; sctPointIdx < m_scatterPointNum; sctPointIdx++) {
      int index1 = block1IndexUni * m_scatterPointNum + sctPointIdx;
      int index2 = block2IndexUni * m_scatterPointNum + sctPointIdx;
      // cal vector s -> crystal
      auto s_cry1 = m_dsCrystalGeometry[block1IndexUni].position -
                    m_scatterPoints[sctPointIdx].sssPosition.position; // vector s->cry1
      auto s_cry2 = m_dsCrystalGeometry[block2IndexUni].position -
                    m_scatterPoints[sctPointIdx].sssPosition.position; // vector s->cry2
      double distance_cry1_S = s_cry1.l2() * 0.1;                      // cm
      double distance_cry2_S = s_cry2.l2() * 0.1;                      // cm
      // cal cosTheta
      double scatCosTheta = -basic::cosine(s_cry1, s_cry2);
      // cal diffCross
      double diffCross = calDiffCrossSection(scatCosTheta);
      // cal sctEnergy
      double scatterEnergy = 511 / (2 - scatCosTheta); // the energy of scatteted photon
      if (scatterEnergy < m_scannerEffTableEnergy.x || scatterEnergy > m_scannerEffTableEnergy.y)
        return;
      // get scannerEff from  table
      int EffTableIndex = int((scatterEnergy - m_scannerEffTableEnergy.x) / m_scannerEffTableEnergy.z);
      float scannerEffNow = m_scannerEffTable[EffTableIndex];
      // cal totalAttn
      double Icry1 =
          m_in_sssAttnCoff[index1] * scatter::calTotalAttenInScatterEnergy(m_in_sssAttnCoff[index2], scatterEnergy);
      double Icry2 =
          m_in_sssAttnCoff[index2] * scatter::calTotalAttenInScatterEnergy(m_in_sssAttnCoff[index1], scatterEnergy);
      // cal projectArea
      auto n1_vector = m_dsCrystalGeometry[block1IndexUni].directionU.cross(
          m_dsCrystalGeometry[block1IndexUni].directionV); // 法向量
      auto n2_vector = m_dsCrystalGeometry[block2IndexUni].directionU.cross(
          m_dsCrystalGeometry[block2IndexUni].directionV); // 法向量
      double projectArea1 = basic::calculateProjectionArea(
          m_crystalArea, n1_vector, s_cry1); // notice:though we use block as crystal when downSampling ,but the
                                             // crystalArea geometry is still crystal geometry
      double projectArea2 = basic::calculateProjectionArea(m_crystalArea, n2_vector, s_cry2);
      double scatOnePoint_part = m_scatterPoints[sctPointIdx].mu * projectArea1 * projectArea2 * diffCross *
                                 scannerEffNow /
                                 (distance_cry1_S * distance_cry1_S * distance_cry2_S * distance_cry2_S);
      // tof intergral by scat-lor
      for (int tofbinIdx = 0; tofbinIdx < m_tofBinNum; tofbinIdx++) {
        auto [tofIntergral1, tofIntergral2] =
            _calTOFActivityIntergral<ProjectionMethod>{m_in_emap_3d,
                                                       m_dsCrystalGeometry[block1IndexUni].position,
                                                       m_dsCrystalGeometry[block2IndexUni].position,
                                                       m_scatterPoints[sctPointIdx].sssPosition.position,
                                                       m_projector,
                                                       m_tofBinWidth}(tofbinIdx);
        double Ia = tofIntergral1 * Icry1;
        double Ib = tofIntergral2 * Icry2;
        m_scatOneLOR[tofbinIdx] += scatOnePoint_part * (Ia + Ib);
      }
    }
    // gauss blur
    for (int TOFBinIdx = 0; TOFBinIdx < m_tofBinNum; TOFBinIdx++) {
      int startCov = openpni::basic::FMath<int>::max(0, TOFBinIdx - int(m_gaussSize * 0.5));
      int endCov = openpni::basic::FMath<int>::min(m_tofBinNum - 1, TOFBinIdx + int(m_gaussSize * 0.5));
      for (int cov = startCov; cov <= endCov; cov++)
        m_scatOneLORBlur[TOFBinIdx] += m_scatOneLOR[cov] * m_guassBlur[cov - TOFBinIdx + int(m_gaussSize * 0.5)];
    }
    // add to down sampled TOF mich
    for (int tofbinIdx = 0; tofbinIdx < m_tofBinNum; tofbinIdx++) {
      m_out_tofBinSSS[dsLORidx * m_tofBinNum + tofbinIdx] = m_scatOneLORBlur[tofbinIdx] * m_commonfactor;
    }
  }
};

struct _sumBin {
  float *__out_scatBinSumMich;
  float *__in_scatDsMichWithTofBin;
  std::size_t __donwSamplingLORNum;
  int __timeBinNum;
  __PNI_CUDA_MACRO__ void operator()(
      std::size_t idx) {
    std::size_t subLORIndex = idx / __timeBinNum;
    __out_scatBinSumMich[subLORIndex] += __in_scatDsMichWithTofBin[idx];
  }
};

struct _upSamplingByInterpolation2D {

  struct _upSamplingBySlice {
    float *__out_scatDSMichFullSlice;
    float *__in_scatBinSumMich;

    // size_t __binNum;
    // size_t __viewNum;
    // size_t __sliceNum;
    // size_t __dsBinNum;
    // size_t __dsViewNum;

    example::PolygonalSystem __polygon;
    basic::DetectorGeometry __detectorGeometry;
    example::PolygonalSystem __dsPolygon;
    basic::DetectorGeometry __dsDetectorGeometry;

    __PNI_CUDA_MACRO__ void operator()(
        std::size_t idx) {
      auto __dsBinNum = example::polygon::getBinNum(__dsPolygon, __dsDetectorGeometry);
      auto __dsViewNum = example::polygon::getViewNum(__dsPolygon, __dsDetectorGeometry);
      auto sliceIdx = idx / (__dsBinNum * __dsViewNum);
      auto dsbiviIdx = idx % (__dsBinNum * __dsViewNum);
      auto [ring1, ring2] = example::polygon::calRing1Ring2FromSlice(__polygon, __detectorGeometry, sliceIdx);
      int dsRing1 = ring1 * example::polygon::getRingNum(__dsPolygon, __dsDetectorGeometry) /
                    example::polygon::getRingNum(__polygon, __detectorGeometry);
      int dsRing2 = ring2 * example::polygon::getRingNum(__dsPolygon, __dsDetectorGeometry) /
                    example::polygon::getRingNum(__polygon, __detectorGeometry);
      int dsSliceIdx = dsRing1 + example::polygon::getRingNum(__dsPolygon, __dsDetectorGeometry) * dsRing2;

      __out_scatDSMichFullSlice[idx] = __in_scatBinSumMich[dsbiviIdx + dsSliceIdx * (__dsBinNum * __dsViewNum)];
    }
  };

  struct _2DIngerpolationUpSampling {
    float *__out_scatDsMich;
    float *__in_scatDSMichFullSlice;
    example::PolygonalSystem __polygon;
    basic::DetectorGeometry __detectorGeometry;
    example::PolygonalSystem __dsPolygon;
    basic::DetectorGeometry __dsDetectorGeometry;

    basic::CrystalGeometry *in_crystalGeometry;
    basic::CrystalGeometry *in_dsCrystalGeometry;

    __PNI_CUDA_MACRO__ void chooseNearestBlockAndCalWeight(
        int &blockNearest, float &w, int block, int cryID) {
      int block_left = block - 1;
      int block_right = block + 1;
      auto cry_block_left = in_crystalGeometry[cryID].position - in_dsCrystalGeometry[block_left].position;
      auto cry_block_right = in_crystalGeometry[cryID].position - in_dsCrystalGeometry[block_right].position;
      auto distance_cry_block_left = cry_block_left.l2();
      auto distance_cry_block_right = cry_block_right.l2();
      if (distance_cry_block_left > distance_cry_block_right) {
        blockNearest = block_right;
        w = distance_cry_block_right;
      } else {
        blockNearest = block_left;
        w = distance_cry_block_left;
      }
    }

    __PNI_CUDA_MACRO__ float getInterpolationValue(
        std::size_t lorIdx) {
      auto [cryA, cryB] = example::polygon::calRectangleFlatCrystalIDFromLORID(__polygon, __detectorGeometry, lorIdx);
      // cal where cry1 cry2 in block,this is also the index in ds image
      int blockA = cryA / (__detectorGeometry.crystalNumU * __detectorGeometry.crystalNumV);
      auto cry1_blockA = in_crystalGeometry[cryA].position - in_dsCrystalGeometry[blockA].position;
      float wA = cry1_blockA.l2();
      int blockB = cryB / (__detectorGeometry.crystalNumU * __detectorGeometry.crystalNumV);
      auto cry2_blockB = in_crystalGeometry[cryB].position - in_dsCrystalGeometry[blockB].position;
      float wB = cry2_blockB.l2();
      // choose the nearest block and cal the weight
      float wA_, wB_;
      int blockA_, blockB_;
      chooseNearestBlockAndCalWeight(blockA_, wA_, blockA, cryA);
      chooseNearestBlockAndCalWeight(blockB_, wB_, blockB, cryB);
      // cal  A-B ,A-B_ ,A_-B ,A_-B_'s dsLORID
      std::size_t dsLORIDAB =
          example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA, blockB);
      std::size_t dsLORIDAB_ =
          example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA, blockB_);
      std::size_t dsLORIDA_B =
          example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA_, blockB);
      std::size_t dsLORIDA_B_ =
          example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA_, blockB_);
      // because slice has been upSampled
      auto biviNum = example::polygon::getBinNum(__dsPolygon, __dsDetectorGeometry) *
                     example::polygon::getViewNum(__dsPolygon, __dsDetectorGeometry);
      auto sliceNow = lorIdx / (example::polygon::getBinNum(__polygon, __detectorGeometry) *
                                example::polygon::getViewNum(__polygon, __detectorGeometry));
      dsLORIDAB = dsLORIDAB % biviNum + sliceNow * biviNum;
      dsLORIDAB_ = dsLORIDAB_ % biviNum + sliceNow * biviNum;
      dsLORIDA_B = dsLORIDA_B % biviNum + sliceNow * biviNum;
      dsLORIDA_B_ = dsLORIDA_B_ % biviNum + sliceNow * biviNum;
      // bilinear interpolation
      float value = __in_scatDSMichFullSlice[dsLORIDAB] * wA * wB + __in_scatDSMichFullSlice[dsLORIDAB_] * wA * wB_ +
                    __in_scatDSMichFullSlice[dsLORIDA_B] * wA_ * wB + __in_scatDSMichFullSlice[dsLORIDA_B_] * wA_ * wB_;
      float w_All = wA * wB + wA * wB_ + wA_ * wB + wA_ * wB_;

      return value / w_All;
    }

    __PNI_CUDA_MACRO__ void operator()(
        std::size_t lorIdx) {
      auto [cryA, cryB] = example::polygon::calRectangleFlatCrystalIDFromLORID(__polygon, __detectorGeometry, lorIdx);
      // cal where cry1 cry2 in block,this is also the index in ds image
      int blockA = cryA / (__detectorGeometry.crystalNumU * __detectorGeometry.crystalNumV);
      auto cry1_blockA = in_crystalGeometry[cryA].position - in_dsCrystalGeometry[blockA].position;
      float wA = cry1_blockA.l2();
      int blockB = cryB / (__detectorGeometry.crystalNumU * __detectorGeometry.crystalNumV);
      auto cry2_blockB = in_crystalGeometry[cryB].position - in_dsCrystalGeometry[blockB].position;
      float wB = cry2_blockB.l2();
      // choose the nearest block and cal the weight
      float wA_, wB_;
      int blockA_, blockB_;
      chooseNearestBlockAndCalWeight(blockA_, wA_, blockA, cryA);
      chooseNearestBlockAndCalWeight(blockB_, wB_, blockB, cryB);
      // cal  A-B ,A-B_ ,A_-B ,A_-B_'s dsLORID
      std::size_t dsLORIDAB =
          example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA, blockB);
      std::size_t dsLORIDAB_ =
          example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA, blockB_);
      std::size_t dsLORIDA_B =
          example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA_, blockB);
      std::size_t dsLORIDA_B_ =
          example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA_, blockB_);
      // because slice has been upSampled
      auto biviNum = example::polygon::getBinNum(__dsPolygon, __dsDetectorGeometry) *
                     example::polygon::getViewNum(__dsPolygon, __dsDetectorGeometry);
      auto sliceNow = lorIdx / (example::polygon::getBinNum(__polygon, __detectorGeometry) *
                                example::polygon::getViewNum(__polygon, __detectorGeometry));
      dsLORIDAB = dsLORIDAB % biviNum + sliceNow * biviNum;
      dsLORIDAB_ = dsLORIDAB_ % biviNum + sliceNow * biviNum;
      dsLORIDA_B = dsLORIDA_B % biviNum + sliceNow * biviNum;
      dsLORIDA_B_ = dsLORIDA_B_ % biviNum + sliceNow * biviNum;
      // bilinear interpolation
      float value = __in_scatDSMichFullSlice[dsLORIDAB] * wA * wB + __in_scatDSMichFullSlice[dsLORIDAB_] * wA * wB_ +
                    __in_scatDSMichFullSlice[dsLORIDA_B] * wA_ * wB + __in_scatDSMichFullSlice[dsLORIDA_B_] * wA_ * wB_;
      float w_All = wA * wB + wA * wB_ + wA_ * wB + wA_ * wB_;
      __out_scatDsMich[lorIdx] = value / w_All;
    }
  };
};

template <process::IndexerContract _Indexer, typename _FactorValue = float>
struct _SSSTOFAdaptor {
  _FactorValue *sssValue; // sssValueSize = dsViewNum * dsBinNum * sliceNum * tofBinNum
  _Indexer indexer;
  double tofBinWidth;
  int tofBinNum;

  example::PolygonalSystem __polygon;
  basic::DetectorGeometry __detectorGeometry;
  example::PolygonalSystem __dsPolygon;
  basic::DetectorGeometry __dsDetectorGeometry;

  basic::CrystalGeometry *in_crystalGeometry;
  basic::CrystalGeometry *in_dsCrystalGeometry;

  __PNI_CUDA_MACRO__ _FactorValue operator()(
      const basic::Event<_FactorValue> listmodeTOFevent) const {
    if (!sssValue)
      return 0;
    // basic info
    auto fullSliceDsLorNum = example::polygon::getBinNum(__dsPolygon, __dsDetectorGeometry) *
                             example::polygon::getViewNum(__dsPolygon, __dsDetectorGeometry) *
                             example::polygon::getSliceNum(__polygon, __detectorGeometry);

    // cal tofBinIndex
    int tofBinIndex = int(floor(*listmodeTOFevent.time1_2 / tofBinWidth + 0.5)) + int(tofBinNum / 2);
    // cal sssvalue by tofBinIndex & lorIndex
    _upSamplingByInterpolation2D::_2DIngerpolationUpSampling upsampler{.__out_scatDsMich = nullptr, // not used
                                                                       .__in_scatDSMichFullSlice =
                                                                           sssValue + tofBinIndex * fullSliceDsLorNum,
                                                                       .__polygon = __polygon,
                                                                       .__detectorGeometry = __detectorGeometry,
                                                                       .__dsPolygon = __dsPolygon,
                                                                       .__dsDetectorGeometry = __dsDetectorGeometry,
                                                                       .in_crystalGeometry = in_crystalGeometry,
                                                                       .in_dsCrystalGeometry = in_dsCrystalGeometry};
    auto lorIndex = indexer.indexInMich(listmodeTOFevent.crystal1.crystalIndex, listmodeTOFevent.crystal2.crystalIndex);
    return upsampler.getInterpolationValue(lorIndex);
  }
};

template <typename ImageValueType>
struct _singleScatterSimulation {
  ImageValueType *__out_d_scatterValue;
  ImageValueType *__sssEmission;
  ImageValueType *__sssAttnCoff;
  basic::CrystalGeometry *__d_crystalGeometry;
  ScatterPoint *__scatterPoints;
  float *__scannerEffTable;

  basic::Vec3<double> __scannerEffTableEnergy;

  int __countScatter;
  float crystalArea;
  double __commonfactor;
  example::PolygonalSystem polygon;
  basic::DetectorGeometry detectorGeometry; // 探测器几何信息
  __device__ void operator()(
      size_t __lorIndex) const {
    auto [cry1IndexR, cry2IndexR] =
        example::polygon::calRectangleFlatCrystalIDFromLORID(polygon, detectorGeometry, __lorIndex);
    auto cry1IndexU = example::polygon::getUniformIDFromRectangleID(polygon, detectorGeometry, cry1IndexR);
    auto cry2IndexU = example::polygon::getUniformIDFromRectangleID(polygon, detectorGeometry, cry2IndexR);
    auto binNum = example::polygon::getBinNum(polygon, detectorGeometry);
    auto binNumOutFOVOneSide = example::polygon::calBinNumOutFOVOneSide(polygon, detectorGeometry, 4);
    int binIndex = __lorIndex % binNum;
    if (binIndex < binNumOutFOVOneSide || binIndex >= binNum - binNumOutFOVOneSide)
      return;
    for (int scatterIndex = 0; scatterIndex < __countScatter; scatterIndex++) {
      // cal index
      int index1 = cry1IndexU * __countScatter + scatterIndex;
      int index2 = cry2IndexU * __countScatter + scatterIndex;

      // cal vector s -> crystal
      auto s_cry1 = (__d_crystalGeometry[cry1IndexU].position) -
                    __scatterPoints[scatterIndex].sssPosition.position; // vector s->cry1
      auto s_cry2 = (__d_crystalGeometry[cry2IndexU].position) -
                    __scatterPoints[scatterIndex].sssPosition.position; // vector s->cry2
      double distance_cry1_S = s_cry1.l2() * 0.1;                       // cm
      double distance_cry2_S = s_cry2.l2() * 0.1;                       // cm
      // cac scatCosTheta
      double scatCosTheta = -basic::cosine(s_cry1, s_cry2); // notice the definition of cosTheta is different with
      //  cal difCross
      double diffCross = calDiffCrossSection(scatCosTheta);
      // cal sctEnergy
      double scatterEnergy = 511 / (2 - scatCosTheta); // the energy of scatteted photon
      if (scatterEnergy < __scannerEffTableEnergy.x || scatterEnergy > __scannerEffTableEnergy.y)
        return;
      // get scannerEff from  table
      int EffTableIndex = int((scatterEnergy - __scannerEffTableEnergy.x) / __scannerEffTableEnergy.z);
      float scannerEffNow = __scannerEffTable[EffTableIndex];
      // cal projectArea
      auto n1_vector =
          __d_crystalGeometry[cry1IndexU].directionU.cross(__d_crystalGeometry[cry1IndexU].directionV); // 法向量
      auto n2_vector =
          __d_crystalGeometry[cry2IndexU].directionU.cross(__d_crystalGeometry[cry2IndexU].directionV); // 法向量
      double projectArea1 = basic::calculateProjectionArea(crystalArea, n1_vector, s_cry1);
      double projectArea2 = basic::calculateProjectionArea(crystalArea, n2_vector, s_cry2);

      // cal Icry1 Icry2
      double Icry1 = __sssAttnCoff[index1] * __sssEmission[index1] *
                     calTotalAttenInScatterEnergy(__sssAttnCoff[index2], scatterEnergy);
      double Icry2 = __sssAttnCoff[index2] * __sssEmission[index2] *
                     calTotalAttenInScatterEnergy(__sssAttnCoff[index1], scatterEnergy);
      // cal sctValue
      // atomicAdd(__out_d_scatterValue + lorIndex,
      //           __commonfactor * __scatterPoints[scatterIndex].mu * projectArea1 * projectArea2 * diffCross *
      //               (Icry1 + Icry2) * scannerEffNow /
      //               (distance_cry1_S * distance_cry1_S * distance_cry2_S * distance_cry2_S));
      // atomicAdd(__out_d_scatterValue + __in_petDataView.indexer.indexInMich(cry1Index, cry2Index),
      //           __commonfactor * __scatterPoints[scatterIndex].mu * projectArea1 * projectArea2 * diffCross *
      //               (Icry1 + Icry2) * scannerEffNow /
      //               (distance_cry1_S * distance_cry1_S * distance_cry2_S * distance_cry2_S));
      // __out_d_scatterValue[lorIndex] = 1;
      atomicAdd(__out_d_scatterValue + __lorIndex,
                __commonfactor * __scatterPoints[scatterIndex].mu * diffCross * projectArea1 * projectArea2 * 100 *
                    (Icry1 + Icry2) * scannerEffNow /
                    (distance_cry1_S * distance_cry1_S * distance_cry2_S * distance_cry2_S));
    }
  }
};

template <typename ImageValueType, typename LinearFitModel>
struct _sssTailFitting {
  ImageValueType *__out_d_scatterValue;
  ImageValueType *__promptMich; // must gived to do tail fitting
  ImageValueType *__normMich;   // must have default : all 1，only use norm mich with: blockProfA,
                                // blockProfT,crystalFct，它们是生成归一化mich是伴随生成的部分组件
  ImageValueType *__randMich;
  ImageValueType *__attnCutBedCoff;
  uint32_t __binNum;
  uint32_t __LORNumOneSlice;
  uint32_t __binNumOutFOVOneSide;

  double scatterTailFittingThreshold;
  __device__ void operator()(
      std::size_t sl) const {
    LinearFitModel __linearFitModel;

    for (int lorInSl = 0; lorInSl < __LORNumOneSlice; lorInSl++) {
      int binIndex = lorInSl % __binNum;
      size_t lorIndex = sl * __LORNumOneSlice + lorInSl;

      if (__attnCutBedCoff[lorIndex] >= scatterTailFittingThreshold && binIndex >= __binNumOutFOVOneSide &&
          binIndex < __binNum - __binNumOutFOVOneSide) {
        if (__normMich[lorIndex] == 0)
          continue;

        __linearFitModel.add(__out_d_scatterValue[lorIndex], __promptMich[lorIndex] - __randMich[lorIndex]);
      }
    }

    for (int lorInSl = 0; lorInSl < __LORNumOneSlice; lorInSl++) {
      int binIndex = lorInSl % __binNum;
      size_t lorIndex = sl * __LORNumOneSlice + lorInSl;

      if (binIndex >= __binNumOutFOVOneSide && binIndex < __binNum - __binNumOutFOVOneSide) {
        if (__normMich[lorIndex] == 0) {
          __out_d_scatterValue[lorIndex] = 0; // 如果归一化mich为0，则scatter值也置为0
          continue;
        }
        __out_d_scatterValue[lorIndex] = __linearFitModel.predict(__out_d_scatterValue[lorIndex]);
        // __out_d_scatterValue[lorIndex] = 2;

      } else
        __out_d_scatterValue[lorIndex] = 0; // 对于FOV外的bin，scatter值不进行tail fitting，直接置为0
    };
  };
};

template <typename ImageValueType, typename LinearFitModel>
struct _sssTailFittingTOF {
  ImageValueType *__out_d_scatDSTOFMichFullSlice;
  ImageValueType *__in_d_predictScatterValue;
  ImageValueType *__in_d_scatDSTOFMich;
  ImageValueType *__promptMich; // must gived to do tail fitting
  ImageValueType *__normMich;   // must have default : all 1，only use norm mich with: blockProfA,
                                // blockProfT,crystalFct，它们是生成归一化mich是伴随生成的部分组件
  ImageValueType *__randMich;
  ImageValueType *__attnCutBedCoff;

  example::PolygonalSystem __polygon;
  basic::DetectorGeometry __detectorGeometry;
  example::PolygonalSystem __dsPolygon;
  basic::DetectorGeometry __dsDetectorGeometry;
  int __minSectorDifference;
  int __timeBinNum;
  double scatterTailFittingThreshold;

  __PNI_CUDA_MACRO__ void operator()(
      std::size_t sl) const {
    auto __binNum = example::polygon::getBinNum(__polygon, __detectorGeometry);
    auto __viewNum = example::polygon::getViewNum(__polygon, __detectorGeometry);
    auto __sliceNum = example::polygon::getSliceNum(__polygon, __detectorGeometry);
    auto __LORNumOneSlice = __binNum * __viewNum;
    auto __binNumOutFOVOneSide =
        example::polygon::calBinNumOutFOVOneSide(__polygon, __detectorGeometry, __minSectorDifference);
    auto __dsBinNum = example::polygon::getBinNum(__dsPolygon, __dsDetectorGeometry);
    auto __dsViewNum = example::polygon::getViewNum(__dsPolygon, __dsDetectorGeometry);
    auto __dsSliceNum = example::polygon::getSliceNum(__dsPolygon, __dsDetectorGeometry);
    auto [__ring1, __ring2] = example::polygon::calRing1Ring2FromSlice(__polygon, __detectorGeometry, sl);
    int __dsRing1 = __ring1 * example::polygon::getRingNum(__dsPolygon, __dsDetectorGeometry) /
                    example::polygon::getRingNum(__polygon, __detectorGeometry);
    int __dsRing2 = __ring2 * example::polygon::getRingNum(__dsPolygon, __dsDetectorGeometry) /
                    example::polygon::getRingNum(__polygon, __detectorGeometry);
    int __dsSliceIdx = __dsRing1 + example::polygon::getRingNum(__dsPolygon, __dsDetectorGeometry) * __dsRing2;
    //=====do
    LinearFitModel __linearFitModel;
    for (int lorInSl = 0; lorInSl < __LORNumOneSlice; lorInSl++) {
      int binIndex = lorInSl % __binNum;
      size_t lorIndex = sl * __LORNumOneSlice + lorInSl;

      if (binIndex >= __binNumOutFOVOneSide && binIndex < __binNum - __binNumOutFOVOneSide) {
        if (__normMich[lorIndex] == 0) {
          continue;
        }
        if (__attnCutBedCoff[lorIndex] >= scatterTailFittingThreshold)
          continue;

        __linearFitModel.add(__in_d_predictScatterValue[lorIndex], __promptMich[lorIndex] - __randMich[lorIndex]);
      }
    }
    for (size_t bivi = 0; bivi < __dsBinNum * __dsViewNum; bivi++)
      for (int tofbinIdx = 0; tofbinIdx < __timeBinNum; tofbinIdx++)
        __out_d_scatDSTOFMichFullSlice[bivi + sl * __dsBinNum * __dsViewNum +
                                       tofbinIdx * __dsBinNum * __dsViewNum * __sliceNum] =
            __linearFitModel.predict(
                __in_d_scatDSTOFMich[(bivi + __dsSliceIdx * __dsBinNum * __dsViewNum) * __timeBinNum + tofbinIdx]);
  }
};

} // namespace scatter
} // namespace openpni::process