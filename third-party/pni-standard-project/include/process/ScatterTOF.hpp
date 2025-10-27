// #pragma once
// #include "../basic/Math.hpp"
// #include "../example/PolygonalSystem.hpp"
// #include "Scatter.hpp"

// namespace openpni::process {

// namespace sss_tof {

// constexpr double PI = 3.1415926535898;

// double calTotalComptonCrossSection(
//     double energy) {
//   constexpr double Re = 2.818E-13; // cm
//   double k = energy / 511.f;
//   double a = log(1 + 2 * k);
//   double prefactor = 2 * PI * Re * Re;
//   return prefactor * ((1.0 + k) / (k * k) * (2.0 * (1.0 + k) / (1.0 + 2.0 * k) - a / k) + a / (2.0 * k) -
//                       (1.0 + 3.0 * k) / (1.0 + 2.0 * k) / (1.0 + 2.0 * k));
// }

// std::vector<double> generateGaussianBlurKernel(
//     double __systemTimeRes_ns, double __TOFBinWidth) {
//   double sigma = __systemTimeRes_ns / 2.355; // unit: ns
//   int validTofBinNumHalf = int(3.0 * sigma / __TOFBinWidth);
//   std::vector<double> gauss(2 * validTofBinNumHalf + 1, 0);
//   for (auto idx : std::views::iota(0, validTofBinNumHalf)) {
//     int idp = validTofBinNumHalf + idx;
//     int idn = validTofBinNumHalf - idx;
//     gauss[idp] = exp(-(idx * __TOFBinWidth) * (idx * __TOFBinWidth) / (2 * sigma * sigma));
//     gauss[idn] = exp(-(idx * __TOFBinWidth) * (idx * __TOFBinWidth) / (2 * sigma * sigma));
//   }
//   return gauss;
// }

// template <typename SamplerMethod, typename calculationPrecision>
// struct _calTOFActivityIntergral {

//   p3df m_cry1Position;
//   p3df m_cry2Position;
//   p3df m_scatterPosition;
//   SamplerMethod m_sampler;
//   float m_bin_size_mm;

//   basic::Event<float> initialEvent(
//       p3df pos1, p3df pos2) {
//     basic::Event<float> tempEvent;
//     tempEvent.crystal1.geometry = &pos1;
//     tempEvent.crystal2.geometry = &pos2;
//     tempEvent.crystal1.tof_deviation = nullptr;
//     tempEvent.crystal2.tof_deviation = nullptr;
//     tempEvent.crystal1.tof_mean = nullptr;
//     tempEvent.crystal2.tof_mean = nullptr;
//     tempEvent.eventIndex = nullptr;
//     tempEvent.gain = nullptr;
//     tempEvent.time1_2 = nullptr;
//     return tempEvent;
//   }

//   double intergralBySampling(
//       const basic::Event<float> &evnet) {
//     if (!m_sampler.setInfo(tempEvent, __image3dSize, __roi))
//       return 0;
//     m_sampler.reset();
//     double sum = T(0);
//     while (!m_sampler.isEnd()) {
//       const auto samplePoint = m_sampler.next();
//       const auto stepSize = m_sampler.getlastStepSize();
//       if (stepSize == 0)
//         continue; // 跳过无效采样点
//       sum += __projectionMethod.interpolator(samplePoint, __in_img_3d, *__image3dSize) * stepSize;
//     }
//     return sum;
//   }

//   basic::Vec2<calculationPrecision> operator()(
//       std::size_t tofBinIdx) {
//     auto cry1_cry2 = m_cry2Position - m_cry1Position;
//     auto cry1_cry2_len = cry1_cry2.l2();
//     double line_binstart = tofBinIdx * m_bin_size_mm; // bin goes from cry1 to cry2
//     double line_binend = basic::FMath<calculationPrecision>::min((tofBinIdx + 1) * m_bin_size_mm, cry1_cry2_len);
//     // when P is binStart
//     float distanceP1A = line_binstart;
//     float distanceP1B = cry1_cry2_len - line_binstart;
//     float distanceP1A_P1B = distanceP1A - distanceP1B;
//     auto polylineStart = basic::polylinePosition(m_cry1Position, m_scatterPosition, m_cry2Position, distanceP1A_P1B);
//     // when P is binEnd
//     float distanceP2A = line_binend;
//     float distanceP2B = cry1_cry2_len - line_binend;
//     float distanceP2A_P2B = distanceP2A - distanceP2B;
//     auto polylineEnd = basic::polylinePosition(m_cry1Position, m_scatterPosition, m_cry2Position, distanceP2A_P2B);
//     // SA_SB
//     float distanceSA_SB = (m_scatterPosition - m_cry1Position).l2() - (m_scatterPosition - m_cry2Position).l2();
//     auto deltaLen1 = distanceP1A_P1B - distanceSA_SB;
//     auto deltaLen2 = distanceP2A_P2B - distanceSA_SB;
//     float TOFIntergral1;
//     float TOFIntergral2;
//     if (deltaLen1 < 0 && deltaLen2 < 0) // case1 binstart binend all on sa
//     {
//       TOFIntergral2 = 0;
//       auto event = initialEvent(polylineStart, polylineEnd);
//       TOFIntergral1 = intergralBySampling(event);
//     } else if (deltaLen1 > 0 && deltaLen2 > 0) // case2 binstart binend all on sb
//     {
//       TOFIntergral1 = 0;
//       auto event = initialEvent(polylineStart, polylineEnd);
//       TOFIntergral2 = intergralBySampling(event);
//     } else if (deltaLen1 * deltaLen2 < 0) {
//       auto eventA = initialEvent(polylineStart, m_scatterPosition);
//       TOFIntergral1 = intergralBySampling(eventA);
//       auto eventB = initialEvent(m_scatterPosition, polylineEnd);
//       TOFIntergral2 = intergralBySampling(eventB);
//     }
//     return make_vec2<calculationPrecision>(TOFIntergral1, TOFIntergral2);
//   }
// };

// struct _singleScatterSimulationTOF {

//   float *m_out_SSSBlur; // size is dsLORNum * scatterPointNum * TOFBinNum
//   float *m_in_sssAttnCoff;
//   ScatterPoint *m_scatterPoints;
//   const p3df *m_detectorPos;
//   const example::PolygonalSystem m_polygon;
//   const basic::DetectorGeometry m_detectorGeometry;
//   std::size_t m_scatterPointNum;
//   int m_tofBinNum;
//   double m_crystalArea;
//   double m_commonfactor;

//   void operator()(
//       std::size_t idx) {
//     // cal index
//     std::size_t dsLORIdx = idx / (m_scatterPointNum * m_tofBinNum);
//     std::size_t sctPointIdx = idx % (m_scatterPointNum * m_tofBinNum) / m_tofBinNum;
//     std::size_t TOFBinIdx = idx % m_tofBinNum;
//     //==== cal dsCryPos
//     // dawnSampling
//     const basic::DetectorGeometry dsDetectorGeometry(m_detectorGeometry);
//     dsDetectorGeometry.crystalNumU = 1;
//     dsDetectorGeometry.crystalNumV = 1;
//     auto [block1, block2] = example::polygon::calCrystalIDFromLORID(
//         m_polygon, dsDetectorGeometry, dsLORIdx); // downSampling,consider a block as a crystal
//     int index1 = block1 * m_scatterPointNum + sctPointIdx;
//     int index2 = block2 * m_scatterPointNum + sctPointIdx;
//     auto blockPos1 = m_detectorPos[block1];
//     auto blockPos2 = m_detectorPos[block2];

//     // cal vector s -> crystal
//     auto s_cry1 = blockPos1 - __scatterPoints[sctPointIdx].position; // vector s->cry1
//     auto s_cry2 = blockPos2 - __scatterPoints[sctPointIdx].position; // vector s->cry2
//     double distance_cry1_S = s_cry1.l2() * 0.1;                      // cm
//     double distance_cry2_S = s_cry2.l2() * 0.1;                      // cm
//     // cal cosTheta
//     double scatCosTheta = basic::cosine(s_cry1, s_cry2);
//     // cal sctEnergy
//     double scatterEnergy = 511 / (2 - scatCosTheta); // the energy of scatteted photon
//     if (scatterEnergy < __sctProtocol.scannerEffTableEnergy.x || scatterEnergy >
//     __sctProtocol.scannerEffTableEnergy.y)
//       return;
//     // get scannerEff from  table
//     int EffTableIndex =
//         int((scatterEnergy - __sctProtocol.scannerEffTableEnergy.x) / __sctProtocol.scannerEffTableEnergy.z);
//     float scannerEffNow = __scannerEffTable[EffTableIndex];
//     // cal diffCross
//     double diffCross = calDiffCrossSection(scatCosTheta);
//     // cal projectArea
//     auto n1_vector = m_detectorGeometry[block1].directionU.cross(m_detectorGeometry[block1].directionV); // 法向量
//     auto n2_vector = m_detectorGeometry[block2].directionU.cross(m_detectorGeometry[block2].directionV); // 法向量
//     double projectArea1 = basic::calculateProjectionArea(
//         m_crystalArea, n1_vector, s_cry1); // notice:though we use block as crystal when downSampling ,but the
//         geometry
//                                            // is still crystal geometry
//     double projectArea2 = basic::calculateProjectionArea(m_crystalArea, n2_vector, s_cry2);
//     // cal totalAttn
//     double Icry1 = __sssAttnCoff[index1] * scatter::calTotalAttenInScatterEnergy(__sssAttnCoff[index2],
//     scatterEnergy); double Icry2 = __sssAttnCoff[index2] *
//     scatter::calTotalAttenInScatterEnergy(__sssAttnCoff[index1], scatterEnergy);
//     // cal tof activity intergral
//     auto [tofIntergral1, tofIntergral2] = _calTOFActivityIntergral(...)(TOFBinIdx);

//     double sctPointFactor = m_scatterPoints[sctPointIdx].mu * projectArea1 * projectArea2 * diffCross * scannerEffNow
//     /
//                             (distance_cry1_S * distance_cry1_S * distance_cry2_S * distance_cry2_S);
//     double Ia = tofIntergral1 * Icry1;
//     double Ib = tofIntergral2 * Icry2;
//     m_out_SSSBlur[dsLORIdx * m_tofBinNum + TOFBinIdx] += sctPointFactor * (Ia + Ib) * m_commonfactor; // S 层做加和
//   }

//   struct _SSSTOFResolutionBlur {
//     float *m_in_scatLORBlur;
//     float *m_out_sssTOF;
//     float *m_guass;
//     std::size_t m_scatterPointNum;
//     int m_tofBinNum;
//     int m_gaussSize;
//     void operator()(
//         std::size_t idx) // size = dsLorNum * TOFBinNum
//     {
//       size_t dsLorIdx = idx / m_tofBinNum;
//       size_t TOFBinIdx = idx % m_tofBinNum;
//       int startCov = openpni::basic::FMath<int>::max(0, TOFBinIdx - int(m_gaussSize * 0.5));
//       int endCov = openpni::basic::FMath<int>::min(m_tofBinNum - 1, TOFBinIdx + int(m_gaussSize * 0.5));
//       for (int i = startCov; i <= endCov; i++)
//         m_out_sssTOF[dsLorIdx * m_tofBinNum + TOFBinIdx] +=
//             m_in_scatLORBlur[i] * m_guass[i - TOFBinIdx + int(m_gaussSize * 0.5)];
//     }
//   };
// };

// struct _sumBin {
//   float *__in_scatDsMich;
//   float *__out_scatBinSumMich;
//   std::size_t __donwSamplingLORNum;
//   int __timeBinNum;
//   void operator()(
//       std::size_t idx) {
//     std::size_t subLORIndex = idx / __timeBinNum;
//     __out_scatBinSumMich[subLORIndex] += __in_scatDsMich[idx];
//   }
// };

// struct _upSamplingByInterpolation2D {

//   struct _upSamplingBySlice {
//     float *__in_scatBinSumMich;
//     float *__out_scatDSMichFullSlice;

//     int __binNum;
//     int __viewNum;
//     int __sliceNum;
//     int __dsBinNum;
//     int __dsViewNum;

//     example::PolygonalSystem __polygon;
//     basic::DetectorGeometry __detectorGeometry;
//     example::PolygonalSystem __dsPolygon;
//     basic::DetectorGeometry __dsDetectorGeometry;

//     void operator()(
//         std::size_t idx) {
//       auto sliceIdx = idx / (__dsBinNum * __dsViewNum);
//       auto dsbiviIdx = idx % (__dsBinNum * __dsViewNum);
//       auto [ring1, ring2] = example::polygon::calRing1Ring2FromSlice(__polygon, __detectorGeometry, sliceIdx);
//       int dsRing1 = ring1 * example::polygon::getRingNum(__dsPolygon, __dsDetectorGeometry) /
//                     example::polygon::getRingNum(__polygon, __detectorGeometry);
//       int dsRing2 = ring2 * example::polygon::getRingNum(__dsPolygon, __dsDetectorGeometry) /
//                     example::polygon::getRingNum(__polygon, __detectorGeometry);
//       int dsSliceIdx = dsRing1 + example::polygon::getRingNum(__dsPolygon, __dsDetectorGeometry) * dsRing2;

//       __out_scatDSMichFullSlice[idx] = __in_scatBinSumMich[dsbiviIdx + dsSliceIdx * (__dsBinNum * __dsViewNum)];
//     }
//   };

//   struct _2DIngerpolationUpSampling {
//     float *__in_scatDSMichFullSlice;
//     float *__out_scatDsMich;
//     example::PolygonalSystem __polygon;
//     basic::DetectorGeometry __detectorGeometry;
//     example::PolygonalSystem __dsPolygon;
//     basic::DetectorGeometry __dsDetectorGeometry;
//     p3df *__crystalPos;
//     p3df *__detectorPos;

//     void chooseNearestBlockAndCalWeight(
//         int &blockNearest, float &w, int block, int cryID) {
//       int block_left = block - 1;
//       int block_right = block + 1;
//       auto cry_block_left = __crystalPos[cryID] - __detectorPos[block_left];
//       auto cry_block_right = __crystalPos[cryID] - __detectorPos[block_right];
//       auto distance_cry_block_left = cry_block_left.l2();
//       auto distance_cry_block_right = cry_block_right.l2();
//       if (distance_cry_block_left > distance_cry_block_right) {
//         blockNearest = block_right;
//         w = distance_cry_block_right;
//       } else {
//         blockNearest = block_left;
//         w = distance_cry_block_left;
//       }
//     }

//     void operator()(
//         std::size_t lorIdx) {
//       auto [cryA, cryB] = example::polygon::calCrystalIDFromLORID(__polygon, __detectorGeometry, lorIdx);
//       // cal where cry1 cry2 in block,this is also the index in ds image
//       int blockA = cryA / (__detectorGeometry.crystalNumU * __detectorGeometry.crystalNumV);
//       auto cry1_blockA = __crystalPos[cryA] - __detectorPos[blockA];
//       float wA = cry1_blockA.l2();
//       int blockB = cryB / (__detectorGeometry.crystalNumU * __detectorGeometry.crystalNumV);
//       auto cry2_blockB = __crystalPos[cryB] - __detectorPos[blockB];
//       float wB = cry2_blockB.l2();
//       // choose the nearest block and cal the weight
//       float wA_, wB_;
//       int blockA_, blockB_;
//       chooseNearestBlockAndCalWeight(blockA_, wA_, blockA, cryA);
//       chooseNearestBlockAndCalWeight(blockB_, wB_, blockB, cryB);
//       // cal  A-B ,A-B_ ,A_-B ,A_-B_'s dsLORID
//       std::size_t dsLORIDAB =
//           example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA, blockB);
//       std::size_t dsLORIDAB_ =
//           example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA, blockB_);
//       std::size_t dsLORIDA_B =
//           example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA_, blockB);
//       std::size_t dsLORIDA_B_ =
//           example::polygon::calLORIDFromCrystalRectangleID(__dsPolygon, __dsDetectorGeometry, blockA_, blockB_);
//       // because slice has been upSampled
//       auto biviNum = example::polygon::getBinNum(__dsPolygon, __dsDetectorGeometry) *
//                      example::polygon::getViewNum(__dsPolygon, __dsDetectorGeometry);
//       auto sliceNow = lorIdx / (example::polygon::getBinNum(__polygon, __detectorGeometry) *
//                                 example::polygon::getViewNum(__polygon, __detectorGeometry));
//       dsLORIDAB = dsLORIDAB % biviNum + sliceNow * biviNum;
//       dsLORIDAB_ = dsLORIDAB_ % biviNum + sliceNow * biviNum;
//       dsLORIDA_B = dsLORIDA_B % biviNum + sliceNow * biviNum;
//       dsLORIDA_B_ = dsLORIDA_B_ % biviNum + sliceNow * biviNum;
//       // bilinear interpolation
//       float value = __in_scatDSMichFullSlice[dsLORIDAB] * wA * wB + __in_scatDSMichFullSlice[dsLORIDAB_] * wA * wB_ +
//                     __in_scatDSMichFullSlice[dsLORIDA_B] * wA_ * wB + __in_scatDSMichFullSlice[dsLORIDA_B_] * wA_ *
//                     wB_;
//       float w_All = wA * wB + wA * wB_ + wA_ * wB + wA_ * wB_;
//       __out_scatDsMich[lorIdx] = value / w_All;
//     }
//   };
// };

// template <typename ImageValueType>
// struct _ScatterTOF {
//   bool operator()() { // 单个晶体的面积position
//     const float crystalArea =
//         __in_SSSDataView.__detectorGeometry.crystalSizeU * __in_SSSDataView.__detectorGeometry.crystalSizeV;
//     // first of all do sss tof
//     for_each(dsLorNum * scatterPointNum * TOFBinNum,
//              _singleScatterSimulationTOF{__out_sssTOFBlur, __in_sssAttnCoff, __scatterPoints,
//                                          __in_SSSDataView.__crystalPos, __in_SSSDataView.__polygon,
//                                          __in_SSSDataView.__detectorGeometry, scatterPointNum, TOFBinNum,
//                                          crystalArea, commonFactor},
//              cpuThreads);
//     // add gaus blur
//     for_each(dsLorNum * TOFBinNum,
//              _SSSTOFResolutionBlur{__out_sssTOFBlur, __out_sssTOF, __gaussKernel.data(), scatterPointNum, TOFBinNum,
//                                    int(__gaussKernel.size())},
//              cpuThreads);
//     // do up sampling
//     // sum bin
//     for_each(std::size_t dsLorNum *TOFBinNum _sumBin{__out_sssTOF, __out_sssTOFBinSum, dsLorNum, TOFBinNum},
//              cpuThreads);
//     // do up sampling
//     for_each(std::size_t fullSliceNum *dsBinNum *dsViewNum,
//              _upSamplingByInterpolation2D::_upSamplingBySlice{
//                  __out_sssTOFBinSum, __out_scatDSMichFullSlice, TOFBinNum, viewNum, sliceNum, dsBinNum, dsViewNum,
//                  __in_SSSDataView.__polygon, __in_SSSDataView.__detectorGeometry, __dsPolygon, __dsDetectorGeometry},
//              cpuThreads);
//     // do tailFit
//     for_each(std::size_t fullSliceNum *dsBinNum *dsViewNum,
//              _SSSTailFitTOF{__out_sssTOFBinSum, __out_scatDSMichFullSlice, TOFBinNum, viewNum, sliceNum, dsBinNum,
//                             dsViewNum, __in_SSSDataView.__polygon, __in_SSSDataView.__detectorGeometry, __dsPolygon,
//                             __dsDetectorGeometry},
//              cpuThreads);
//   }
// };

// } // namespace sss_tof

// } // namespace openpni::process
