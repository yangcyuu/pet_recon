// #pragma once
// #include <fstream>

// #include "include/basic/CudaPtr.hpp"
// #include "include/basic/DataView.hpp"
// #include "include/example/PolygonalSystem.hpp"
// #include "include/math/EMStep.hpp"
// #include "include/process/EM.cuh"
// #include "include/process/Foreach.cuh"

// namespace openpni::example::polygon::corrections {
// __device__ void getCrystalIDFromLORID(size_t LORID, int *cryID1, int *cryID2, int cryNumOneRing,
//                                       int ringNum);

// __PNI_CUDA_MACRO__ double calScatCosTheta_new(basic::Vec3<float> cryPosA,
//                                               basic::Vec3<float> cryPosB,
//                                               basic::Vec3<float> sctPos);

// __PNI_CUDA_MACRO__ double calTotalComptonCrossSectionRelativeTo511keV(double scatterEnergy);

// __PNI_CUDA_MACRO__ double calTotalAttenInScatterEnergy(double totalAtten511, double
// scatterEnergy);

// __PNI_CUDA_MACRO__ double calDiffCrossSection(double scatCosTheta);

// void singleScatterSimulation_kernel_impl(
//     float *d_scatMich_noTailFitting, const openpni::basic::Vec3<float> *d_crystalPos,
//     const ScatterCorrection_CUDA::scatterPoint3D *d_scatPoint, const float *d_totalEmission,
//     const float *d_totalAttn, const float *d_scannerEff, const float *d_projectArea,
//     size_t firstLORIndex, int scatterPointNum, int binNum, int sliceNum, int crystalNumOneRing,
//     int ringNum, int binNumOutFOVOneSide, double scannerEffTableEnergyLowerLim,
//     double scannerEffTableEnergyUpperLim, double scannerEffTableEnergyInterval, size_t LORRemain,
//     // 剩余的LOR的数目 int batch = 4096);

// __device__ void getCrystalIDFromLORID(
//     size_t LORID, int *cryID1, int *cryID2, int cryNumOneRing, int ringNum) {
//   int binNum = cryNumOneRing - 1;
//   int viewNum = cryNumOneRing / 2;
//   int bin = LORID % binNum;
//   int view = LORID / binNum % viewNum;
//   int slice = LORID / (binNum * viewNum);
//   int cry1, cry2, ring1, ring2;
//   cry2 = bin / 2 + 1;
//   cry1 = cryNumOneRing + (1 - bin % 2) - cry2;
//   cry2 = (cry2 + view) % cryNumOneRing;
//   cry1 = (cry1 + view) % cryNumOneRing;
//   ring1 = slice / ringNum;
//   ring2 = slice % ringNum;
//   *cryID1 = cry1 + ring1 * cryNumOneRing;
//   *cryID2 = cry2 + ring2 * cryNumOneRing;
// }

// __PNI_CUDA_MACRO__ double calScatCosTheta_new(
//     basic::Vec3<float> cryPosA, basic::Vec3<float> cryPosB, basic::Vec3<float> sctPos) {
//   auto _as = cryPosA - sctPos;  // pointA to sctpoint
//   double d_as = _as.l2() * 0.1; // cm
//   auto _sb = cryPosB - sctPos;  // sctpoint to pointB
//   double d_sb = _sb.l2() * 0.1; // cm
//   auto _ab = cryPosA - cryPosB; // pointA to pointB
//   double d_ab = _ab.l2() * 0.1; // cm
//   return -(d_as * d_as + d_sb * d_sb - d_ab * d_ab) / (2 * d_as * d_sb);
// }
// __PNI_CUDA_MACRO__ double calTotalComptonCrossSectionRelativeTo511keV(
//     double scatterEnergy) {
//   const double a = scatterEnergy / 511.0;
//   // Klein-Nishina formula for a=1 & devided with 0.75 == (40 - 27*log(3)) / 9
//   const double prefactor = 9.0 / (-40 + 27 * log(3.));

//   return // checked this in Mathematica
//       prefactor * (((-4 - a * (16 + a * (18 + 2 * a))) / ((1 + 2 * a) * (1 + 2 * a)) +
//                     ((2 + (2 - a) * a) * log(1 + 2 * a)) / a) /
//                    (a * a));
// }
// __PNI_CUDA_MACRO__ double calTotalAttenInScatterEnergy(
//     double totalAtten511, double scatterEnergy) {
//   return pow(totalAtten511, calTotalComptonCrossSectionRelativeTo511keV(scatterEnergy));
// }
// __PNI_CUDA_MACRO__ double calDiffCrossSection(
//     double scatCosTheta) {
//   // Kelin-Nishina formula. re is classical electron radius
//   const double Re = 2.818E-13;               // cm
//   double waveRatio = 1 / (2 - scatCosTheta); //  lamda/lamda'
//   return 0.5 * Re * Re * waveRatio * waveRatio *
//          (waveRatio + 1 / waveRatio + scatCosTheta * scatCosTheta - 1);
// }

// __global__ void singleScatterSimulation_kernel(
//     float *d_scatMich_noTailFitting, const openpni::basic::Vec3<float> *d_crystalPos,
//     const ScatterCorrection_CUDA::scatterPoint3D *d_scatPoint, const float *d_totalEmission,
//     const float *d_totalAttn, const float *d_scannerEff, const float *d_projectArea,
//     size_t firstLORIndex, int scatterPointNum, int binNum, int sliceNum, int crystalNumOneRing,
//     int ringNum, int binNumOutFOVOneSide, double scannerEffTableEnergyLowerLim,
//     double scannerEffTableEnergyUpperLim, double scannerEffTableEnergyInterval) {
//   __shared__ double cache[128];
//   int cacheIndex = threadIdx.x;
//   if (cacheIndex >= 128) {
//     printf("[ERROR] cacheIndex %d exceeds 128!\n", cacheIndex);
//     return; // Ensure cacheIndex is within bounds
//   }
//   cache[cacheIndex] = 0;
//   double scat = 0; // intermediate variable to store scatter value

//   size_t LORIndex = firstLORIndex + blockIdx.x;
//   int binIndex = LORIndex % binNum;
//   if (binIndex < binNumOutFOVOneSide || binIndex >= binNum - binNumOutFOVOneSide)
//     return;

//   int scatPointIndex = threadIdx.x;
//   while (scatPointIndex < scatterPointNum) {
//     int crystal1, crystal2;
//     getCrystalIDFromLORID(LORIndex, &crystal1, &crystal2, crystalNumOneRing, ringNum);
//     int index1 = crystal1 * scatterPointNum + scatPointIndex;
//     int index2 = crystal2 * scatterPointNum + scatPointIndex;

//     double scatCosTheta = calScatCosTheta_new(d_crystalPos[crystal1], d_crystalPos[crystal2],
//                                               d_scatPoint[scatPointIndex].loc);
//     double scatterEnergy = 511 / (2 - scatCosTheta);
//     if (scatterEnergy < scannerEffTableEnergyLowerLim ||
//         scatterEnergy > scannerEffTableEnergyUpperLim)
//       continue;

//     double scannerEff = d_scannerEff[int((scatterEnergy - scannerEffTableEnergyLowerLim) /
//                                          scannerEffTableEnergyInterval)];
//     double totalEmission1 = d_totalEmission[index1];
//     double totalEmission2 = d_totalEmission[index2];
//     double totalAttn1 = d_totalAttn[index1];
//     double totalAttn2 = d_totalAttn[index2];
//     double projectArea1 = d_projectArea[index1];
//     double projectArea2 = d_projectArea[index2];

//     double Ia =
//         totalEmission1 * totalAttn1 * calTotalAttenInScatterEnergy(totalAttn2, scatterEnergy);
//     double Ib =
//         totalEmission2 * totalAttn2 * calTotalAttenInScatterEnergy(totalAttn1, scatterEnergy);

//     auto P_rAS = d_crystalPos[crystal1] - d_scatPoint[scatPointIndex].loc;
//     double rAS = P_rAS.l2() * 0.1; // cm
//     auto P_rBS = d_crystalPos[crystal2] - d_scatPoint[scatPointIndex].loc;
//     double rBS = P_rBS.l2() * 0.1; // cm
//     double diffCross = calDiffCrossSection(scatCosTheta);
//     scat += d_scatPoint[scatPointIndex].mu * projectArea1 * projectArea2 * diffCross * (Ia + Ib)
//     *
//             scannerEff / (rAS * rAS * rBS * rBS);

//     scatPointIndex += blockDim.x;
//   }

//   cache[cacheIndex] = scat;
//   __syncthreads();

//   // Reduction
//   int i = blockDim.x / 2;
//   while (i != 0) {
//     if (cacheIndex < i)
//       cache[cacheIndex] += cache[cacheIndex + i];
//     __syncthreads();
//     i /= 2;
//   }
//   if (cacheIndex == 0)
//     atomicAdd(&d_scatMich_noTailFitting[LORIndex], cache[0]);
// }

// void singleScatterSimulation_kernel_impl(
//     float *d_scatMich_noTailFitting, const openpni::basic::Vec3<float> *d_crystalPos,
//     const ScatterCorrection_CUDA::scatterPoint3D *d_scatPoint, const float *d_totalEmission,
//     const float *d_totalAttn, const float *d_scannerEff, const float *d_projectArea,
//     size_t firstLORIndex, int scatterPointNum, int binNum, int sliceNum, int crystalNumOneRing,
//     int ringNum, int binNumOutFOVOneSide, double scannerEffTableEnergyLowerLim,
//     double scannerEffTableEnergyUpperLim, double scannerEffTableEnergyInterval, size_t LORRemain,
//     // 剩余的LOR的数目 int batch) {
//   singleScatterSimulation_kernel<<<min((size_t)batch, LORRemain), 128>>>(
//       d_scatMich_noTailFitting, d_crystalPos, d_scatPoint, d_totalEmission, d_totalAttn,
//       d_scannerEff, d_projectArea, firstLORIndex, scatterPointNum, binNum, sliceNum,
//       crystalNumOneRing, ringNum, binNumOutFOVOneSide, scannerEffTableEnergyLowerLim,
//       scannerEffTableEnergyUpperLim, scannerEffTableEnergyInterval);
//   cudaDeviceSynchronize();
// }
// }; // namespace openpni::example::polygon::corrections

// namespace openpni::example::polygon::corrections {

// struct AttnCoffExpFunctor {
//   float *__d_attnCoff;
//   __device__ void operator()(
//       size_t lorIndex) const {
//     __d_attnCoff[lorIndex] = basic::FMath<float>::fexp(-__d_attnCoff[lorIndex]);
//   }
// };
// // attn cuda impl
// class AttnCorrection_CUDA::AttnCorrection_CUDA_impl {
// public:
//   bool AttnMapToAttnCoffAt511keV_CUDA(
//       float *__d_attnCoff, const float *__d_attnMap,
//       const openpni::basic::Image3DGeometry &__attnMap3dSize,
//       const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo) {
//     const auto totalDetectorNum = polygon.getTotalDetectorNum();
//     const auto crystalNumInDetector = detectorGeo.getTotalCrystalNum();
//     auto LORNum = example::polygon::getLORNum(polygon, detectorGeo);

//     openpni::cuda_sync_ptr<openpni::basic::Vec3<float>> d_crystalCoordinates;
//     {
//       std::vector<openpni::basic::Vec3<float>> crystalCoordinates;
//       std::vector<openpni::basic::Coordinate3D<float>> detectorCoordinatesWithDirection;
//       for (const auto detectorIndex : std::views::iota(0u, totalDetectorNum)) {
//         detectorCoordinatesWithDirection.push_back(openpni::example::coordinateFromPolygon(
//             polygon, detectorIndex / (polygon.detectorPerEdge * polygon.edges),
//             detectorIndex % (polygon.detectorPerEdge * polygon.edges)));
//         for (const auto crystalIndex : std::views::iota(0u, crystalNumInDetector)) {
//           const auto coord = openpni::basic::calculateCrystalGeometry(
//               detectorCoordinatesWithDirection.back(), detectorGeo, crystalIndex);
//           crystalCoordinates.push_back(coord.position);
//         }
//       }
//       d_crystalCoordinates = openpni::make_cuda_sync_ptr_from_hcopy(crystalCoordinates);
//     }

//     // EM cuda_dataset
//     openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float, float>
//         dataSetForEMSum;
//     dataSetForEMSum.qtyValue = nullptr;
//     dataSetForEMSum.crystalPosition = d_crystalCoordinates.data();
//     dataSetForEMSum.indexer.scanner = polygon;
//     dataSetForEMSum.indexer.detector = detectorGeo;
//     dataSetForEMSum.indexer.subsetId = 0;
//     dataSetForEMSum.indexer.subsetNum = 1;
//     dataSetForEMSum.indexer.binCut = 0; // no bin cut
//     process::EMSum_CUDA(__d_attnMap, __attnMap3dSize, __attnMap3dSize.roi(), __d_attnCoff,
//                         dataSetForEMSum, openpni::math::ProjectionMethodSiddon());

//     process::for_each_CUDA(LORNum, AttnCoffExpFunctor{__d_attnCoff});

//     return true;
//   }
// };

// // sct cuda impl
// class ScatterCorrection_CUDA::ScatterCorrection_CUDA_impl {
// public:
//   explicit ScatterCorrection_CUDA_impl(
//       const ScatterCorrection_CUDA::scatterProtocal &_scatterProtocal)
//       : m_scatterProtocal(_scatterProtocal) {};

// private: // math
//   double calTotalComptonCrossSection(
//       double energy) const {
//     const double Re = 2.818E-13; // cm
//     double k = energy / 511.f;
//     double a = log(1 + 2 * k);
//     double prefactor = 2 * M_PI * Re * Re;
//     return prefactor * ((1.0 + k) / (k * k) * (2.0 * (1.0 + k) / (1.0 + 2.0 * k) - a / k) +
//                         a / (2.0 * k) - (1.0 + 3.0 * k) / (1.0 + 2.0 * k) / (1.0 + 2.0 * k));
//   }

//   double calScannerEffwithScatter(
//       double energy) const {
//     // The cumulative distribution function (CDF) of the normal, or Gaussian,
//     // distribution with standard deviation sigma and mean mu is:
//     // F(x) = 0.5 * ( 1 + erf((x-mu)/sqrt(2)*sigma) )
//     // 2.35482=2 * sqrt(2 * ln(2)); sqrt(2) = 1.41421
//     double sigmaTimesSqrt2 =
//         m_scatterProtocal.scatterEnergyWindow.otherInfo * 511 / 2.35482 * 1.41421;
//     return 0.5 * (erf((m_scatterProtocal.scatterEnergyWindow.high - energy) / sigmaTimesSqrt2) -
//                   erf((m_scatterProtocal.scatterEnergyWindow.low - energy) / sigmaTimesSqrt2));
//   }

//   double calTotalComptonCrossSectionRelativeTo511keV(
//       double scatterEnergy) const {
//     const double a = scatterEnergy / 511.0;
//     // Klein-Nishina formula for a=1 & devided with 0.75 == (40 - 27*log(3)) / 9
//     static const double prefactor = 9.0 / (-40 + 27 * log(3.));

//     return // checked this in Mathematica
//         prefactor * (((-4 - a * (16 + a * (18 + 2 * a))) / ((1 + 2 * a) * (1 + 2 * a)) +
//                       ((2 + (2 - a) * a) * log(1 + 2 * a)) / a) /
//                      (a * a));
//   }
//   double calTotalAttenInScatterEnergy(
//       double totalAtten511, double scatterEnergy) const {
//     return pow(totalAtten511, calTotalComptonCrossSectionRelativeTo511keV(scatterEnergy));
//   }

//   double calScatCosTheta(
//       int cryIndex1, int scatIndex, int cryIndex2,
//       const std::vector<basic::Vec3<float>> crystalPosition) const {
//     auto _as = crystalPosition[cryIndex1] - m_scatterPoint[scatIndex].loc; // pointA to sctpoint
//     double d_as = _as.l2() * 0.1;                                          // cm
//     auto _sb = crystalPosition[cryIndex2] - m_scatterPoint[scatIndex].loc; // sctpoint to pointB
//     double d_sb = _sb.l2() * 0.1;                                          // cm
//     auto _ab = crystalPosition[cryIndex1] - crystalPosition[cryIndex2];    // pointA to pointB
//     double d_ab = _ab.l2() * 0.1;                                          // cm
//     return -(d_as * d_as + d_sb * d_sb - d_ab * d_ab) / (2 * d_as * d_sb);
//   }

//   double calScatCosTheta_new(
//       basic::Vec3<float> cryPosA, basic::Vec3<float> cryPosB, basic::Vec3<float> sctPos) const {
//     auto _as = cryPosA - sctPos;  // pointA to sctpoint
//     double d_as = _as.l2() * 0.1; // cm
//     auto _sb = cryPosB - sctPos;  // sctpoint to pointB
//     double d_sb = _sb.l2() * 0.1; // cm
//     auto _ab = cryPosA - cryPosB; // pointA to pointB
//     double d_ab = _ab.l2() * 0.1; // cm
//     return -(d_as * d_as + d_sb * d_sb - d_ab * d_ab) / (2 * d_as * d_sb);
//   }

//   double calDiffCrossSection(
//       double scatCosTheta) const {
//     // Kelin-Nishina formula. re is classical electron radius
//     const double Re = 2.818E-13;               // cm
//     double waveRatio = 1 / (2 - scatCosTheta); //  lamda/lamda'
//     return 0.5 * Re * Re * waveRatio * waveRatio *
//            (waveRatio + 1 / waveRatio + scatCosTheta * scatCosTheta - 1);
//   }

//   bool averageFilterApply(
//       std::vector<double> &array, int filterLength = 0) const {
//     int arrayLength = static_cast<int>(array.size()) / 2;
//     std::vector<double> array_temp(array);
//     for (int index = 0; index < arrayLength; index++) {
//       int lowerLimit = std::max(0, index - filterLength);
//       int upperLimit = std::min(arrayLength - 1, index + filterLength);
//       double sum_a = 0;
//       double sum_b = 0;
//       for (int i = lowerLimit; i <= upperLimit; i++) {
//         sum_a += array_temp[2 * i];
//         sum_b += array_temp[2 * i + 1];
//       }
//       array[2 * index] = sum_a / (upperLimit - lowerLimit + 1);
//       array[2 * index + 1] = sum_b / (upperLimit - lowerLimit + 1);
//     }
//     return true;
//   }
//   //==========================================================initialtor
//   bool initScatterPoint(
//       float *attnMap, const openpni::basic::Image3DGeometry &attnMapSize) {
//     openpni::basic::Vec3<float> gridNum;
//     gridNum = attnMapSize.voxelNum * attnMapSize.voxelSize / m_scatterProtocal.scatterGrid;
//     int totalNum = (int)ceil(gridNum.x) * (int)ceil(gridNum.y) * (int)ceil(gridNum.z);
//     m_scatterPoint.reserve(totalNum);
//     std::cout << m_scatterPoint.size() << " scatter points reserved." << std::endl;
//     if (m_scatterProtocal.randomSelect) {
//       std::cout << "Notice: We will select scatter point randomly. " << std::endl;
//       srand(0);
//     }
//     // current grid minimum border
//     openpni::basic::Vec3<float> Lim = {0.0, 0.0, 0.0};
//     openpni::basic::Vec3<float> upperLim;
//     openpni::basic::Vec3<float> pointRand;
//     openpni::basic::Vec3<int> voxelIndex;
//     scatterPoint3D scatterPointTemp;
//     while (Lim.x < attnMapSize.voxelSize.x * attnMapSize.voxelNum.x) {
//       upperLim.x = std::min(Lim.x + m_scatterProtocal.scatterGrid.x,
//                             (float)(attnMapSize.voxelSize.x * attnMapSize.voxelNum.x));
//       Lim.y = 0;
//       while (Lim.y < attnMapSize.voxelSize.y * attnMapSize.voxelNum.y) {
//         upperLim.y = std::min(Lim.y + m_scatterProtocal.scatterGrid.y,
//                               (float)(attnMapSize.voxelSize.y * attnMapSize.voxelNum.y));
//         Lim.z = 0;
//         while (Lim.z < attnMapSize.voxelSize.z * attnMapSize.voxelNum.z) {
//           upperLim.z = std::min(Lim.z + m_scatterProtocal.scatterGrid.z,
//                                 (float)(attnMapSize.voxelSize.z * attnMapSize.voxelNum.z));
//           pointRand = (upperLim - Lim) * static_cast<float>(m_scatterProtocal.randomSelect) *
//                       rand() / (float)RAND_MAX;
//           pointRand = pointRand + Lim;
//           voxelIndex = basic::make_vec3<int>(pointRand / attnMapSize.voxelSize);
//           if (attnMap[voxelIndex.z * attnMapSize.voxelNum.x * attnMapSize.voxelNum.y +
//                       voxelIndex.y * attnMapSize.voxelNum.x + voxelIndex.x] >=
//               m_scatterProtocal.scatterPointThreshold) {
//             scatterPointTemp.loc =
//                 pointRand - attnMapSize.voxelSize * attnMapSize.voxelNum / 2 +
//                 attnMapSize.centre();
//             // cm-1
//             scatterPointTemp.mu =
//                 attnMap[voxelIndex.z * attnMapSize.voxelNum.x * attnMapSize.voxelNum.y +
//                         voxelIndex.y * attnMapSize.voxelNum.x + voxelIndex.x] *
//                 10;
//             m_scatterPoint.push_back(scatterPointTemp);
//           }
//           Lim.z += m_scatterProtocal.scatterGrid.z;
//         }
//         Lim.y += m_scatterProtocal.scatterGrid.y;
//       }
//       Lim.x += m_scatterProtocal.scatterGrid.x;
//     }
//     m_scatterPoint.shrink_to_fit();
//     if (m_scatterPoint.size() <= 0) {
//       std::cout << "no good scatter point. At initScatterPoint." << std::endl;
//       return false;
//     }
//     return true;
//   }
//   bool initScannerEffTable() {
//     int energyBinNum = int((m_scatterProtocal.scannerEffTableEnergy.high -
//                             m_scatterProtocal.scannerEffTableEnergy.low) /
//                            m_scatterProtocal.scannerEffTableEnergy.otherInfo) +
//                        1;
//     m_scannerEff.resize(energyBinNum, 0);
// #pragma omp parallel for num_threads(m_limitedThreadNum)
//     for (int i = 0; i < energyBinNum; i++)
//       m_scannerEff[i] = static_cast<float>(
//           calScannerEffwithScatter(m_scatterProtocal.scannerEffTableEnergy.low +
//                                    i * m_scatterProtocal.scannerEffTableEnergy.otherInfo));
//     return true;
//   }

//   bool initAttnCutBedCoff(
//       float *attnImg, const openpni::basic::Image3DGeometry &attnMap3dSize,
//       const std::vector<openpni::basic::Vec3<float>> &crystalPosition,
//       const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo) {
//     auto LORNum = example::polygon::getLORNum(polygon, detectorGeo);
//     std::unique_ptr<float[]> attn_fwdMich = std::make_unique<float[]>(LORNum);

//     m_attnCutBedCoff.resize(LORNum, 0);
//     // cutBed
//     for (int i = 0; i < attnMap3dSize.totalVoxelNum(); i++) {
//       if (attnImg[i] < 0.0096 / 3.0) {
//         attnImg[i] = 0.0;
//       }
//     }
//     // fwd
//     const auto cryPos = example::calCrystalPosition(polygon, detectorGeo);
//     basic::DataViewQTY<openpni::example::polygon::RearrangerOfSubsetForMich, float, float>
//         dataSetForEMSum;
//     dataSetForEMSum.qtyValue = nullptr;
//     dataSetForEMSum.crystalPosition = cryPos.data();
//     dataSetForEMSum.indexer.scanner = polygon;
//     dataSetForEMSum.indexer.detector = detectorGeo;
//     dataSetForEMSum.indexer.subsetId = 0;
//     dataSetForEMSum.indexer.subsetNum = 1;
//     dataSetForEMSum.indexer.binCut = 0; // no bin cut
//     // process::EMSum(attnImg, attnMap3dSize, attnMap3dSize.roi(), attn_fwdMich.get(),
//     dataSetForEMSum, openpni::math::ProjectionMethodSiddon(),
//         //                basic::CpuMultiThread::callWithAllThreads());

//         for (auto LORIndex = 0; LORIndex < LORNum; LORIndex++) {
//       m_attnCutBedCoff[LORIndex] = exp(-m_attnCutBedCoff[LORIndex]);
//     }
//     // lgxtest save m_attnCutBedCoff to file
//     std::ofstream attnCutBedCoffFile("/media/wzx/d7517b17-b5a8-48e2-a3ef-401b7cb5fb2e/"
//                                      "lgxtest/sss_attnCutBedCoff0624.img3d",
//                                      std::ios::binary);
//     if (attnCutBedCoffFile.is_open()) {
//       attnCutBedCoffFile.write(reinterpret_cast<char *>(m_attnCutBedCoff.data()),
//                                sizeof(float) * m_attnCutBedCoff.size());
//       attnCutBedCoffFile.close();
//     }
//     return true;
//   }

//   template <std::floating_point attnValueType>
//   bool initTotalAttn(
//       const std::vector<basic::Vec3<float>> &crystalPosition, const attnValueType *attnImg,
//       const basic::Image3DGeometry &image3dSize) {
//     m_totalAttn.resize(crystalPosition.size() * m_scatterPoint.size(), 0);
//     std::cout << "initTotalAttn: crystalPosition.size() = " << crystalPosition.size()
//               << ",
//                  m_scatterPoint.size() = " << m_scatterPoint.size()
//                                          << std::endl;
//     std::vector<basic::Vec2<basic::Vec3<float>>> totalCryPos;
//     totalCryPos.reserve(crystalPosition.size() * m_scatterPoint.size());
//     for (const auto i : std::views::iota(std::size_t(0), crystalPosition.size())) {
//       for (const auto j : std::views::iota(std::size_t(0), m_scatterPoint.size())) {
//         totalCryPos.emplace_back(crystalPosition[i], m_scatterPoint[j].loc);
//       }
//     }

//     // basic::PosViewForSum<float, float> totalCryPosView;
//     // totalCryPosView.posLOR = totalCryPos.data();

//     // process::EMSum(attnImg, image3dSize, image3dSize.roi(), m_totalAttn.data(),
//     totalCryPosView, openpni::math::ProjectionMethodSiddon(),
//         //  basic::CpuMultiThread::callWithAllThreads());

//         for (auto &attn : m_totalAttn) {
//       attn = exp(-attn);
//     }

//     return true;
//   }

//   // bool initTotalEmission(
//   //     const std::vector<basic::Vec3<float>> &crystalPosition, const float *reconImg, const
//   basic::Image3DGeometry &image3dSize) {
//     //   m_totalEmission.resize(crystalPosition.size() * m_scatterPoint.size(), 0);
//     //   std::vector<basic::Vec2<basic::Vec3<float>>> totalCryPos;
//     //   totalCryPos.reserve(crystalPosition.size() * m_scatterPoint.size());
//     //   for (const auto i : std::views::iota(std::size_t(0), crystalPosition.size())) {
//     //     for (const auto j : std::views::iota(std::size_t(0), m_scatterPoint.size())) {
//     //       totalCryPos.emplace_back(
//     //           basic::Vec2<basic::Vec3<float>>{crystalPosition[i], m_scatterPoint[j].loc}); //
//     第indexNow =
//         i * m_scatterPoint.size() +
//         // // j与m_totalEmission.data() 对应
//             //     }
//             //   }
//             //   basic::PosViewForSum<float, float> totalCryPosView;
//             //   totalCryPosView.posLOR = totalCryPos.data();
//             //   totalCryPosView.count = totalCryPos.size();
//             //   process::EMSum(reconImg, image3dSize, image3dSize.roi(), m_totalEmission.data(),
//             totalCryPosView,
//     openpni::math::ProjectionMethodSiddon(),
//     //                  basic::CpuMultiThread::callWithAllThreads());
//     //   return true;
//     // }

//         bool InitProjectArea(const std::vector<openpni::basic::Vec3<float>> &crystalPosition) {
//       auto cryNumOneRing = example::polygon::getCrystalNumOneRing(m_scatterProtocal.polygon,
//                                                                   m_scatterProtocal.detectorGeo);
//       auto crystalNumYInPanel = example::polygon::getCrystalNumYInPanel(
//           m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       int panelNum = m_scatterProtocal.polygon.edges * m_scatterProtocal.polygon.detectorPerEdge;
//       double crystalArea =
//           m_scatterProtocal.detectorGeo.crystalSizeU * m_scatterProtocal.detectorGeo.crystalSizeV
//           * 0.01; // cm2 int crystalNum = crystalPosition.size(); int scatterPointNum =
//       m_scatterPoint.size();
//       m_projectArea.resize(crystalNum * scatterPointNum, 0);
// #pragma omp parallel for num_threads(m_limitedThreadNum)
//       for (int cryIndex = 0; cryIndex < crystalNum; cryIndex++) {
//         int crystalInRing = cryIndex % cryNumOneRing;
//         int panel = crystalInRing / crystalNumYInPanel;
//         double sita = double(panel) * 2 * M_PI / panelNum;
//         auto crystalPosNow = crystalPosition[cryIndex];
//         for (int scatIndex = 0; scatIndex < scatterPointNum; scatIndex++) {
//           auto c_s = crystalPosition[cryIndex] - m_scatterPoint[scatIndex].loc;
//           double c2sDistance = c_s.l2() * 0.1; // cm
//           auto scatterPosNow = m_scatterPoint[scatIndex].loc;
//           double cosTheta = (crystalPosNow.x - scatterPosNow.x) * cos(sita) +
//                             (crystalPosNow.y - scatterPosNow.y) * sin(sita) / c2sDistance *
//                                 0.1; // cm m_projectArea[cryIndex *
//         scatterPointNum + scatIndex] = static_cast<float>(crystalArea * cosTheta);
//         }
//       }
//       return true;
//     }

//     //====================================================================Function
//   bool getScaleBySlice(
//       const float *scatNoTailFitting, int ring1, int ring2, double &a, double &b, const float
//       *prompt, const float *norm, // default : 0 const float *rand) // default : 1
//   {
//       auto ringNum =
//           example::polygon::getRingNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto binNum =
//           example::polygon::getBinNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto viewNum =
//           example::polygon::getViewNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto LORNumOneSlice = binNum * viewNum;
//       auto binNumOutFOVOneSide = example::polygon::calBinNumOutFOVOneSide(
//           m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo,
//           m_scatterProtocal.minSectorDifference);
//       size_t sliceIndexNow = ring1 * ringNum + ring2;
//       size_t indexStart = sliceIndexNow * LORNumOneSlice;

//       int num = 0;      // n
//       double sumXY = 0; // sum(x*y)
//       double sumX = 0;  // sum(x)
//       double sumY = 0;  // sum(y)
//       double sumXX = 0; // sum(x*x)

//       for (int LORIndex = 0; LORIndex < LORNumOneSlice; LORIndex++) {
//         int binIndex = LORIndex % binNum;
//         if (m_attnCutBedCoff[LORIndex + indexStart] >=
//                 m_scatterProtocal.scatterTailFittingThreshold &&
//             binIndex >= binNumOutFOVOneSide && binIndex < binNum - binNumOutFOVOneSide) {
//           if (norm[LORIndex + indexStart] == 0)
//             continue;
//           num++;
//           sumX += scatNoTailFitting[LORIndex + indexStart];
//           sumY += prompt[LORIndex + indexStart] - rand[LORIndex + indexStart];
//           sumXY += scatNoTailFitting[LORIndex + indexStart] *
//                    (prompt[LORIndex + indexStart] - rand[LORIndex + indexStart]);
//           sumXX +=
//               scatNoTailFitting[LORIndex + indexStart] * scatNoTailFitting[LORIndex +
//               indexStart];
//         }
//       }

//       if (m_scatterProtocal.with_bias) {
//         a = (num * sumXY - sumX * sumY) / (num * sumXX - sumX * sumX);
//         b = (sumXX * sumY - sumX * sumXY) / (num * sumXX - sumX * sumX);
//       } else {
//         a = fabs(sumXY / sumXX);
//         b = 0;
//       }
//       return true;
//   }

//   bool doScaleBySlice(
//       float *scatter, const float *scat_noTailFitting, int ring1, int ring2, double a, double b,
//       const float *norm) {
//       auto binNum =
//           example::polygon::getBinNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto sliceNum =
//           example::polygon::getSliceNum(m_scatterProtocal.polygon,
//           m_scatterProtocal.detectorGeo);
//       auto ringNum =
//           example::polygon::getRingNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto LORNumOneSlice = binNum * sliceNum;
//       size_t sliceIndexNow = ring1 * ringNum + ring2;
//       size_t indexStart = sliceIndexNow * LORNumOneSlice;
//       auto binNumOutFOVOneSide = example::polygon::calBinNumOutFOVOneSide(
//           m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo,
//           m_scatterProtocal.minSectorDifference);
//       for (size_t LORIndex = 0; LORIndex < LORNumOneSlice; LORIndex++) {
//         int binIndex = LORIndex % binNum;
//         if (binIndex >= binNumOutFOVOneSide && binIndex < binNum - binNumOutFOVOneSide) {
//           // exclude the bad channel
//           if (norm[LORIndex + indexStart] == 0) {
//             scatter[LORIndex + indexStart] = 0;
//             continue;
//           }
//           scatter[LORIndex + indexStart] =
//               static_cast<float>(a * scat_noTailFitting[LORIndex + indexStart] + b);
//         }

//         else
//           scatter[LORIndex + indexStart] = 0;
//       }
//       return true;
//   }
//   bool singleScatterSimulation(
//       float *scatterTriangle_noTailFitting, const std::vector<openpni::basic::Vec3<float>>
//       &crystalPosition, const float *reconImg, const basic::Image3DGeometry &image3dSize) //
//       最好不要用cpu版本
//   {
//       auto binNum =
//           example::polygon::getBinNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto viewNum =
//           example::polygon::getViewNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto sliceNum =
//           example::polygon::getSliceNum(m_scatterProtocal.polygon,
//           m_scatterProtocal.detectorGeo);
//       auto LORNum =
//           example::polygon::getLORNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto binNumOutFOVOneSide = example::polygon::calBinNumOutFOVOneSide(
//           m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo,
//           m_scatterProtocal.minSectorDifference);
//       openpni::basic::Vec3<float> scatterGridNum;
//       scatterGridNum = image3dSize.voxelSize * image3dSize.voxelNum /
//       m_scatterProtocal.scatterGrid; int totalNum =
//           (int)ceil(scatterGridNum.x) * (int)ceil(scatterGridNum.y) *
//           (int)ceil(scatterGridNum.z);
//       const double scatter_volume = image3dSize.voxelSize.x * image3dSize.voxelNum.x *
//                                     image3dSize.voxelSize.y * image3dSize.voxelNum.y *
//                                     image3dSize.voxelSize.z * image3dSize.voxelNum.z / totalNum *
//                                     1e3;
//       const double totalComptonCrossSection511keV = calTotalComptonCrossSection(511.f);
//       const double ScannerEff511keV = calScannerEffwithScatter(511.f);
//       double common_factor =
//           0.25 / M_PI * scatter_volume * ScannerEff511keV / totalComptonCrossSection511keV;

//       int scatterPointNum = static_cast<int>(m_scatterPoint.size());
//       int LORNumOneSlice = viewNum * binNum;
//       std::cout << "here" << std::endl;
//       for (int sliceIndex = 0; sliceIndex < sliceNum; sliceIndex++) {
//         std::cout << "sliceIndex = " << sliceIndex << std::endl;
// #pragma omp parallel for num_threads(m_limitedThreadNum)
//         for (int indexOneSlice = 0; indexOneSlice < LORNumOneSlice; indexOneSlice++) {
//           int LORIndex = sliceIndex * LORNumOneSlice + indexOneSlice;
//           int binIndex = LORIndex % binNum;
//           if (binIndex < binNumOutFOVOneSide || binIndex >= binNum - binNumOutFOVOneSide)
//             continue;
//           const auto [cryID1, cryID2] = example::polygon::calCrystalIDFromLORID(
//               m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo, LORIndex);
//           for (int scatPointIndex = 0; scatPointIndex < scatterPointNum; scatPointIndex++) {
//             // the cosine of scatter angle
//             int index1 = cryID1 * scatterPointNum + scatPointIndex;
//             int index2 = cryID2 * scatterPointNum + scatPointIndex;
//             double scatCosTheta = calScatCosTheta(cryID1, scatPointIndex, cryID2,
//             crystalPosition); double scatterEnergy = 511 / (2 - scatCosTheta); // the energy of
//             scatteted photon if (scatterEnergy < m_scatterProtocal.scannerEffTableEnergy.low ||
//                 scatterEnergy > m_scatterProtocal.scannerEffTableEnergy.high)
//               continue;
//             double scannerEffNow =
//                 m_scannerEff[int((scatterEnergy - m_scatterProtocal.scannerEffTableEnergy.low) /
//                                  m_scatterProtocal.scannerEffTableEnergy.otherInfo)];

//             double Ia = m_totalEmission[index1] * m_totalAttn[index1] *
//                         calTotalAttenInScatterEnergy(m_totalAttn[index2], scatterEnergy);
//             double Ib = m_totalEmission[index2] * m_totalAttn[index2] *
//                         calTotalAttenInScatterEnergy(m_totalAttn[index1], scatterEnergy);

//             auto p_rAS = crystalPosition[cryID1] - m_scatterPoint[scatPointIndex].loc;
//             double rAS = p_rAS.l2() * 0.1; // cm
//             auto p_rBS = crystalPosition[cryID2] - m_scatterPoint[scatPointIndex].loc;
//             double rBS = p_rBS.l2() * 0.1; // cm

//             scatterTriangle_noTailFitting[LORIndex] +=
//                 static_cast<float>(m_scatterPoint[scatPointIndex].mu * m_projectArea[index1] *
//                                    m_projectArea[index2] * calDiffCrossSection(scatCosTheta) *
//                                    (Ia + Ib) * scannerEffNow / (rAS * rAS * rBS * rBS));
//           }
//           scatterTriangle_noTailFitting[LORIndex] *= common_factor;
//         }
//       }
//       return true;
//   }

//   bool singleScatterSimulation_CUDA(
//       float *scatterMich_noTailFitting, const std::vector<openpni::basic::Vec3<float>>
//       &crystalPosition, const float *reconImg, const basic::Image3DGeometry &image3dSize) {
//       auto binNum =
//           example::polygon::getBinNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto viewNum =
//           example::polygon::getViewNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto sliceNum =
//           example::polygon::getSliceNum(m_scatterProtocal.polygon,
//           m_scatterProtocal.detectorGeo);
//       auto ringNum =
//           example::polygon::getRingNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto LORNum =
//           example::polygon::getLORNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto binNumOutFOVOneSide = example::polygon::calBinNumOutFOVOneSide(
//           m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo,
//           m_scatterProtocal.minSectorDifference);
//       auto crystalNumOneRing = example::polygon::getCrystalNumOneRing(
//           m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       // int LORNumOneSlice = viewNum * binNum;
//       openpni::basic::Vec3<float> scatterGridNum;
//       scatterGridNum = image3dSize.voxelSize * image3dSize.voxelNum /
//       m_scatterProtocal.scatterGrid; int totalNum =
//           (int)ceil(scatterGridNum.x) * (int)ceil(scatterGridNum.y) *
//           (int)ceil(scatterGridNum.z);
//       const double scatter_volume = image3dSize.voxelSize.x * image3dSize.voxelNum.x *
//                                     image3dSize.voxelSize.y * image3dSize.voxelNum.y *
//                                     image3dSize.voxelSize.z * image3dSize.voxelNum.z / totalNum *
//                                     1e3;
//       const double totalComptonCrossSection511keV = calTotalComptonCrossSection(511.f);
//       const double ScannerEff511keV = calScannerEffwithScatter(511.f);
//       double common_factor =
//           0.25 / M_PI * scatter_volume * ScannerEff511keV / totalComptonCrossSection511keV;
//       int scatterPointNum = static_cast<int>(m_scatterPoint.size());

//       // prepare device data
//       std::cout << "start allocate d_crystalPos" << std::endl;
//       auto d_crystalPos = make_cuda_sync_ptr_from_hcopy(crystalPosition);
//       std::cout << "start allocate d_scatterPoint" << std::endl;
//       auto d_scatterPoint = make_cuda_sync_ptr_from_hcopy(m_scatterPoint);
//       std::cout << "start allocate d_projectArea" << std::endl;
//       auto d_projectArea = make_cuda_sync_ptr_from_hcopy(m_projectArea);
//       std::cout << "start allocate d_totalEmission" << std::endl;
//       auto d_totalEmission = make_cuda_sync_ptr_from_hcopy(m_totalEmission);
//       std::cout << "start allocate d_totalAttn" << std::endl;
//       auto d_totalAttn = make_cuda_sync_ptr_from_hcopy(m_totalAttn);
//       std::cout << "start allocate d_scannerEff" << std::endl;
//       auto d_scannerEff = make_cuda_sync_ptr_from_hcopy(m_scannerEff);
//       std::cout << "start allocate d_scatterTriangle_noTailFitting" << std::endl;
//       auto d_scatMich_noTailFitting = make_cuda_sync_ptr_from_hcopy(
//           std::span<const float>(scatterMich_noTailFitting, scatterMich_noTailFitting + LORNum));
//       // batch = 4096
//       int batch = 4096;
//       std::cout << "start doing kernel" << std::endl;
//       for (size_t lorIndex = 0; lorIndex < LORNum; lorIndex += batch) {
//         singleScatterSimulation_kernel_impl(
//             d_scatMich_noTailFitting, d_crystalPos, d_scatterPoint, d_totalEmission, d_totalAttn,
//             d_scannerEff, d_projectArea, lorIndex, scatterPointNum, binNum, sliceNum,
//             crystalNumOneRing, ringNum, binNumOutFOVOneSide,
//             m_scatterProtocal.scannerEffTableEnergy.low,
//             m_scatterProtocal.scannerEffTableEnergy.high,
//             m_scatterProtocal.scannerEffTableEnergy.otherInfo, LORNum - lorIndex, batch);
//       }
//       cudaDeviceSynchronize();

//       std::cout << "kernel done" << std::endl;
//       d_scatMich_noTailFitting.allocator().copy_from_device_to_host(
//           scatterMich_noTailFitting, d_scatMich_noTailFitting.cspan());
//       std::cout << "copy to host done" << std::endl;
//       for (int i = 0; i < LORNum; i++) {
//         scatterMich_noTailFitting[i] *= common_factor;
//       }

//       return true;
//   }

//   bool tailFittingLeastSquareBySlice(
//       float *scatterTriangle, const float *scatterTriangle_noTailFitting, const float *norm,
//       const float *rand, const float *prompt) {
//       auto LORNum =
//           example::polygon::getLORNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       auto ringNum =
//           example::polygon::getRingNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       std::unique_ptr<float[]> scatterTriangle_noTailFittingTemp =
//           std::make_unique<float[]>(LORNum);
//       std::copy(&scatterTriangle_noTailFitting[0], &scatterTriangle_noTailFitting[0] + LORNum,
//                 &scatterTriangle_noTailFittingTemp[0]);
//       double oldSum = 0;
//       double newSum = 0;
//       for (auto LORIndex = 0; LORIndex < LORNum; LORIndex++) {
//         oldSum += scatterTriangle_noTailFittingTemp[LORIndex];
//         scatterTriangle_noTailFittingTemp[LORIndex] *= norm[LORIndex];
//         newSum += scatterTriangle_noTailFittingTemp[LORIndex];
//       }
//       double scale = oldSum / newSum;
//       for (size_t LORIndex = 0; LORIndex < LORNum; LORIndex++)
//         scatterTriangle_noTailFittingTemp[LORIndex] /= static_cast<float>(scale);

//     // -ringNum + 1 < ringdiff < 0
// #pragma omp parallel for num_threads(m_limitedThreadNum)
//       for (int ringdiff = -ringNum + 1; ringdiff < 0; ringdiff++) {
//         int ring1Start = 0;
//         int ring2Start = ring1Start - ringdiff;
//         // store scale value a and b
//         std::vector<double> scale;
//         for (int ring1Temp = ring1Start, ring2Temp = ring2Start; ring2Temp < ringNum;
//              ring1Temp++, ring2Temp++) {
//           double a = 1;
//           double b = 0;
//           getScaleBySlice(scatterTriangle_noTailFittingTemp.get(), ring1Temp, ring2Temp, a, b,
//                           prompt, norm, rand);
//           scale.push_back(a);
//           scale.push_back(b);
//         }
//         averageFilterApply(scale);
//         for (int ring1Temp = ring1Start, ring2Temp = ring2Start; ring2Temp < ringNum;
//         ring1Temp++,
//                  ring2Temp++) { // ring1Temp will always start from 0
//           double a = scale[2 * ring1Temp];
//           double b = scale[2 * ring1Temp + 1];
//           doScaleBySlice(scatterTriangle, scatterTriangle_noTailFittingTemp.get(), ring1Temp,
//                          ring2Temp, a, b, norm);
//         }
//       }
//     // 0 <= ringdiff < ringNum
// #pragma omp parallel for num_threads(m_limitedThreadNum)
//       for (int ringdiff = 0; ringdiff < ringNum; ringdiff++) {
//         int ring2Start = 0;
//         int ring1Start = ringdiff - ring2Start;
//         // store scale value a and b
//         std::vector<double> scale;
//         for (int ring1Temp = ring1Start, ring2Temp = ring2Start; ring1Temp < ringNum;
//              ring1Temp++, ring2Temp++) {
//           double a = 1;
//           double b = 0;
//           getScaleBySlice(scatterTriangle_noTailFittingTemp.get(), ring1Temp, ring2Temp, a, b,
//                           prompt, norm, rand);

//           scale.push_back(a);
//           scale.push_back(b);
//         }
//         averageFilterApply(scale);
//         for (int ring1Temp = ring1Start, ring2Temp = ring2Start; ring1Temp < ringNum;
//         ring1Temp++,
//                  ring2Temp++) { // ring2Temp will always start from 0
//           double a = scale[2 * ring2Temp];
//           double b = scale[2 * ring2Temp + 1];
//           doScaleBySlice(scatterTriangle, scatterTriangle_noTailFittingTemp.get(), ring1Temp,
//                          ring2Temp, a, b, norm);
//         }
//       }

//       return true;
//   }

// public: // main
//   bool scatterCorrection_CUDA(
//       float *prompt_mich,  // reconMich
//       float *scatter_mich, // saving final result
//       float *attnImg,      // must have
//       float *norm,         // default : 1，only use norm mich with: float* blockProfA, float*
//                            // blockProfT, float*
//                            crystalFct，它们是生成归一化mich是伴随生成的部分组件
//       float *rand,         // default : 0,must have in v3,需要smooth预处理
//       const basic::Image3DGeometry &attnMap3dSize,
//       std::vector<basic::DataViewQTY<example::polygon::RearrangerOfSubsetForMich, float, float>>
//       dataViewForSenmap,
//       std::vector<basic::DataViewQTY<example::polygon::RearrangerOfSubsetForMich, float, float>>
//       dataViewForOSEM, const ConvKernel5 *d_kernel) {
//       const auto LORNum =
//           example::polygon::getLORNum(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       const auto cryPos =
//           example::calCrystalPosition(m_scatterProtocal.polygon, m_scatterProtocal.detectorGeo);
//       std::vector<float> scatterTmpMich(LORNum, 0);
//       // 0.get a copy of michAdd
//       std::cout << "initScatterPoint" << std::endl;
//       if (initScatterPoint(attnImg, attnMap3dSize) != true) {
//         std::cout << "error at initScatterPoint" << std::endl;
//         return false;
//       }
//       std::cout << "initScannerEffTable" << std::endl;
//       if (initScannerEffTable() != true) {
//         std::cout << "error at initScannerEfftable" << std::endl;
//         return false;
//       }
//       if (initAttnCutBedCoff(attnImg, attnMap3dSize, cryPos, m_scatterProtocal.polygon,
//                              m_scatterProtocal.detectorGeo) != true) {
//         std::cout << "error at initAttnCutBedCoff" << std::endl;
//         return false;
//       }
//       std::cout << "initTotalAttn" << std::endl;
//       if (initTotalAttn(cryPos, attnImg, attnMap3dSize) != true) {
//         std::cout << "error at initTotalAttn" << std::endl;
//         return false;
//       }
//       std::cout << "InitProjectArea" << std::endl;
//       if (InitProjectArea(cryPos) != true) // these are not depended on reconImg
//       {
//         std::cout << "error at InitProjectArea" << std::endl;
//         return false;
//       }

//       // 2.do OSEM
//       for (int iteration = 0; iteration < m_scatterProtocal.iterationNum; iteration++) {
//         std::vector<float> imgOSEM(attnMap3dSize.totalVoxelNum(), 0);
//         std::vector<openpni::cuda_sync_ptr<float>> d_senmap;
//         d_senmap.push_back(openpni::make_cuda_sync_ptr<float>(attnMap3dSize.totalVoxelNum()));
//         if (iteration == 0) {
//           // osem only calSenmap at first timedataSetsForRecon
//           process::calSenmap_CUDA(dataViewForSenmap.front(), attnMap3dSize,
//           d_senmap.back().get(),
//                                   d_kernel, openpni::math::ProjectionMethodSiddon());
//           openpni::process::fixSenmap_simple_CUDA(d_senmap.back().get(), attnMap3dSize, 0.05f);
//         }

//         std::cout << "doing scatterEsitimate[" << iteration << "]" << std::endl;
//         openpni::cuda_sync_ptr<float> d_imgOSEM =
//             openpni::make_cuda_sync_ptr<float>(attnMap3dSize.totalVoxelNum());
//         openpni::cuda_sync_ptr<char> d_buffer;
//         std::size_t bufferSize = 0;
//         auto t =
//             d_senmap | std::views::transform([](const auto &ptr) noexcept { return ptr.get(); });
//         auto v = std::vector<float *>(t.begin(), t.end());
//         while (!openpni::process::SEM_simple_CUDA(
//             dataViewForOSEM, attnMap3dSize, d_imgOSEM.data(), d_kernel, v, 1, d_buffer.get(),
//             bufferSize, openpni::math::ProjectionMethodSiddon(),
//             openpni::process::EMSumSimpleUpdate_CUDA(),
//             openpni::process::ImageSimpleUpdate_CUDA()))

//         {
//           bufferSize += 20 * 1024 * 1024;
//           std::cout << "Resize buffer to " << bufferSize << " bytes." << std::endl;
//           d_buffer = openpni::make_cuda_sync_ptr<char>(bufferSize);
//         }
//         std::cout << "doing totalEmission" << std::endl;
//         // test out OSEM
//         imgOSEM = make_vector_from_cuda_sync_ptr(d_imgOSEM);
//         // d_imgOSEM.copyToHost(imgOSEM.data());
//         std::ofstream midOSEM("/media/wzx/d7517b17-b5a8-48e2-a3ef-401b7cb5fb2e/lgxtest/"
//                               "sss_midOSEM_CUDA0701.img3d",
//                               std::ios::binary);
//         if (midOSEM.is_open()) {
//           midOSEM.write(reinterpret_cast<char *>(imgOSEM.data()), sizeof(float) *
//           imgOSEM.size()); midOSEM.close();
//         }
//         // if (initTotalEmission(cryPos, imgOSEM.data(), attnMap3dSize) != true) // depended on
//         reconImg
//             // {
//             //   std::cout << "error at initTotalEmission" << std::endl;
//             //   return false;
//             // }
//             std::vector<float>
//                 scattermich_noTailFitting(LORNum, 0);
//         std::cout << "doing singleScatterSimulation" << std::endl;
//         if (singleScatterSimulation_CUDA(scattermich_noTailFitting.data(), cryPos,
//         imgOSEM.data(),
//                                          attnMap3dSize) != true) {
//           std::cout << "error at SingleScatterSimulation_CUDA" << std::endl;
//           return false;
//         }
//         // test singleSCat
//         std::cout << "singleScatterSimulation finished" << std::endl;
//         std::ofstream mideScat("/media/wzx/d7517b17-b5a8-48e2-a3ef-401b7cb5fb2e/lgxtest/"
//                                "sss_midSctNoFit0624.img3d",
//                                std::ios::binary);
//         if (mideScat.is_open()) {
//           mideScat.write(reinterpret_cast<char *>(scattermich_noTailFitting.data()),
//                          sizeof(float) * scattermich_noTailFitting.size());
//           mideScat.close();
//         }
//         // 3. do tail fittings
//         std::cout << "doing tailFittingLeastSquareBySlice" << std::endl;
//         std::fill(scatterTmpMich.begin(), scatterTmpMich.end(), 0.0f);
//         if (tailFittingLeastSquareBySlice(scatterTmpMich.data(),
//         scattermich_noTailFitting.data(),
//                                           norm, rand, prompt_mich) != true) {
//           std::cout << "error at tailFittingLeastSquareBySlice" << std::endl;
//           return false;
//         }
//         std::cout << "finish tailFittingLeastSquareBySlice" << std::endl;
//       }
//       for (int i = 0; i < LORNum; i++) {
//         scatter_mich[i] = scatterTmpMich[i];
//       }
//       return true;
//   }

// public:
//   int m_limitedThreadNum = 8;

// private:
//   ScatterCorrection_CUDA::scatterProtocal m_scatterProtocal;
//   std::vector<float> m_scannerEff;            // energyEffTable,which size is int((upperEnergy -
//                                               // lowerEnergy) / energyInterval) + 1)
//   std::vector<scatterPoint3D> m_scatterPoint; // 散射点location + mu
//   std::vector<float> m_totalAttn;             // size = model中cry数目 *
//   scatterPoint中location的数目 std::vector<float> m_attnCutBedCoff;        // fwd of umapCutBed
//   std::vector<float> m_totalEmission;
//   std::vector<float> m_projectArea;
//   };
// } // namespace openpni::example::polygon::corrections
