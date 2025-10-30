
// #include "Correction_impl.hpp"

// namespace openpni::example::polygon::corrections
// {
//   // attn
//   AttnCorrection::AttnCorrection()
//       : m_impl(std::make_unique<AttnCorrection_impl>()) {}

//   AttnCorrection::~AttnCorrection() {};

//   bool AttnCorrection::ringPETAttnCorr(
//       float *__attnCoff, const float *__attnMap,
//       const openpni::basic::Image3DGeometry &__attnMap3dSize, // same as petImg
//       const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo)
//   {
//     return m_impl->AttnMapToAttnCoffAt511keV(__attnCoff, __attnMap, __attnMap3dSize,
//                                              polygon, detectorGeo);
//   }

//   // deadTime
//   DeadTimeCorrection::DeadTimeCorrection(
//       const deadTimeProtocal &__dtPrtocal)
//       : m_impl(std::make_unique<DeadTimeCorrection_impl>(__dtPrtocal)) {}
//   DeadTimeCorrection::~DeadTimeCorrection() = default;

//   bool DeadTimeCorrection::ringPETDeadTimeCorr(
//       double *dt_mich,                // DT是由table计算出来的所以类型相同
//       const float *delay_mich,        // randMich
//       const double *calibrationtable) // size:2 * m_acquisitionNum * m_dsSlice
//   {
//     return m_impl->modelBasedDTComponent(dt_mich, delay_mich, calibrationtable);
//   }
//   // norm
//   NormCorrection::NormCorrection(
//       const normCorrProtocal &__normPrtocal)
//       : m_impl(std::make_unique<NormCorrection_impl>(__normPrtocal)) {}
//   NormCorrection::~NormCorrection() {};

//   std::vector<float> &NormCorrection::getCryCount()
//   {
//     return m_impl->m_cryCount;
//   }
//   std::vector<float> &NormCorrection::getBlockFctA()
//   {
//     return m_impl->m_blockFctA;
//   }
//   std::vector<float> &NormCorrection::getBlockFctT()
//   {
//     return m_impl->m_blockFctT;
//   }
//   std::vector<float> &NormCorrection::getRadialFct()
//   {
//     return m_impl->m_radialFct;
//   }
//   std::vector<float> &NormCorrection::getPlaneFct()
//   {
//     return m_impl->m_planeFct;
//   }
//   std::vector<float> &NormCorrection::getInterferenceFct()
//   {
//     return m_impl->m_interferenceFct;
//   }
//   std::vector<float> &NormCorrection::getCryFct()
//   {
//     return m_impl->m_cryFct;
//   }

//   bool NormCorrection::ringPETNormCorrection(
//       float *normScan_mich, float *fwd_mich)
//   {
//     return m_impl->ringScannerNormFctGenerate(normScan_mich, fwd_mich);
//   }

//   bool NormCorrection::ringPETSelfNormalization(
//       const float *delay_mich)
//   {
//     return m_impl->selfNormalization(delay_mich);
//   }

//   bool NormCorrection::cutBin(
//       int &binStart, int &binEnd, float *normScan_mich, float *fwd_mich)
//   {
//     return m_impl->cutBin(binStart, binEnd, normScan_mich, fwd_mich);
//   }

//   bool NormCorrection::ActivityCorr(
//       float *normScan_mich, const float *fwd_mich)
//   {
//     return m_impl->ActivityCorr(normScan_mich, fwd_mich);
//   }

//   bool NormCorrection::cryCount(
//       const float *normScan_mich)
//   {
//     return m_impl->calCryCount(normScan_mich);
//   }

//   bool NormCorrection::BlockFct()
//   {
//     return m_impl->calBlockFct();
//   }

//   bool NormCorrection::PlaneFct(
//       const float *normScan_mich, const float *fwd_mich)
//   {
//     return m_impl->calPlaneFct(normScan_mich, fwd_mich);
//   }

//   bool NormCorrection::radialFct(
//       const float *normScan_mich, const float *fwd_mich)
//   {
//     return m_impl->calRadialFct(normScan_mich, fwd_mich);
//   }

//   bool NormCorrection::interferenceFct(
//       const float *normScan_mich, const float *fwd_mich)
//   {
//     return m_impl->calInterferenceFct(normScan_mich, fwd_mich);
//   }
//   // rand
//   RandCorrection::RandCorrection()
//       : m_impl(std::make_unique<RandCorrection_impl>()) {}
//   RandCorrection::~RandCorrection() {};

//   bool RandCorrection::ringPETRandCorrection(
//       float *rand_mich, const example::PolygonalSystem &polygon,
//       const basic::DetectorGeometry &detectorGeo, const unsigned minSectorDifference,
//       const unsigned radialModuleNumS)
//   {
//     return m_impl->smoothMichByNiu(rand_mich, polygon, detectorGeo, minSectorDifference,
//                                    radialModuleNumS);
//   }
// // sct
// ScatterCorrection::ScatterCorrection(const scatterProtocal &__scPrtocal)
//     : m_impl(std::make_unique<ScatterCorrection_impl>(__scPrtocal)) {}
// ScatterCorrection::~ScatterCorrection() = default;

// template <typename ProjectionMethod,
//           typename ConvKernel,
//           std::floating_point ImageValueType>
// bool ScatterCorrection::ringPETScatterCorrection_CUDA(
//     float *prompt_mich,
//     float *scatter_mich,
//     float *attnImg,
//     float *norm,
//     float *rand,
//     const openpni::example::RingPETScanner &scanner,
//     basic::DataSetFromMich_CUDA<float> &dataSets,
//     const basic::Image3DGeometry &attnMap3dSize,
//     openpni::example::OSEMParam<ImageValueType> &OSEMparam,
//     const ConvKernel &kernel)
// {
//     return m_impl->scatterCorrection_CUDA(prompt_mich, scatter_mich, attnImg, norm,
//     scanner);
// }

// template <typename ProjectionMethod,
//           typename ConvKernel,
//           std::floating_point ImageValueType>
// bool ScatterCorrection::ringPETScatterCorrection(
//     float *prompt_mich,
//     float *scatter_mich,
//     float *attnImg,
//     float *norm,
//     float *rand,
//     const openpni::example::RingPETScanner &scanner,
//     basic::DataSetFromMich_CUDA<float> &dataSets,
//     const basic::Image3DGeometry &attnMap3dSize,
//     openpni::example::OSEMParam<ImageValueType> &OSEMparam,
//     const ConvKernel &kernel)
// {
//     return m_impl->scatterCorrection_CUDA(prompt_mich, scatter_mich, attnImg, norm,
//     scanner);
// }
// }
// ; // namespace openpni::example::polygon::corrections
