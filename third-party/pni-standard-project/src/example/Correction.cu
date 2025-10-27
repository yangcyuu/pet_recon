
// #include "Correction_impl.cuh"

// namespace openpni::example::polygon::corrections {
// // attn
// AttnCorrection_CUDA::AttnCorrection_CUDA()
//     : m_impl(std::make_unique<AttnCorrection_CUDA::AttnCorrection_CUDA_impl>()) {}

// AttnCorrection_CUDA::~AttnCorrection_CUDA() {};

// bool AttnCorrection_CUDA::ringPETAttnCorr_CUDA(
//     float *__d_attnCoff, const float *__d_attnMap,
//     const openpni::basic::Image3DGeometry &__attnMap3dSize, // same as petImg
//     const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detectorGeo) {
//   return m_impl->AttnMapToAttnCoffAt511keV_CUDA(__d_attnCoff, __d_attnMap, __attnMap3dSize,
//   polygon, detectorGeo);
// }
// // sct
// ScatterCorrection_CUDA::ScatterCorrection_CUDA(
//     const ScatterCorrection_CUDA::scatterProtocal &__scPrtocal)
//     : m_impl(std::make_unique<ScatterCorrection_CUDA::ScatterCorrection_CUDA_impl>(__scPrtocal))
//     {}

// ScatterCorrection_CUDA::~ScatterCorrection_CUDA() {};

// bool ScatterCorrection_CUDA::ringPETScatterCorrection_CUDA(
//     float *prompt_mich, float *scatter_mich, float *attnImg, float *norm, float *rand, const
//     basic::Image3DGeometry &attnMap3dSize,
//     std::vector<basic::DataViewQTY<example::polygon::RearrangerOfSubsetForMich, float, float>>
//     dataViewForSenmap,
//     std::vector<basic::DataViewQTY<example::polygon::RearrangerOfSubsetForMich, float, float>>
//     dataViewForOSEM, const ConvKernel5 *__d_kernel) {
//   return m_impl->scatterCorrection_CUDA(prompt_mich, scatter_mich, attnImg, norm, rand,
//   attnMap3dSize, dataViewForSenmap, dataViewForOSEM,
//                                         __d_kernel);
// }
// } // namespace openpni::example::polygon::corrections
