#pragma once
#include <random>
#include <ranges>
#include <thrust/inner_product.h>

#include "../basic/DataView.hpp"
#include "../basic/Image.hpp"
#include "../example/PolygonalSystem.hpp"
#include "../math/EMStep.hpp"
#include "../math/Geometry.hpp"
#include "../process/EM.cuh"
#include "Attenuation.hpp"
#include "Foreach.cuh"
#include "Scatter.hpp"
namespace openpni::process::scatter {
template <typename ImageValueType, typename LinearFitModel>
struct _Scatter_CUDA {
  _SSSDataView<ImageValueType> __in_SSSDataView;
  bool operator()(
      cudaStream_t __stream = cudaStreamDefault) {
    const float crystalArea = __in_SSSDataView.detectorGeometry.crystalSizeU *
                              __in_SSSDataView.detectorGeometry.crystalSizeV * 0.01; // 单个晶体的面积position,cm2
    auto lorNum = example::polygon::getLORNum(__in_SSSDataView.polygon, __in_SSSDataView.detectorGeometry);
    auto sliceNum = example::polygon::getSliceNum(__in_SSSDataView.polygon, __in_SSSDataView.detectorGeometry);
    auto binNum = example::polygon::getBinNum(__in_SSSDataView.polygon, __in_SSSDataView.detectorGeometry);
    auto viewNum = example::polygon::getViewNum(__in_SSSDataView.polygon, __in_SSSDataView.detectorGeometry);
    auto binNumOutFOVOneSide = example::polygon::calBinNumOutFOVOneSide(
        __in_SSSDataView.polygon, __in_SSSDataView.detectorGeometry, __in_SSSDataView.minSectorDifference);
    auto LORNumOneSlice = binNum * viewNum;
    // cal common factor
    auto sssGridNum = __in_SSSDataView.in_d_attnMapImage3dSpan.geometry.voxelNum *
                      __in_SSSDataView.in_d_attnMapImage3dSpan.geometry.voxelSize / __in_SSSDataView.sssGridSize;
    int sssTotalNum = (int)ceil(sssGridNum.x) * (int)ceil(sssGridNum.y) * (int)ceil(sssGridNum.z);
    const constexpr double PI = 3.1415926535898;
    const double sssAverageVolume = __in_SSSDataView.in_d_attnMapImage3dSpan.geometry.voxelSize.x *
                                    __in_SSSDataView.in_d_attnMapImage3dSpan.geometry.voxelSize.y *
                                    __in_SSSDataView.in_d_attnMapImage3dSpan.geometry.voxelSize.z *
                                    __in_SSSDataView.in_d_attnMapImage3dSpan.geometry.voxelNum.x *
                                    __in_SSSDataView.in_d_attnMapImage3dSpan.geometry.voxelNum.y *
                                    __in_SSSDataView.in_d_attnMapImage3dSpan.geometry.voxelNum.z / sssTotalNum * 1e3;
    double totalComptonCrossSection511keV = calTotalComptonCrossSection(511.f);
    double scannerEff511KeV = calScannerEFFWithScatterEnergy(511.f, __in_SSSDataView.scatterEnergyWindow);
    double commonfactor = 0.25 / PI * sssAverageVolume * scannerEff511KeV / totalComptonCrossSection511keV;
    //  do sss
    for_each_CUDA(lorNum,
                  _singleScatterSimulation<ImageValueType>(
                      __in_SSSDataView.out_scatterValue, __in_SSSDataView.in_sssEmission,
                      __in_SSSDataView.in_sssAttnCoff, __in_SSSDataView.in_crystalGeometry,
                      __in_SSSDataView.scatterPoints, __in_SSSDataView.scannerEffTable,
                      __in_SSSDataView.scannerEffTableEnergy, __in_SSSDataView.countScatter, crystalArea, commonfactor,
                      __in_SSSDataView.polygon, __in_SSSDataView.detectorGeometry),
                  __stream);

    // // 同步设备，确保SSS计算完成
    // cudaError_t cudaStatus = cudaDeviceSynchronize();
    // if (cudaStatus != cudaSuccess) {
    //   std::cerr << "CUDA sync failed after SSS: " << cudaGetErrorString(cudaStatus) << std::endl;
    //   return false;
    // }

    // std::cout << "SSS calculation completed. Saving intermediate results..." << std::endl;

    // // 保存SSS中间结果（tail fitting之前）
    // std::vector<ImageValueType> sss_before_tail(lorNum);
    // cudaStatus = cudaMemcpy(sss_before_tail.data(), __in_SSSDataView.out_scatterValue, lorNum *
    // sizeof(ImageValueType),
    //                         cudaMemcpyDeviceToHost);

    // if (cudaStatus != cudaSuccess) {
    //   std::cerr << "CUDA memcpy failed for SSS results: " << cudaGetErrorString(cudaStatus) << std::endl;
    //   return false;
    // }

    // // 保存SSS中间结果到文件
    // std::string sss_mid_file = "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/test_0919/sssDebug/SSSNotailFit.bin";
    // std::ofstream midFile(sss_mid_file, std::ios::binary);
    // if (midFile.is_open()) {
    //   midFile.write(reinterpret_cast<const char *>(sss_before_tail.data()),
    //                 sss_before_tail.size() * sizeof(ImageValueType));
    //   midFile.close();
    //   std::cout << "SSS intermediate result saved to: " << sss_mid_file << std::endl;
    // }
    // normalization before tail fitting
    double oldSum = thrust::reduce(thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue),
                                   thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue + lorNum), 0.0);
    thrust::transform(thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue),
                      thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue + lorNum),
                      thrust::device_pointer_cast(__in_SSSDataView.in_normMich),
                      thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue),
                      thrust::multiplies<ImageValueType>());
    double newSum = thrust::reduce(thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue),
                                   thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue + lorNum), 0.0);
    double scale = oldSum / newSum;
    thrust::transform(thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue),
                      thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue + lorNum),
                      thrust::device_pointer_cast(__in_SSSDataView.out_scatterValue),
                      [scale] __device__(ImageValueType val) { return val / scale; });
    // do tailFitting
    std::cout << "Start tail fitting..." << std::endl;
    for_each_CUDA(sliceNum,
                  _sssTailFitting<ImageValueType, LinearFitModel>(
                      __in_SSSDataView.out_scatterValue, __in_SSSDataView.in_promptMich, __in_SSSDataView.in_normMich,
                      __in_SSSDataView.in_randMich, __in_SSSDataView.in_attnCutBedCoff, binNum, LORNumOneSlice,
                      binNumOutFOVOneSide, __in_SSSDataView.scatterTailFittingThreshold),
                  __stream);
  }
};

template <typename T>
inline void output_device_array(
    const cuda_sync_ptr<T> &d_ptr, std::string fileName) {
  const auto h_vec = make_vector_from_cuda_sync_ptr(d_ptr);
  std::ofstream outFile(fileName, std::ios::binary);
  outFile.write(reinterpret_cast<const char *>(h_vec.data()), h_vec.size() * sizeof(T));
}

template <typename ImageValueType, typename LinearFitModel, typename ProjectionMethod>
struct _ScatterTOF_CUDA {
  void operator()(
      const _SSSTOFDataView<ImageValueType, ProjectionMethod> &__in_SSSTOFDataView,
      cudaStream_t __stream = cudaStreamDefault) {
    const float crystalArea =
        __in_SSSTOFDataView.detectorGeometry.crystalSizeU * __in_SSSTOFDataView.detectorGeometry.crystalSizeV;
    auto dsLorNum = example::polygon::getLORNum(__in_SSSTOFDataView.dsPolygon, __in_SSSTOFDataView.dsDetectorGeometry);
    auto dsBinNum = example::polygon::getBinNum(__in_SSSTOFDataView.dsPolygon, __in_SSSTOFDataView.dsDetectorGeometry);
    auto dsViewNum =
        example::polygon::getViewNum(__in_SSSTOFDataView.dsPolygon, __in_SSSTOFDataView.dsDetectorGeometry);
    auto lorNum = example::polygon::getLORNum(__in_SSSTOFDataView.polygon, __in_SSSTOFDataView.detectorGeometry);
    auto binNum = example::polygon::getBinNum(__in_SSSTOFDataView.polygon, __in_SSSTOFDataView.detectorGeometry);
    auto viewNum = example::polygon::getViewNum(__in_SSSTOFDataView.polygon, __in_SSSTOFDataView.detectorGeometry);
    auto sliceNum = example::polygon::getSliceNum(__in_SSSTOFDataView.polygon, __in_SSSTOFDataView.detectorGeometry);
    auto LORNumOneSlice = binNum * sliceNum;
    auto binNumOutFOVOneSide = example::polygon::calBinNumOutFOVOneSide(
        __in_SSSTOFDataView.polygon, __in_SSSTOFDataView.detectorGeometry, __in_SSSTOFDataView.minSectorDifference);

    std::cout << lorNum << " " << dsLorNum << " " << binNum << " " << dsBinNum << " " << viewNum << " " << dsViewNum
              << " " << sliceNum << std::endl;
    auto dslorTofBinSize = dsLorNum * __in_SSSTOFDataView.tofBinNum;
    std::cout << "dslorTofBinSize: " << dslorTofBinSize << std::endl;
    // cal common factor
    std::cout << "cal common factor..." << std::endl;
    auto sssGridNum = __in_SSSTOFDataView.in_d_attnMapImage3dSpan.geometry.voxelNum *
                      __in_SSSTOFDataView.in_d_attnMapImage3dSpan.geometry.voxelSize / __in_SSSTOFDataView.sssGridSize;
    int sssTotalNum = (int)ceil(sssGridNum.x) * (int)ceil(sssGridNum.y) * (int)ceil(sssGridNum.z);
    const constexpr double PI = 3.1415926535898;
    const double sssAverageVolume = __in_SSSTOFDataView.in_d_attnMapImage3dSpan.geometry.voxelSize.x *
                                    __in_SSSTOFDataView.in_d_attnMapImage3dSpan.geometry.voxelSize.y *
                                    __in_SSSTOFDataView.in_d_attnMapImage3dSpan.geometry.voxelSize.z *
                                    __in_SSSTOFDataView.in_d_attnMapImage3dSpan.geometry.voxelNum.x *
                                    __in_SSSTOFDataView.in_d_attnMapImage3dSpan.geometry.voxelNum.y *
                                    __in_SSSTOFDataView.in_d_attnMapImage3dSpan.geometry.voxelNum.z / sssTotalNum * 1e3;
    double totalComptonCrossSection511keV = calTotalComptonCrossSection(511.f);
    double scannerEff511KeV = calScannerEFFWithScatterEnergy(511.f, __in_SSSTOFDataView.scatterEnergyWindow);
    double commonfactor = 0.25 / PI * sssAverageVolume * scannerEff511KeV / totalComptonCrossSection511keV;
    std::cout << "common factor: " << commonfactor << std::endl;
    // first of all do sss tof
    std::cout << "do sss tof..." << std::endl;
    auto __d_tofBinSSS = openpni::make_cuda_sync_ptr<float>(dslorTofBinSize);
    auto __d_sssOneLOR = openpni::make_cuda_sync_ptr<float>(__in_SSSTOFDataView.tofBinNum);
    auto __d_sssOneLORBlur = openpni::make_cuda_sync_ptr<float>(__in_SSSTOFDataView.tofBinNum);
    for_each_CUDA(dsLorNum,
                  _singleScatterSimulationTOF<ProjectionMethod>{
                      __d_tofBinSSS.get(), __d_sssOneLOR.get(), __d_sssOneLORBlur.get(),
                      __in_SSSTOFDataView.in_sssAttnCoff, __in_SSSTOFDataView.in_guassBlur,
                      __in_SSSTOFDataView.in_crystalGeometry, __in_SSSTOFDataView.scatterPoints,
                      __in_SSSTOFDataView.in_scannerEffTable, __in_SSSTOFDataView.dsPolygon,
                      __in_SSSTOFDataView.dsDetectorGeometry, __in_SSSTOFDataView.projector,
                      __in_SSSTOFDataView.in_d_emap3dSpan, __in_SSSTOFDataView.scannerEffTableEnergy,
                      __in_SSSTOFDataView.countScatter, crystalArea, __in_SSSTOFDataView.tofBinWidth, commonfactor,
                      __in_SSSTOFDataView.tofBinNum, __in_SSSTOFDataView.gaussSize},
                  __stream);
    output_device_array(__d_tofBinSSS,
                        "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/test_0909/sss_tof_before_upsampling.bin");
    // do up sampling
    // sum bin
    std::cout << "do up sampling..." << std::endl;
    auto __d_sumBinDsSSS = openpni::make_cuda_sync_ptr<float>(dsLorNum);
    for_each_CUDA(dsLorNum * __in_SSSTOFDataView.tofBinNum,
                  _sumBin{__d_sumBinDsSSS, __d_tofBinSSS, dsLorNum, __in_SSSTOFDataView.tofBinNum}, __stream);
    // do up sampling
    auto __d_dsSSSfullSlice = openpni::make_cuda_sync_ptr<float>(sliceNum * dsBinNum * dsViewNum);
    for_each_CUDA(sliceNum * dsBinNum * dsViewNum,
                  _upSamplingByInterpolation2D::_upSamplingBySlice{
                      __d_dsSSSfullSlice, __d_sumBinDsSSS, __in_SSSTOFDataView.polygon,
                      __in_SSSTOFDataView.detectorGeometry, __in_SSSTOFDataView.dsPolygon,
                      __in_SSSTOFDataView.dsDetectorGeometry},
                  __stream);
    // do 2d interpolation
    std::cout << "do 2d interpolation..." << std::endl;
    auto __d_sssMich = openpni::make_cuda_sync_ptr<float>(lorNum);
    for_each_CUDA(lorNum,
                  _upSamplingByInterpolation2D::_2DIngerpolationUpSampling{
                      __d_sssMich, __d_dsSSSfullSlice, __in_SSSTOFDataView.polygon,
                      __in_SSSTOFDataView.detectorGeometry, __in_SSSTOFDataView.dsPolygon,
                      __in_SSSTOFDataView.dsDetectorGeometry, __in_SSSTOFDataView.in_crystalGeometry,
                      __in_SSSTOFDataView.in_dsCrystalGeometry},
                  __stream);
    // normalization before tail fitting
    std::cout << "normalization before tail fitting..." << std::endl;
    double oldSum = thrust::reduce(thrust::device_pointer_cast(__d_sssMich.get()),
                                   thrust::device_pointer_cast(__d_sssMich.get() + lorNum), 0.0);
    thrust::transform(thrust::device_pointer_cast(__d_sssMich.get()),
                      thrust::device_pointer_cast(__d_sssMich.get() + lorNum),
                      thrust::device_pointer_cast(__in_SSSTOFDataView.in_normMich),
                      thrust::device_pointer_cast(__d_sssMich.get()), thrust::multiplies<ImageValueType>());
    double newSum = thrust::reduce(thrust::device_pointer_cast(__d_sssMich.get()),
                                   thrust::device_pointer_cast(__d_sssMich.get() + lorNum), 0.0);
    double scale = oldSum / newSum;
    thrust::transform(
        thrust::device_pointer_cast(__d_sssMich.get()), thrust::device_pointer_cast(__d_sssMich.get() + lorNum),
        thrust::device_pointer_cast(__d_sssMich.get()), [scale] __device__(ImageValueType val) { return val / scale; });
    // do tail fitting
    for_each_CUDA(sliceNum,
                  _sssTailFittingTOF<ImageValueType, LinearFitModel>{
                      __in_SSSTOFDataView.out_dsSSSTOFfullSliceWithTOFBin, __d_sssMich.get(), __d_tofBinSSS.get(),
                      __in_SSSTOFDataView.in_promptMich, __in_SSSTOFDataView.in_normMich,
                      __in_SSSTOFDataView.in_randMich, __in_SSSTOFDataView.in_attnCutBedCoff,
                      __in_SSSTOFDataView.polygon, __in_SSSTOFDataView.detectorGeometry, __in_SSSTOFDataView.dsPolygon,
                      __in_SSSTOFDataView.dsDetectorGeometry, __in_SSSTOFDataView.minSectorDifference,
                      __in_SSSTOFDataView.tofBinNum, __in_SSSTOFDataView.scatterTailFittingThreshold},
                  __stream);
    // normallization
    thrust::transform(thrust::device_pointer_cast(__in_SSSTOFDataView.out_dsSSSTOFfullSliceWithTOFBin),
                      thrust::device_pointer_cast(__in_SSSTOFDataView.out_dsSSSTOFfullSliceWithTOFBin +
                                                  sliceNum * dsBinNum * dsViewNum * __in_SSSTOFDataView.tofBinNum),
                      thrust::device_pointer_cast(__in_SSSTOFDataView.out_dsSSSTOFfullSliceWithTOFBin),
                      [scale] __device__(ImageValueType val) { return val / scale; });
  }
};

} // namespace openpni::process::scatter
