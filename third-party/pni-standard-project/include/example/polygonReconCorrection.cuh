#pragma once
#include <ranges>

#include "../basic/CudaPtr.hpp"
#include "../process/Attenuation.cuh"
#include "../process/EM.hpp"
#include "../process/Scatter.cuh"
#include "../process/Scatter.hpp"
#include "PolygonCalculation.hpp"
#include "PolygonRecon.cuh"
#include "PolygonalSystem.hpp"
namespace openpni::example {
inline void calAttnCoffWithCTImg_CUDA(
    float *out_AttnFactor, float *in_AttnMap, example::polygon::PolygonModel &model,
    cudaStream_t __stream = cudaStreamDefault) {
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy(model.crystalGeometry());
  openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float> dataView;
  dataView.qtyValue = nullptr;
  dataView.indexer.scanner = model.polygonSystem();
  dataView.indexer.detector = model.detectorInfo().geometry;
  dataView.indexer.subsetId = 0;
  dataView.indexer.subsetNum = 1;
  dataView.indexer.binCut = 0;
  dataView.crystalGeometry = d_crystalGeometry;
  basic::Image3DGeometry imgGeo{{0.5f, 0.5f, 0.5f}, {-80, -80, -100}, {320, 320, 400}};

  auto d_in_AttnMap =
      openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(in_AttnMap, imgGeo.totalVoxelNum()));

  auto d_out_attnFactor = openpni::make_cuda_sync_ptr<float>(model.michSize());
  Image3DSpan<const float> imgSpan{imgGeo, d_in_AttnMap};

  process::attn::cal_attn_coff_CUDA(dataView, imgSpan, d_out_attnFactor.get(), process::attn::attn_model.v0(),
                                    openpni::math::ProjectionMethodSiddon(), __stream);

  d_out_attnFactor.allocator().copy_from_device_to_host(out_AttnFactor, d_out_attnFactor.cspan());
  // // rearrnge attn coff
  // auto copy = out_AttnFactor;
  // openpni::example::polygon::IndexerOfSubsetForMich rearranger;
  // rearranger.scanner = model.polygonSystem();
  // rearranger.detector = model.detectorInfo().geometry;
  // rearranger.subsetId = 0;
  // rearranger.subsetNum = 1;
  // rearranger.binCut = 0; // no bin cut
  // for (const auto index : std::views::iota(0ull, model.michSize()))
  //   out_AttnFactor[rearranger[index]] = copy[index];
}
inline void calAttnCoffWithHUMap_CUDA(
    float *out_AttnFactor, float *in_AttnMap, example::polygon::PolygonModel &model,
    cudaStream_t __stream = cudaStreamDefault) {
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy(model.crystalGeometry());
  openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float> dataView;
  dataView.qtyValue = nullptr;
  dataView.indexer.scanner = model.polygonSystem();
  dataView.indexer.detector = model.detectorInfo().geometry;
  dataView.indexer.subsetId = 0;
  dataView.indexer.subsetNum = 1;
  dataView.indexer.binCut = 0;
  dataView.crystalGeometry = d_crystalGeometry;
  basic::Image3DGeometry imgGeo{{0.5f, 0.5f, 0.5f}, {-80, -80, -100}, {320, 320, 400}};
  auto d_in_AttnMap =
      openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(in_AttnMap, imgGeo.totalVoxelNum()));

  auto d_out_attnFactor = openpni::make_cuda_sync_ptr<float>(model.michSize());

  Image3DSpan<const float> imgSpan{imgGeo, d_in_AttnMap};

  process::attn::cal_attn_coff_CUDA(dataView, imgSpan, d_out_attnFactor.get(), process::attn::attn_model.v1(),
                                    openpni::math::ProjectionMethodSiddon(), __stream);

  d_out_attnFactor.allocator().copy_from_device_to_host(out_AttnFactor, d_out_attnFactor.cspan());
}
inline void generateSSS_CUDA(
    float *out_sssMich, float *attnMapCutBed, float *osemImgData, float *attnMapCutBedCoff, float *promptMich,
    float *randMich, float *normSSSMich, example::polygon::PolygonModel &model, basic::Image3DGeometry attnMapGeometry,
    basic::Image3DGeometry osemImgGeometry) {
  auto ImgFov = attnMapGeometry.voxelNum * attnMapGeometry.voxelSize;
  //=====1. preparSSS
  openpni::basic::Vec3<float> sssGrid{10.f, 10.f, 10.f};
  //   openpni::basic::Image3DGeometry sss3DGeometry =
  //       basic::make_ImageSizeByCenter(sssGrid, {0, 0, 0}, {ImgFov.x / 10, ImgFov.y / 10, ImgFov.z / 10});
  //   auto sssRandom = openpni::process::scatter::_ScatterPointsGenerateRules().centerRandom(sss3DGeometry);
  //   std::cout << "random points size: " << sssRandom.size() << std::endl;
  //   // choose right sss points by attnCutBed map
  //   std::cout << "choose sss points..." << std::endl;
  //   openpni::Image3DInputSpan<float> attnMapCutBedSpan{attnMapGeometry, attnMapCutBed};
  //   auto sssPoints = openpni::process::scatter::isScatterPoint(10, sssRandom, attnMapCutBedSpan, 0.00124);
  //   std::vector<openpni::p3df> sssPointPositions;
  //   for (const auto &point : sssPoints) {
  //     sssPointPositions.push_back(point.sssPosition.position);
  //   }
  //   std::cout << "scatter points size: " << sssPoints.size() << std::endl;
  //=====v3 generate SSSpoint
  auto sssPoints = openpni::process::scatter::_ScatterPointsGenerateRules().initScatterPoint(
      attnMapCutBed, attnMapGeometry, sssGrid, 0.00124);
  std::vector<openpni::p3df> sssPointPositions;
  for (const auto &point : sssPoints) {
    sssPointPositions.push_back(point.sssPosition.position);
  }

  //======2.generate scatterEffTable
  openpni::basic::Vec3<double> sssTableEnergy{0.01, 700.00, 0.01};
  openpni::basic::Vec3<double> sssEnergyWindow{350.00, 650.00, 0.15};
  auto scatterEffTable = openpni::process::scatter::generateScatterEffTable(sssTableEnergy, sssEnergyWindow);
  std::cout << "scatterEffTable.size(): " << scatterEffTable.size() << std::endl;
  //=====3.generate sss_attnCoff
  // do sssAttnCoff
  std::cout << " generate sssAttnCoff..." << std::endl;
  std::vector<float> sssAttnCoff(model.crystalNum() * sssPoints.size(), 0);
  calculateScatterIntegrals(sssAttnCoff.data(), attnMapCutBed, attnMapGeometry, model, sssPointPositions.data(),
                            sssPointPositions.size());
  for (auto &val : sssAttnCoff)
    val = std::exp(-val);

  //===== 4. generate sssEmission
  std::cout << " generate sssEmission..." << std::endl;
  std::vector<float> sssEmission(model.crystalNum() * sssPoints.size(), 0);
  // temptest
  auto empGeo = example::make_ImgGeometry({80, 80, 100}, {2, 2, 2});
  //

  calculateScatterIntegrals(sssEmission.data(), osemImgData, osemImgGeometry, model, sssPointPositions.data(),
                            sssPointPositions.size());

  //=====5. memcp all these to gpu ,then prepare for _sssCuda
  // sssPoint gpu
  auto d_sssPoints = openpni::make_cuda_sync_ptr_from_hcopy(sssPoints);
  auto sssPointNum = sssPoints.size();
  sssPoints.clear();
  sssPoints.shrink_to_fit();
  // 0 scannerEffTable
  auto d_scatterEffTable = openpni::make_cuda_sync_ptr_from_hcopy(scatterEffTable);
  scatterEffTable.clear();
  scatterEffTable.shrink_to_fit();
  // 1 sssAttnCoff
  auto d_sssAttnCoff = openpni::make_cuda_sync_ptr_from_hcopy(sssAttnCoff);
  sssAttnCoff.clear();
  sssAttnCoff.shrink_to_fit();
  // 2 sssEmission
  auto d_sssEmission = openpni::make_cuda_sync_ptr_from_hcopy(sssEmission);
  sssEmission.clear();
  sssEmission.shrink_to_fit();
  // 3 attnCuBedCoff，promptMich，randMich，normMich
  auto d_attnCutBedCoff =
      openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(attnMapCutBedCoff, model.michSize()));
  auto d_promptMich = openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(promptMich, model.michSize()));
  auto d_randMich = openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(randMich, model.michSize()));
  auto d_normSSSMich = openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(normSSSMich, model.michSize()));
  // 4 attnMapImg3dspan gpu，osemImgSpan gpu
  auto d_attnMapCutBed =
      openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(attnMapCutBed, attnMapGeometry.totalVoxelNum()));
  openpni::Image3DInputSpan<float> d_attnMapCutBedSpan{attnMapGeometry, d_attnMapCutBed};
  auto d_osemImg =
      openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(osemImgData, osemImgGeometry.totalVoxelNum()));
  openpni::Image3DInputSpan<float> d_osemImgSpan{osemImgGeometry, d_osemImg};
  // 5 petdataview gpu
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy(model.crystalGeometry());
  //===========6 prepare sssdataview then do sss
  auto d_sssout = openpni::make_cuda_sync_ptr<float>(model.michSize());
  auto sssdataview = openpni::process::_SSSDataView<float>{d_sssout.get(),
                                                           d_sssAttnCoff.get(),
                                                           d_sssEmission.get(),
                                                           d_attnCutBedCoff.get(),
                                                           d_promptMich.get(),
                                                           d_randMich.get(),
                                                           d_normSSSMich.get(),
                                                           d_crystalGeometry.get(),
                                                           d_attnMapCutBedSpan,
                                                           d_osemImgSpan,
                                                           d_scatterEffTable.get(),
                                                           d_sssPoints.get(),
                                                           sssEnergyWindow,
                                                           sssTableEnergy,
                                                           sssGrid,
                                                           sssPointNum,
                                                           4,
                                                           0.95,
                                                           model.polygonSystem(),
                                                           model.detectorInfo().geometry};
  std::cout << "do sssCuda..." << std::endl;
  openpni::process::scatter::_Scatter_CUDA<float, openpni::basic::_LinearFitting::_WithoutBias<float>>{sssdataview}();
  d_sssout.allocator().copy_from_device_to_host(out_sssMich, d_sssout.cspan());
  std::cout << "sss done." << std::endl;
}

struct SSSTOF_Params {
  double m_timeBinWidth = 0.05;            ///< Width of each TOF bin (in nanoseconds).
  double m_timeBinStart = -1.5;            ///< Start time of the TOF bin range (in nanoseconds).
  double m_timeBinEnd = 1.5;               ///< End time of the TOF bin range (in nanoseconds).
  double m_systemTimeRes_ns = 320 * 0.001; ///< System time resolution (in nanoseconds).
  size_t m_tofBinNum;

  void calTofBinNum() {
    m_tofBinNum = ceil((m_timeBinEnd - m_timeBinStart) / m_timeBinWidth);
    m_tofBinNum -= 1 - (m_tofBinNum % 2); // Let the number of time bins be odd
  }
};
inline void generateSSSTOF_CUDA(
    float *out_SSStable, float *attnMapCutBed, float *osemImgData, float *attnMapCutBedCoff, float *promptMich,
    float *randMich, float *normSSSMich, example::polygon::PolygonModel &model, SSSTOF_Params ssstofParams,
    basic::Image3DGeometry attnMapGeometry, basic::Image3DGeometry osemImgGeometry) {
  auto ImgFov = attnMapGeometry.voxelNum * attnMapGeometry.voxelSize;
  openpni::basic::Vec3<float> imageLength{160.f, 160.f, 200.f};
  auto imgGeometry = example::defaultImgGeometry();
  ssstofParams.calTofBinNum();
  //=====1. preparSSS
  openpni::basic::Vec3<float> sssGrid{10.f, 10.f, 10.f};
  // openpni::basic::Image3DGeometry sss3DGeometry{
  //     sssGrid, {0, 0, 0}, {imageLength.x / 10, imageLength.y / 10, imageLength.z / 10}};
  // auto sssRandom = openpni::process::scatter::_ScatterPointsGenerateRules().centerRandom(sss3DGeometry);
  // std::cout << "random points size: " << sssRandom.size() << std::endl;
  // // choose right sss points by attnCutBed map
  // std::cout << "choose sss points..." << std::endl;
  // openpni::Image3DInputSpan<float> attnMapCutBedSpan{imgGeometry, attnMapCutBed};
  // auto sssPoints = openpni::process::scatter::isScatterPoint(10, sssRandom, attnMapCutBedSpan, 0.00124);
  // std::cout << "scatter points size: " << sssPoints.size() << std::endl;
  //=====v3 generate SSSpoint
  auto sssPoints = openpni::process::scatter::_ScatterPointsGenerateRules().initScatterPoint(
      attnMapCutBed, attnMapGeometry, sssGrid, 0.00124);
  std::vector<openpni::p3df> sssPointPositions;
  for (const auto &point : sssPoints) {
    sssPointPositions.push_back(point.sssPosition.position);
  }
  //=====2.generate scatterEffTable
  openpni::basic::Vec3<double> sssTableEnergy{0.01, 700.00, 0.01};
  openpni::basic::Vec3<double> sssEnergyWindow{350.00, 650.00, 0.15};
  auto scatterEffTable = openpni::process::scatter::generateScatterEffTable(sssTableEnergy, sssEnergyWindow);
  std::cout << "scatterEffTable.size(): " << scatterEffTable.size() << std::endl;
  //=====3.generate sss_attnCoff
  std::cout << " generate sssAttnCoff..." << std::endl;
  std::vector<float> sssAttnCoff(model.crystalNum() * sssPoints.size(), 0);
  calculateScatterIntegrals(sssAttnCoff.data(), attnMapCutBed, attnMapGeometry, model, sssPointPositions.data(),
                            sssPointPositions.size());
  for (auto &val : sssAttnCoff)
    val = std::exp(-val);
  //=====4.get gaussBlur
  std::cout << " generate guassBlur..." << std::endl;
  auto guassBlur = openpni::process::scatter::generateGaussianBlurKernel(ssstofParams.m_systemTimeRes_ns,
                                                                         ssstofParams.m_timeBinWidth);
  //=====5. get downSampling system
  // polygonSystem is the same,so
  auto downSampledPolygonSystem = model.polygonSystem();
  auto dsDetectorGeometry = model.detectorInfo().geometry;
  // detectorGeometry changed,crystalNumU and crystalNumV both =1
  dsDetectorGeometry.crystalNumU = 1;
  dsDetectorGeometry.crystalNumV = 1;
  // then get the dsCrystalGeometry
  const auto dsCrystalGeometry = openpni::example::calCrystalGeo(downSampledPolygonSystem, dsDetectorGeometry);
  std::cout << "downSampled crystal num: " << dsCrystalGeometry.size() << std::endl;
  //=====6. get downSampled indexer and then downSampled dataview
  openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float> dsdataView;
  dsdataView.qtyValue = nullptr; // senmap没有"mich信息“
  dsdataView.crystalGeometry = dsCrystalGeometry.data();
  dsdataView.indexer.scanner = downSampledPolygonSystem;
  dsdataView.indexer.detector = dsDetectorGeometry;
  dsdataView.indexer.subsetId = 0;
  dsdataView.indexer.subsetNum = 1;
  dsdataView.indexer.binCut = 0; // No
  //=====7. set projectionMethod
  auto projectionMethod = openpni::math::ProjectionMethodUniform();
  projectionMethod.sampler.setSampleRatio(0.7f);
  //=====8. memcp all these to gpu ,then prepare for _sssCuda
  //  sssPoint gpu
  auto d_sssPoints = openpni::make_cuda_sync_ptr_from_hcopy(sssPoints);
  auto sssPointNum = sssPoints.size();
  sssPoints.clear();
  sssPoints.shrink_to_fit();
  // 0 scannerEffTable
  auto d_scatterEffTable = openpni::make_cuda_sync_ptr_from_hcopy(scatterEffTable);
  scatterEffTable.clear();
  scatterEffTable.shrink_to_fit();
  // 1 sssAttnCoff
  auto d_sssAttnCoff = openpni::make_cuda_sync_ptr_from_hcopy(sssAttnCoff);
  sssAttnCoff.clear();
  sssAttnCoff.shrink_to_fit();
  // 2 attnCuBedCoff，promptMich，randMich，normMich
  auto d_attnCutBedCoff =
      openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(attnMapCutBedCoff, model.michSize()));
  auto d_promptMich = openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(promptMich, model.michSize()));
  auto d_randMich = openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(randMich, model.michSize()));
  auto d_normSSSMich = openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(normSSSMich, model.michSize()));
  // 3 attnMapImg3dspan
  auto d_attnMapCutBed =
      openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(attnMapCutBed, imgGeometry.totalVoxelNum()));
  openpni::Image3DInputSpan<float> d_attnMapCutBedSpan{imgGeometry, d_attnMapCutBed};
  // 4 osemImgSpan gpu
  auto d_osemImg =
      openpni::make_cuda_sync_ptr_from_hcopy(std::span<const float>(osemImgData, imgGeometry.totalVoxelNum()));
  openpni::Image3DInputSpan<float> d_osemImgSpan{imgGeometry, d_osemImg};
  // 5 petdataview gpu
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy(model.crystalGeometry());
  // 6 guassBlur
  auto d_gaussBlur = openpni::make_cuda_sync_ptr_from_hcopy(guassBlur);
  auto gaussSize = guassBlur.size();
  guassBlur.clear();
  guassBlur.shrink_to_fit();
  // 7 dsCrystalGeometry
  auto d_dsCrystalGeometry = openpni::make_cuda_sync_ptr_from_hcopy(dsCrystalGeometry);
  auto sliceNum = openpni::example::polygon::getSliceNum(model.polygonSystem(), model.detectorInfo().geometry);
  auto dsBinNum = openpni::example::polygon::getBinNum(downSampledPolygonSystem, dsDetectorGeometry);
  auto dsviewNum = openpni::example::polygon::getViewNum(downSampledPolygonSystem, dsDetectorGeometry);
  auto d_out_SSStable = openpni::make_cuda_sync_ptr<float>(sliceNum * dsBinNum * dsviewNum * ssstofParams.m_tofBinNum);
  // 8 dsDataView gpu
  dsdataView.crystalGeometry = d_dsCrystalGeometry.get();
  //======prepare sssTOFdataView then do sssTOF
  auto sssTOFDataview = openpni::process::_SSSTOFDataView<float, decltype(projectionMethod)>(
      d_out_SSStable.get(), d_sssAttnCoff.get(), d_attnCutBedCoff.get(), d_promptMich.get(), d_randMich.get(),
      d_normSSSMich.get(), d_sssPoints.get(), d_scatterEffTable.get(), d_gaussBlur.get(), d_crystalGeometry.get(),
      d_dsCrystalGeometry.get(), projectionMethod, d_attnMapCutBedSpan, d_osemImgSpan, sssEnergyWindow, sssTableEnergy,
      sssGrid, sssPointNum, 4, 0.95, ssstofParams.m_timeBinWidth, ssstofParams.m_tofBinNum, gaussSize,
      model.polygonSystem(), model.detectorInfo().geometry, downSampledPolygonSystem, dsDetectorGeometry);
  std::cout << "do sssTOFCuda..." << std::endl;
  openpni::process::scatter::_ScatterTOF_CUDA<float, openpni::basic::_LinearFitting::_WithoutBias<float>,
                                              decltype(projectionMethod)>()(sssTOFDataview);
  d_out_SSStable.allocator().copy_from_device_to_host(out_SSStable, d_out_SSStable.cspan());
  std::cout << "sssTOF done." << std::endl;
}

} // namespace openpni::example
