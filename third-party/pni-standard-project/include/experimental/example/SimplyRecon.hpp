#pragma once
#include <cuda_runtime.h>

#include "../core/Image.hpp"
#include "../core/Mich.hpp"
#include "../core/Random.hpp"
#include "../interface/Conv3D.hpp"
#include "../node/MichNorm.hpp"
#include "../node/MichRandom.hpp"
#include "../node/MichScatter.hpp"
namespace openpni::experimental::example {
struct OSEM_params {
  int subsetNum = 12;
  int iterNum = 4;
  int scatterSimulations = 1;
  float binCutRatio = 0;
  float sample_rate = 0.5f;
};
void instant_OSEM_mich_CPU(core::Image3DOutput<float> out_OSEMImg, OSEM_params params, interface::Conv3D &conv3D,
                           float const *michValue, node::MichNormalization *michNorm, node::MichRandom *michRand,
                           node::MichScatter *michScat, core::MichDefine const &mich);
void instant_OSEM_mich_CUDA(core::Image3DOutput<float> out_OSEMImg, OSEM_params params, interface::Conv3D &conv3D,
                            float const *h_michValue, node::MichNormalization *michNorm, node::MichRandom *michRand,
                            node::MichScatter *michScat, node::MichAttn *michAttn, core::MichDefine const &mich);
struct OSEM_TOF_params {
  int subsetNum = 12;
  int iterNum = 4;
  int scatterSimulations = 1;
  float binCutRatio = 0;
  unsigned long long size_GB = 4;
  float sample_rate = 0.5f;
  float timeWindow_ps = 10000;
  int16_t TOF_division = 1000;
  int16_t TOFBinWid_ps = 100;
  std::string randomListmodeFile = "";
  uint32_t listmodeFileTimeBegin_ms = 0;
  uint32_t listmodeFileTimeEnd_ms = 0xFFFFFFFF;
};
void instant_OSEM_listmode_CUDA(core::Image3DOutput<float> out_OSEMImg, OSEM_TOF_params params,
                                interface::Conv3D &conv3D, std::string listmode_path, node::MichNormalization *michNorm,
                                node::MichRandom *michRand, node::MichScatter *michScat, node::MichAttn *michAttn,
                                core::MichDefine const &mich);
void instant_backwardProjection_listmode_CUDA(core::Image3DOutput<float> out_Img, OSEM_TOF_params params,
                                              interface::Conv3D &conv3D, std::string listmode_path,
                                              core::MichDefine const &mich);
void instant_OSEM_listmodeTOF_CUDA(core::Image3DOutput<float> out_OSEMImg, OSEM_TOF_params params,
                                   interface::Conv3D &conv3D, std::string listmode_path,
                                   node::MichNormalization *michNorm, node::MichRandom *michRand,
                                   node::MichScatter *michScat, node::MichAttn *michAttn, core::MichDefine const &mich);
enum class ConvolutionMethod { GAUSSIAN, KNN, GKNN };
struct GaussianConvParams {
  float HWHM;
};
struct KNNConvParams {
  int KNNNumbers;
  float SigmaG2;
  core::Vector<int64_t, 3> FeatureSizeHalf;
  core::Vector<int64_t, 3> SearchSizeHalf;
  core::TensorDataInput<float, 3> d_kemImgTensorDataIn;
};
struct GKNNConvParams {
  int KNNNumbers;
  float HWHM;
  float SigmaG2;
  core::Vector<int64_t, 3> FeatureSizeHalf;
  core::Vector<int64_t, 3> SearchSizeHalf;
  core::TensorDataInput<float, 3> h_kemImgTensorDataIn;
  // core::TensorDataInput<float, 3> d_kemImgTensorDataIn;

  // void bindHTensorDataIn(
  //     core::TensorDataInput<float, 3> const &h_data) {
  //   h_kemImgTensorDataIn = h_data;
  // }
  // void bindDTensorDataIn(
  //     core::TensorDataInput<float, 3> const &d_data) {
  //   d_kemImgTensorDataIn = d_data;
  // }
  // core::TensorDataInput<float, 3> getHTensorDataIn() {
  //   if (h_kemImgTensorDataIn.ptr != nullptr) {
  //     return h_kemImgTensorDataIn;
  //   } else if (d_kemImgTensorDataIn.ptr != nullptr) {
  //     h_kemImgTensorDataIn.grid = d_kemImgTensorDataIn.grid;
  //     auto in_ptr = std::make_unique_for_overwrite<float[]>(d_kemImgTensorDataIn.grid.totalSize());
  //     cudaMemcpy(in_ptr.get(), d_kemImgTensorDataIn.ptr, sizeof(float) * d_kemImgTensorDataIn.grid.totalSize(),
  //                cudaMemcpyDeviceToHost);
  //     h_kemImgTensorDataIn.ptr = in_ptr.get();
  //     return h_kemImgTensorDataIn;
  //   } else {
  //     throw std::runtime_error("No kem image data bound in GKNNConvParams.");
  //   }
  // }
  // core::TensorDataInput<float, 3> getDTensorDataIn() {
  //   if (d_kemImgTensorDataIn.ptr != nullptr) {
  //     return d_kemImgTensorDataIn;
  //   } else if (h_kemImgTensorDataIn.ptr != nullptr) {
  //     d_kemImgTensorDataIn.grid = h_kemImgTensorDataIn.grid;
  //     auto d_data = openpni::make_cuda_sync_ptr_from_hcopy(
  //         std::span<const float>(d_kemImgTensorDataIn.ptr, h_kemImgTensorDataIn.grid.totalSize()),
  //         "GKNNConv3D_kemImg");
  //     d_kemImgTensorDataIn.ptr = d_data.get();
  //     return d_kemImgTensorDataIn;
  //   } else {
  //     throw std::runtime_error("No kem image data bound in GKNNConvParams.");
  //   }
  // }
};
using ConvolutionParams = std::variant<GaussianConvParams, KNNConvParams, GKNNConvParams>;

struct OSEM_TOF_MULTI_params {
  int subsetNum = 12;
  int iterNum = 4;
  int scatterSimulations = 1;
  float binCutRatio = 0;
  unsigned long long size_GB = 4;
  float sample_rate = 0.5f;
  int16_t TOF_division_ps = 1000;
  float timeWindow_ps = 10000;
  int16_t TOFBinWid_ps = 100;
  std::vector<std::string> randomListmodeFile;
  uint32_t timeBegin_ms = 0;
  uint32_t timeEnd_ms = 0x3FFFFFFF;
  std::vector<std::string> listmode_paths;
  core::Image3DOutput<float> h_AttnImg;
  bool doScatter = false;
  int minSectorDiff = 4;
  int randRadialModuleNumS = 6;
  bool doNorm = false;
  std::string normFactorsFile = "";
  bool doSelfNorm = false;
  bool doDeadTime = false;
  float *selfNormMich = nullptr;
  node::DeadTimeTable deadTimetable = {};
  bool doAttn = false;
  core::Image3DInput<float> attnMap;
  float tailFittingThreshold = 0.95;
  float scatterPointsThreshold = 0.02;
  core::Vector<double, 3> scatterEnergyWindow = {350.00, 650.00, 0.15};
  core::Vector<double, 3> scatterEffTableEnergy = {0.01, 700.00, 0.01};
  core::Grids<3, float> scatterPointGrid;
  ConvolutionParams convParams = GaussianConvParams{1.f};
  bool bigVoxelScatterSimulation = false;
  double tofSSS_timeBinWidth_ns = 0;
  double tofSSS_timeBinStart_ns = 0;
  double tofSSS_timeBinEnd_ns = 0;
  double tofSSS_systemTimeRes_ns = 0;
  uint16_t bitmap_gpu_usage = 0xffff; // default use all available GPUs
};

void instant_OSEM_listmodeTOF_MULTI_CUDA(core::Image3DOutput<float> out_OSEMImg, OSEM_TOF_MULTI_params params,
                                         core::MichDefine const &mich);
enum class FBP_RebinMethod { SSRB, FORE };
struct FBP_params {
  FBP_RebinMethod rebinMethod = FBP_RebinMethod::SSRB;
  std::size_t nRingDiff = 5;
  int nSampNumInBin = 256;
  int nSampNumInView = 156;
  int deltalim = 5;
  int klim = 20;
  int wlim = 20;
  double sampling_distance_in_s = 1;
  double detectorLen = 208;
};
void instant_FBP_mich_CUDA(core::Image3DOutput<float> out_FBPImg, FBP_params params, float const *h_michValue,
                           core::MichDefine const &mich);

struct FDK_params {
  float geo_angle = 0.001292;
  float geo_offsetU = 0.030425;
  float geo_offsetV = 0.340016;
  float geo_SDD = 437.0425;
  float geo_SOD = 311.56;
  unsigned fouriorCutoffLength = 250;
  float beamHardenParamA = 0.8;
  float beamHardenParamB = 0;
  float ct_slope = 4704.162109375f;
  float ct_intercept = -988.8380126953125f;
  float co_offset_x = 2.0f;
  float co_offset_y = 0.7f;
  float pixelSizeU = 0.1496f;
  float pixelSizeV = 0.1496f;
};
void instant_FDK_CUDA(core::Image3DOutput<float> out_FDKImg, FDK_params params, io::U16Image const &ctRawData,
                      io::U16Image const &airRawData);

} // namespace openpni::experimental::example
