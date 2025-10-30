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
  int16_t TOF_division = 1000;
  std::string randomListmodeFile = "";
  uint32_t listmodeFileTimeBegin_ms = 0;
  uint32_t listmodeFileTimeEnd_ms = 0xFFFFFFFF;
};
void instant_OSEM_listmode_CUDA(core::Image3DOutput<float> out_OSEMImg, OSEM_TOF_params params,
                                interface::Conv3D &conv3D, std::string listmode_path, node::MichNormalization *michNorm,
                                node::MichRandom *michRand, node::MichScatter *michScat, node::MichAttn *michAttn,
                                core::MichDefine const &mich);
void instant_OSEM_listmodeTOF_CUDA(core::Image3DOutput<float> out_OSEMImg, OSEM_TOF_params params,
                                   interface::Conv3D &conv3D, std::string listmode_path,
                                   node::MichNormalization *michNorm, node::MichRandom *michRand,
                                   node::MichScatter *michScat, node::MichAttn *michAttn, core::MichDefine const &mich);
struct OSEM_TOF_MULTI_params {
  int subsetNum = 12;
  int iterNum = 4;
  int scatterSimulations = 1;
  float binCutRatio = 0;
  unsigned long long size_GB = 4;
  float sample_rate = 0.5f;
  int16_t TOF_division = 1000;
  std::vector<std::string> randomListmodeFile;
  uint32_t timeBegin_ms = 0;
  uint32_t timeEnd_ms = 0xFFFFFFFF;
  std::vector<std::string> listmode_paths;
  core::Image3DOutput<float> h_AttnImg;
  bool doScatter = false;
  int minSectorDiff = 4;
  int randRadialModuleNumS = 6;
  bool doNorm = false;
  std::string normFactorsFile = "";
  bool doSelfNorm = false;
  float *selfNormMich = nullptr;
  bool doAttn = false;
  core::Image3DInput<float> attnMap;
  float tailFittingThreshold = 0.95;
  float scatterPointsThreshold = 0.02;
  core::Vector<double, 3> scatterEnergyWindow = {350.00, 650.00, 0.15};
  core::Vector<double, 3> scatterEffTableEnergy = {0.01, 700.00, 0.01};
  float gauss_hwhm_mm = 1.0f;
  bool bigVoxelScatterSimulation = false;
  double tofSSS_timeBinWidth_ps=100; 
  double tofSSS_timeBinStart_ps=-1500; 
  double tofSSS_timeBinEnd_ps=1500; 
  double tofSSS_systemTimeRes_ns=0.3;
};

void instant_OSEM_listmodeTOF_MULTI_CUDA(core::Image3DOutput<float> out_OSEMImg, OSEM_TOF_MULTI_params params,
                                         core::MichDefine const &mich);
} // namespace openpni::experimental::example
