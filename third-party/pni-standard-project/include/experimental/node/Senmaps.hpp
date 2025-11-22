#pragma once
#include <memory>

#include "../../basic/CudaPtr.hpp"
#include "../../basic/Exceptions.hpp"
#include "../core/Mich.hpp"
#include "../interface/Conv3D.hpp"
#include "MichAttn.hpp"
namespace openpni::experimental::node {
class MichNormalization;
class MichSenmap_impl;
class MichSenmap {
  using Grids3f = core::Grids<3, float>;

public:
  enum SenmapMode { Mode_mich, Mode_listmode };
  enum SenmapSource { Senmap_CPU, Senmap_GPU };

public:
  MichSenmap(interface::Conv3D &__conv3D, const core::MichDefine &__mich);
  MichSenmap(std::unique_ptr<MichSenmap_impl> &&impl);
  ~MichSenmap();

public:
  void setSubsetNum(int num);
  void setMaxBufferedImages(int num);
  void setPreferredSource(SenmapSource source);
  void setMode(SenmapMode mode);
  void preBaking(std::vector<std::pair<Grids3f, int>> const &gridsAndSubsetList);
  void clearCache();
  void bindNormalization(MichNormalization *normalization);
  void bindAttenuation(MichAttn *attn);
  std::unique_ptr<float[]> dumpHSenmap(int subsetIndex, core::Grids<3, float> grids);
  cuda_sync_ptr<float> dumpDSenmap(int subsetIndex);
  void updateHImage(float *h_updateImage, float *h_out, int subsetIndex, core::Grids<3, float> grids);
  void updateDImage(float *d_updateImage, float *d_out, int subsetIndex, core::Grids<3, float> grids);
  float lastUpdateMeasurement();

private:
  std::unique_ptr<MichSenmap_impl> m_impl;
};
} // namespace openpni::experimental::node
