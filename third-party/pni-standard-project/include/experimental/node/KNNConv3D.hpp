#pragma once
#include <memory>

#include "../interface/Conv3D.hpp"
namespace openpni::experimental::node {
class KNNConv3D_impl;
class KNNConv3D : public interface::Conv3D {
public:
  KNNConv3D();
  virtual ~KNNConv3D();

public:
  void setKNNNumbers(int knnNumbers);
  void setFeatureSizeHalf(core::Vector<int64_t, 3> featureSizeHalf);
  void setKNNSearchSizeHalf(core::Vector<int64_t, 3> searchSizeHalf);
  void setKNNSigmaG2(float sigmaG2);

  void setHTensorDataIn(core::TensorDataInput<float, 3> h_imageIO);
  void setHTensorDataIn(float *in_hptr, core::Grids<3> grid);
  void setDTensorDataIn(core::TensorDataInput<float, 3> d_imageIn);
  void setDTensorDataIn(float *in_dptr, core::Grids<3> grid);

public:
  void convH(core::TensorDataIO<float, 3> __image) override;
  void deconvH(core::TensorDataIO<float, 3> __image) override;
  void convD(core::TensorDataIO<float, 3> __image) override;
  void deconvD(core::TensorDataIO<float, 3> __image) override;

private:
  std::unique_ptr<KNNConv3D_impl> m_impl;
};
} // namespace openpni::experimental::node