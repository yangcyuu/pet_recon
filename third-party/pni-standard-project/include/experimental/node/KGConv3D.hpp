#pragma once
#include <memory>

#include "../interface/Conv3D.hpp"
namespace openpni::experimental::node {
class KGConv3D_impl;
class KGConv3D : public interface::Conv3D {
public:
  KGConv3D();
  virtual ~KGConv3D();

public:
  void setHWHM(float hwhm);
  void setHWHM(core::Vector<float, 3> hwhm);
  void setKNNNumbers(int knnNumbers);
  void setFeatureSizeHalf(core::Vector<int64_t, 3> featureSizeHalf);
  void setKNNSearchSizeHalf(core::Vector<int64_t, 3> searchSizeHalf);
  void setKNNSigmaG2(float sigmaG2);

public:
  void convH(core::TensorDataIO<float, 3> __image) override;
  void deconvH(core::TensorDataIO<float, 3> __image) override;
  void convD(core::TensorDataIO<float, 3> __image) override;
  void deconvD(core::TensorDataIO<float, 3> __image) override;

private:
  std::unique_ptr<KGConv3D_impl> m_impl;
};
} // namespace openpni::experimental::node
