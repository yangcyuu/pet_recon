#pragma once
#include <memory>

#include "../interface/Conv3D.hpp"
namespace openpni::experimental::node {
class GaussianConv3D_impl;
class GaussianConv3D : public interface::Conv3D {
public:
  GaussianConv3D();
  virtual ~GaussianConv3D();

public:
  void setHWHM(float hwhm);
  void setHWHM(core::Vector<float, 3> hwhm);

public:
  void convH(core::TensorDataIO<float, 3> __image) override;
  void deconvH(core::TensorDataIO<float, 3> __image) override;
  void convD(core::TensorDataIO<float, 3> __image) override;
  void deconvD(core::TensorDataIO<float, 3> __image) override;

private:
  std::unique_ptr<GaussianConv3D_impl> m_impl;
};
} // namespace openpni::experimental::node
