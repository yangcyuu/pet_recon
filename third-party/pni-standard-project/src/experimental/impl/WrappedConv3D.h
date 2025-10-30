#pragma once
#include <memory>

#include "include/experimental/algorithms/GaussianConvolutionCUDA.cuh"
namespace openpni::experimental::node::impl {
class WrappedConv3D_impl;
class WrappedConv3D {
public:
  WrappedConv3D(core::Vector<float, 3> hwhm);
  WrappedConv3D(float hwhm);
  WrappedConv3D();
  ~WrappedConv3D();

public:
  void setHWHM(core::Vector<float, 3> hwhm);
  void setHWHM(float hwhm);

public:
  void conv(core::TensorDataIO<float, 3> image) const;
  void deconv(core::TensorDataIO<float, 3> imagem) const;

private:
  std::unique_ptr<WrappedConv3D_impl> m_impl;
};
} // namespace openpni::experimental::node::impl