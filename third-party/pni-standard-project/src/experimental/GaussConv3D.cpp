#include "include/experimental/node/GaussConv3D.hpp"

#include "impl/WrappedConv3D.h"
namespace openpni::experimental::node {
class GaussianConv3D_impl {
public:
  GaussianConv3D_impl() {
    m_conv_cpu.setHWHM(1.0f);
    m_conv_gpu.setHWHM(1.0f);
  }
  ~GaussianConv3D_impl() {}

public:
  void setHWHM(
      float hwhm) {
    m_conv_cpu.setHWHM(hwhm);
    m_conv_gpu.setHWHM(hwhm);
  }
  void setHWHM(
      core::Vector<float, 3> hwhm) {
    m_conv_cpu.setHWHM(hwhm);
    m_conv_gpu.setHWHM(hwhm);
  }

public:
  void convH(
      core::TensorDataIO<float, 3> __image) {
    m_conv_cpu.conv(__image);
  }
  void deconvH(
      core::TensorDataIO<float, 3> __image) {
    m_conv_cpu.deconv(__image);
  }
  void convD(
      core::TensorDataIO<float, 3> __image) {
    m_conv_gpu.conv(__image);
  }
  void deconvD(
      core::TensorDataIO<float, 3> __image) {
    m_conv_gpu.deconv(__image);
  }

private:
  algorithms::GaussianConvolutionCPU<float, 3> m_conv_cpu;
  impl::WrappedConv3D m_conv_gpu;
};

GaussianConv3D::GaussianConv3D()
    : m_impl(std::make_unique<GaussianConv3D_impl>()) {}
GaussianConv3D::~GaussianConv3D() {}
void GaussianConv3D::convH(
    core::TensorDataIO<float, 3> __image) {
  m_impl->convH(__image);
}
void GaussianConv3D::deconvH(
    core::TensorDataIO<float, 3> __image) {
  m_impl->deconvH(__image);
}
void GaussianConv3D::convD(
    core::TensorDataIO<float, 3> __image) {
  m_impl->convD(__image);
}
void GaussianConv3D::deconvD(
    core::TensorDataIO<float, 3> __image) {
  m_impl->deconvD(__image);
}
void GaussianConv3D::setHWHM(
    float hwhm) {
  m_impl->setHWHM(hwhm);
}
void GaussianConv3D::setHWHM(
    core::Vector<float, 3> hwhm) {
  m_impl->setHWHM(hwhm);
}

} // namespace openpni::experimental::node
