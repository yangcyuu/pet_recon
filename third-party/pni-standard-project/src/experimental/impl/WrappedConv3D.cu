#include "WrappedConv3D.h"
namespace openpni::experimental::node::impl {
class WrappedConv3D_impl {
public:
  WrappedConv3D_impl(
      core::Vector<float, 3> hwhm)
      : m_conv(hwhm) {}
  ~WrappedConv3D_impl() = default;

public:
  void setHWHM(
      core::Vector<float, 3> hwhm) {
    m_conv.setHWHM(hwhm);
  }
  void setHWHM(
      float hwhm) {
    m_conv.setHWHM(hwhm);
  }

public:
  void conv(
      core::TensorDataIO<float, 3> image) const {
    m_conv.conv(image);
  }
  void deconv(
      core::TensorDataIO<float, 3> image) const {
    m_conv.deconv(image);
  }

private:
  algorithms::GaussianConvolutionCUDA<float, 3> m_conv;
};

WrappedConv3D::WrappedConv3D(
    core::Vector<float, 3> hwhm)
    : m_impl(std::make_unique<WrappedConv3D_impl>(hwhm)) {}
WrappedConv3D::WrappedConv3D(
    float hwhm)
    : WrappedConv3D(core::Vector<float, 3>::create(hwhm)) {}
WrappedConv3D::WrappedConv3D()
    : WrappedConv3D(1.0f) {}
WrappedConv3D::~WrappedConv3D() {};
void WrappedConv3D::conv(
    core::TensorDataIO<float, 3> image) const {
  m_impl->conv(image);
}
void WrappedConv3D::deconv(
    core::TensorDataIO<float, 3> image) const {
  m_impl->deconv(image);
}
void WrappedConv3D::setHWHM(
    core::Vector<float, 3> hwhm) {
  m_impl->setHWHM(hwhm);
}
void WrappedConv3D::setHWHM(
    float hwhm) {
  m_impl->setHWHM(hwhm);
}

} // namespace openpni::experimental::node::impl