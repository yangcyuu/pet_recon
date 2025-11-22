#include "include/experimental/node/KGConv3D.hpp"

#include "include/experimental/node/GaussConv3D.hpp"
#include "include/experimental/node/KNNConv3D.hpp"
namespace openpni::experimental::node {
class KGConv3D_impl {
public:
  KGConv3D_impl() {}
  ~KGConv3D_impl() {}

public:
  void setHWHM(
      float hwhm) {
    m_convGauss.setHWHM(hwhm);
  }
  void setHWHM(
      core::Vector<float, 3> hwhm) {
    m_convGauss.setHWHM(hwhm);
  }
  void setKNNNumbers(
      int knnNumbers) {
    m_convKNN.setKNNNumbers(knnNumbers);
  }
  void setFeatureSizeHalf(
      core::Vector<int64_t, 3> featureSizeHalf) {
    m_convKNN.setFeatureSizeHalf(featureSizeHalf);
  }
  void setKNNSearchSizeHalf(
      core::Vector<int64_t, 3> searchSizeHalf) {
    m_convKNN.setKNNSearchSizeHalf(searchSizeHalf);
  }
  void setKNNSigmaG2(
      float sigmaG2) {
    m_convKNN.setKNNSigmaG2(sigmaG2);
  }
  void setHTensorDataIn(
      core::TensorDataInput<float, 3> h_imageIO) {
    m_convKNN.setHTensorDataIn(h_imageIO);
  }
  void setHTensorDataIn(
      float *in_hptr, core::Grids<3> grid) {
    m_convKNN.setHTensorDataIn(in_hptr, grid);
  }
  void setDTensorDataIn(
      core::TensorDataInput<float, 3> d_imageIn) {
    m_convKNN.setDTensorDataIn(d_imageIn);
  }
  void setDTensorDataIn(
      float *in_dptr, core::Grids<3> grid) {
    m_convKNN.setDTensorDataIn(in_dptr, grid);
  }

public:
  void convH(
      core::TensorDataIO<float, 3> __image) {
    m_convGauss.convH(__image);
    m_convKNN.convH(__image);
  }
  void deconvH(
      core::TensorDataIO<float, 3> __image) {
    m_convKNN.deconvH(__image);
    m_convGauss.deconvH(__image);
  }
  void convD(
      core::TensorDataIO<float, 3> __image) {
    m_convGauss.convD(__image);
    m_convKNN.convD(__image);
  }
  void deconvD(
      core::TensorDataIO<float, 3> __image) {
    m_convKNN.deconvD(__image);
    m_convGauss.deconvD(__image);
  }

private:
  node::GaussianConv3D m_convGauss;
  node::KNNConv3D m_convKNN;
};

KGConv3D::KGConv3D()
    : m_impl(std::make_unique<KGConv3D_impl>()) {}
KGConv3D::~KGConv3D() {};
void KGConv3D::setHWHM(
    float hwhm) {
  m_impl->setHWHM(hwhm);
}
void KGConv3D::setHWHM(
    core::Vector<float, 3> hwhm) {
  m_impl->setHWHM(hwhm);
}
void KGConv3D::setKNNNumbers(
    int knnNumbers) {
  m_impl->setKNNNumbers(knnNumbers);
}
void KGConv3D::setFeatureSizeHalf(
    core::Vector<int64_t, 3> featureSizeHalf) {
  m_impl->setFeatureSizeHalf(featureSizeHalf);
}
void KGConv3D::setKNNSearchSizeHalf(
    core::Vector<int64_t, 3> searchSizeHalf) {
  m_impl->setKNNSearchSizeHalf(searchSizeHalf);
}
void KGConv3D::setKNNSigmaG2(
    float sigmaG2) {
  m_impl->setKNNSigmaG2(sigmaG2);
}
void KGConv3D::setHTensorDataIn(
    core::TensorDataInput<float, 3> h_imageIO) {
  m_impl->setHTensorDataIn(h_imageIO);
}
void KGConv3D::setHTensorDataIn(
    float *in_hptr, core::Grids<3> grid) {
  m_impl->setHTensorDataIn(in_hptr, grid);
}
void KGConv3D::setDTensorDataIn(
    core::TensorDataInput<float, 3> d_imageIn) {
  m_impl->setDTensorDataIn(d_imageIn);
}
void KGConv3D::setDTensorDataIn(
    float *in_dptr, core::Grids<3> grid) {
  m_impl->setDTensorDataIn(in_dptr, grid);
}
void KGConv3D::convH(
    core::TensorDataIO<float, 3> __image) {
  m_impl->convH(__image);
}
void KGConv3D::deconvH(
    core::TensorDataIO<float, 3> __image) {
  m_impl->deconvH(__image);
}
void KGConv3D::convD(
    core::TensorDataIO<float, 3> __image) {
  m_impl->convD(__image);
}
void KGConv3D::deconvD(
    core::TensorDataIO<float, 3> __image) {
  m_impl->deconvD(__image);
}

} // namespace  openpni::experimental::node
