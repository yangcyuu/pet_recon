#include "include/experimental/node/KNNConv3D.hpp"

#include <cstring>
#include <format>

#include "impl/KNN.h"
#include "include/basic/CudaPtr.hpp"
#include "src/common/Debug.h"
namespace openpni::experimental::node {
class KNNConv3D_impl {
public:
  KNNConv3D_impl() {}
  ~KNNConv3D_impl() {}

public:
  void setKNNNumbers(
      int knnNumbers) {
    m_knnNumbers = knnNumbers;
  }
  void setFeatureSizeHalf(
      core::Vector<int64_t, 3> featureSizeHalf) {
    m_featureSizeHalf = featureSizeHalf;
  }
  void setKNNSearchSizeHalf(
      core::Vector<int64_t, 3> searchSizeHalf) {
    m_knnSearchSizeHalf = searchSizeHalf;
    m_knnNumbers = std::min<int64_t>(m_knnNumbers, (searchSizeHalf * 2 + 1).lmul());
  }
  void setKNNSigmaG2(
      float sigmaG2) {
    m_sigmaG2 = sigmaG2;
  }
  void setHTensorDataIn(
      core::TensorDataInput<float, 3> h_imageIO) {
    mh_imageIn = h_imageIO;
  }
  void setHTensorDataIn(
      float *in_hptr, core::Grids<3> grid) {
    mh_imageIn.ptr = in_hptr;
    mh_imageIn.grid = grid;
  }
  void setDTensorDataIn(
      core::TensorDataInput<float, 3> d_imageIn) {
    md_imageIn = d_imageIn;
  }
  void setDTensorDataIn(
      float *in_dptr, core::Grids<3> grid) {
    md_imageIn.ptr = in_dptr;
    md_imageIn.grid = grid;
  }

public:
  void convH(
      core::TensorDataIO<float, 3> __image) {
    generateHKNNKernel();
    checkOrThrowHGenerated(__image);
    PNI_DEBUG("Doing KNN Conv\n");
    if (__image.ptr_in != __image.ptr_out)
      impl::h_knn_conv(mh_knnValue.get(), mh_knnTo.get(), m_knnNumbers, __image.input_branch().ptr,
                       __image.input_branch().grid.totalSize(), __image.output_branch().ptr);
    else {
      auto pixelNum = __image.input_branch().grid.totalSize();
      if (m_tempImageSize < pixelNum) {
        mh_tempImage = std::make_unique_for_overwrite<float[]>(pixelNum);
        m_tempImageSize = pixelNum;
      }
      impl::h_knn_conv(mh_knnValue.get(), mh_knnTo.get(), m_knnNumbers, __image.input_branch().ptr,
                       __image.input_branch().grid.totalSize(), mh_tempImage.get());
      std::memcpy(__image.output_branch().ptr, mh_tempImage.get(), sizeof(float) * pixelNum);
    }
  }
  void deconvH(
      core::TensorDataIO<float, 3> __image) {
    generateHKNNKernel();
    checkOrThrowHGenerated(__image);
    PNI_DEBUG("Doing KNN Deconv\n");
    if (__image.ptr_in != __image.ptr_out)
      impl::h_knn_deconv(mh_knnValue.get(), mh_knnTo.get(), m_knnNumbers, __image.input_branch().ptr,
                         __image.input_branch().grid.totalSize(), __image.output_branch().ptr);
    else {
      auto pixelNum = __image.input_branch().grid.totalSize();
      if (m_tempImageSize < pixelNum) {
        mh_tempImage = std::make_unique_for_overwrite<float[]>(pixelNum);
        m_tempImageSize = pixelNum;
      }
      impl::h_knn_deconv(mh_knnValue.get(), mh_knnTo.get(), m_knnNumbers, __image.input_branch().ptr,
                         __image.input_branch().grid.totalSize(), mh_tempImage.get());
      std::memcpy(__image.output_branch().ptr, mh_tempImage.get(), sizeof(float) * pixelNum);
    }
  }
  void convD(
      core::TensorDataIO<float, 3> __image) {
    generateDKNNKernel();
    checkOrThrowDGenerated(__image);
    PNI_DEBUG("Doing KNN Conv\n");
    if (__image.ptr_in != __image.ptr_out)
      impl::d_knn_conv(md_knnValue.get(), md_knnTo.get(), m_knnNumbers, __image.input_branch().ptr,
                       __image.input_branch().grid.totalSize(), __image.output_branch().ptr);
    else {
      auto pixelNum = __image.input_branch().grid.totalSize();
      md_tempImage.reserve(pixelNum);
      impl::d_knn_conv(md_knnValue.get(), md_knnTo.get(), m_knnNumbers, __image.input_branch().ptr,
                       __image.input_branch().grid.totalSize(), md_tempImage.data());
      md_tempImage.allocator().copy_from_device_to_device(__image.output_branch().ptr, md_tempImage.cspan());
    }
  }
  void deconvD(
      core::TensorDataIO<float, 3> __image) {
    generateDKNNKernel();
    checkOrThrowDGenerated(__image);
    PNI_DEBUG("Doing KNN Deconv\n");
    if (__image.ptr_in != __image.ptr_out)
      impl::d_knn_deconv(md_knnValue.get(), md_knnTo.get(), m_knnNumbers, __image.input_branch().ptr,
                         __image.input_branch().grid.totalSize(), __image.output_branch().ptr);
    else {
      auto pixelNum = __image.input_branch().grid.totalSize();
      md_tempImage.reserve(pixelNum);
      impl::d_knn_deconv(md_knnValue.get(), md_knnTo.get(), m_knnNumbers, __image.input_branch().ptr,
                         __image.input_branch().grid.totalSize(), md_tempImage.data());
      md_tempImage.allocator().copy_from_device_to_device(__image.output_branch().ptr, md_tempImage.cspan());
    }
  }

private:
  void checkOrThrowHGenerated(
      core::TensorDataIO<float, 3> __image) {
    if (!m_HKNNKernelGenerated)
      throw exceptions::algorithm_unexpected_condition("No h_KNN kernel generated.");

    auto pixelNum = __image.input_branch().grid.totalSize();

    auto paddedImageSpan = core::MDSpan<3>::create(__image.input_branch().grid.size.dimSize + m_featureSizeHalf * 2);
    if (mh_tempPaddedImageSize != paddedImageSpan.totalSize()) {
#if PNI_STANDARD_CONFIG_ENABLE_CMDLINE_ERROR_MESSAGES
      debug::say_critical_message(std::format("KNNConv3D Error: Expected padded size: {}, but got {}.\n",
                                              mh_tempPaddedImageSize, paddedImageSpan.totalSize()));
#endif
      throw exceptions::algorithm_unexpected_condition("paddedImageSize doesn't match.");
    }

    auto featureSpan = core::MDBeginEndSpan<3>::create(-m_featureSizeHalf, m_featureSizeHalf + 1);
    if (mh_tempFeatureMatrixSize != pixelNum * featureSpan.totalSize())
      throw exceptions::algorithm_unexpected_condition("Feature matrixSize doesn't match.");

    if (mh_knnKernelSize != pixelNum * m_knnNumbers)
      throw exceptions::algorithm_unexpected_condition("KNN kernelSize doesn't match.");
  }
  void checkOrThrowDGenerated(
      core::TensorDataIO<float, 3> __image) {
    if (!m_DKNNKernelGenerated)
      throw exceptions::algorithm_unexpected_condition("No d_KNN kernel generated.");

    auto pixelNum = __image.input_branch().grid.totalSize();
    auto paddedImageSpan = core::MDSpan<3>::create(__image.input_branch().grid.size.dimSize + m_featureSizeHalf * 2);
    if (md_tempPaddedImageSize != paddedImageSpan.totalSize()) {
#if PNI_STANDARD_CONFIG_ENABLE_CMDLINE_ERROR_MESSAGES
      debug::say_critical_message(std::format("KNNConv3D Error: Expected padded size: {}, but got {}.\n",
                                              md_tempPaddedImageSize, paddedImageSpan.totalSize()));
#endif
      throw exceptions::algorithm_unexpected_condition("paddedImageSize doesn't match.");
    }

    auto featureSpan = core::MDBeginEndSpan<3>::create(-m_featureSizeHalf, m_featureSizeHalf + 1);
    if (md_tempFeatureMatrixSize != pixelNum * featureSpan.totalSize())
      throw exceptions::algorithm_unexpected_condition("Feature matrixSize doesn't match.");

    if (md_knnKernelSize != pixelNum * m_knnNumbers)
      throw exceptions::algorithm_unexpected_condition("KNN kernelSize doesn't match.");
  }

private:
  void generateHKNNKernel() {
    if (m_HKNNKernelGenerated)
      return;
    PNI_DEBUG("Generate KNN Kernel\n");
    // 第一步：增广
    auto featureSpan = core::MDBeginEndSpan<3>::create(-m_featureSizeHalf, m_featureSizeHalf + 1);
    auto imageInPaddedSpan =
        core::MDBeginEndSpan<3>::create(m_featureSizeHalf, m_featureSizeHalf + mh_imageIn.grid.size.dimSize);
    auto inSpan = mh_imageIn.grid.index_span();
    auto paddedImageSpan = core::MDSpan<3>::create(inSpan.dimSize + m_featureSizeHalf * 2);
    std::unique_ptr<float[]> tempPaddedImage;
    std::unique_ptr<float[]> tempFeatureMatrix;
    if (mh_tempPaddedImageSize != paddedImageSpan.totalSize()) {
      tempPaddedImage = std::make_unique_for_overwrite<float[]>(paddedImageSpan.totalSize());
      mh_tempPaddedImageSize = paddedImageSpan.totalSize();
    }
    impl::h_image_augmentation(mh_imageIn.ptr, inSpan, tempPaddedImage.get(), m_featureSizeHalf);

    // 第二步：计算特征矩阵
    PNI_DEBUG("  Fill Feature Matrix\n");
    if (mh_tempFeatureMatrixSize != inSpan.totalSize() * featureSpan.totalSize()) {
      tempFeatureMatrix = std::make_unique_for_overwrite<float[]>(inSpan.totalSize() * featureSpan.totalSize());
      mh_tempFeatureMatrixSize = inSpan.totalSize() * featureSpan.totalSize();
    }
    impl::h_fill_feature_matrix(tempPaddedImage.get(), tempFeatureMatrix.get(), featureSpan, imageInPaddedSpan,
                                paddedImageSpan);

    // 第三步：归一化
    PNI_DEBUG("  Feature Matrix Normalize\n");
    auto featureMatrixSpan = core::MDSpan<6>::create(inSpan.dimSize.left_expand(featureSpan.size()));
    impl::h_feature_matrix_normalize(tempFeatureMatrix.get(), featureMatrixSpan);

    // 第四步：计算KNN并生成卷积核
    PNI_DEBUG("  Fill KNN Indices\n");
    if (mh_knnKernelSize != inSpan.totalSize() * m_knnNumbers) {
      mh_knnValue = std::make_unique_for_overwrite<float[]>(inSpan.totalSize() * m_knnNumbers);
      mh_knnTo = std::make_unique_for_overwrite<std::size_t[]>(inSpan.totalSize() * m_knnNumbers);
      mh_knnKernelSize = inSpan.totalSize() * m_knnNumbers;
    }
    auto kNNSearchSpan = core::MDBeginEndSpan<3>::create(-m_knnSearchSizeHalf, m_knnSearchSizeHalf + 1);
    impl::h_fill_KNN_indices(tempFeatureMatrix.get(), featureMatrixSpan, kNNSearchSpan, m_knnNumbers, mh_knnValue.get(),
                             mh_knnTo.get(), m_sigmaG2);
    m_HKNNKernelGenerated = true;
  }
  void generateDKNNKernel() {
    if (m_DKNNKernelGenerated)
      return;
    PNI_DEBUG("Generate KNN Kernel\n");
    // 第一步：增广
    auto featureSpan = core::MDBeginEndSpan<3>::create(-m_featureSizeHalf, m_featureSizeHalf + 1);
    auto imageInPaddedSpan =
        core::MDBeginEndSpan<3>::create(m_featureSizeHalf, m_featureSizeHalf + md_imageIn.grid.size.dimSize);
    auto inSpan = md_imageIn.grid.index_span();
    auto paddedImageSpan = core::MDSpan<3>::create(inSpan.dimSize + m_featureSizeHalf * 2);
    cuda_sync_ptr<float> tempPaddedImage{"KNN_TempPaddedImage"};
    cuda_sync_ptr<float> tempFeatureMatrix{"KNN_TempFeatureMatrix"};
    if (md_tempPaddedImageSize != paddedImageSpan.totalSize()) {
      tempPaddedImage.reserve(paddedImageSpan.totalSize());
      md_tempPaddedImageSize = paddedImageSpan.totalSize();
    }
    impl::d_image_augmentation(md_imageIn.ptr, inSpan, tempPaddedImage.get(), m_featureSizeHalf);

    // 第二步：计算特征矩阵
    PNI_DEBUG("  Fill Feature Matrix\n");
    if (md_tempFeatureMatrixSize != inSpan.totalSize() * featureSpan.totalSize()) {
      tempFeatureMatrix.reserve(inSpan.totalSize() * featureSpan.totalSize());
      md_tempFeatureMatrixSize = inSpan.totalSize() * featureSpan.totalSize();
    }
    impl::d_fill_feature_matrix(tempPaddedImage.get(), tempFeatureMatrix.get(), featureSpan, imageInPaddedSpan,
                                paddedImageSpan);

    // 第三步：归一化
    PNI_DEBUG("  Feature Matrix Normalize\n");
    auto featureMatrixSpan = core::MDSpan<6>::create(inSpan.dimSize.left_expand(featureSpan.size()));
    impl::d_feature_matrix_normalize(tempFeatureMatrix.get(), featureMatrixSpan);

    // 第四步：计算KNN并生成卷积核
    PNI_DEBUG("  Fill KNN Indices\n");
    if (md_knnKernelSize != inSpan.totalSize() * m_knnNumbers) {
      md_knnValue.reserve(inSpan.totalSize() * m_knnNumbers);
      md_knnTo.reserve(inSpan.totalSize() * m_knnNumbers);
      md_knnKernelSize = inSpan.totalSize() * m_knnNumbers;
    }
    auto kNNSearchSpan = core::MDBeginEndSpan<3>::create(-m_knnSearchSizeHalf, m_knnSearchSizeHalf + 1);
    impl::d_fill_KNN_indices(tempFeatureMatrix.get(), featureMatrixSpan, kNNSearchSpan, m_knnNumbers, md_knnValue.get(),
                             md_knnTo.get(), m_sigmaG2);
    m_DKNNKernelGenerated = true;
  }

private:
  core::TensorDataInput<float, 3> mh_imageIn;
  core::TensorDataInput<float, 3> md_imageIn;
  bool m_HKNNKernelGenerated = false;
  bool m_DKNNKernelGenerated = false;

  std::unique_ptr<float[]> mh_knnValue;
  std::unique_ptr<std::size_t[]> mh_knnTo;
  std::size_t mh_knnKernelSize = 0;
  std::size_t mh_tempPaddedImageSize = 0;
  std::size_t mh_tempFeatureMatrixSize = 0;

  std::unique_ptr<float[]> mh_tempImage;
  std::size_t m_tempImageSize = 0;

  cuda_sync_ptr<float> md_knnValue{"KNN_KernelValue"};
  cuda_sync_ptr<std::size_t> md_knnTo{"KNN_KernelTo"};
  std::size_t md_knnKernelSize = 0;
  std::size_t md_tempPaddedImageSize = 0;
  std::size_t md_tempFeatureMatrixSize = 0;

  cuda_sync_ptr<float> md_tempImage;

  int m_knnNumbers = 4;
  core::Vector<int64_t, 3> m_featureSizeHalf = core::Vector<int64_t, 3>{1, 1, 1};
  core::Vector<int64_t, 3> m_knnSearchSizeHalf = core::Vector<int64_t, 3>{1, 1, 1};
  float m_sigmaG2 = 1.0f;
};

KNNConv3D::KNNConv3D()
    : m_impl(std::make_unique<KNNConv3D_impl>()) {}
KNNConv3D::~KNNConv3D() = default;
void KNNConv3D::setKNNNumbers(
    int knnNumbers) {
  m_impl->setKNNNumbers(knnNumbers);
}
void KNNConv3D::setFeatureSizeHalf(
    core::Vector<int64_t, 3> featureSizeHalf) {
  m_impl->setFeatureSizeHalf(featureSizeHalf);
}
void KNNConv3D::setKNNSearchSizeHalf(
    core::Vector<int64_t, 3> searchSizeHalf) {
  m_impl->setKNNSearchSizeHalf(searchSizeHalf);
}
void KNNConv3D::setKNNSigmaG2(
    float sigmaG2) {
  m_impl->setKNNSigmaG2(sigmaG2);
}
void KNNConv3D::setHTensorDataIn(
    core::TensorDataInput<float, 3> h_imageIO) {
  m_impl->setHTensorDataIn(h_imageIO);
}
void KNNConv3D::setHTensorDataIn(
    float *in_hptr, core::Grids<3> grid) {
  m_impl->setHTensorDataIn(in_hptr, grid);
}
void KNNConv3D::setDTensorDataIn(
    core::TensorDataInput<float, 3> d_imageIn) {
  m_impl->setDTensorDataIn(d_imageIn);
}
void KNNConv3D::setDTensorDataIn(
    float *in_dptr, core::Grids<3> grid) {
  m_impl->setDTensorDataIn(in_dptr, grid);
}
void KNNConv3D::convH(
    core::TensorDataIO<float, 3> __image) {
  m_impl->convH(__image);
}
void KNNConv3D::deconvH(
    core::TensorDataIO<float, 3> __image) {
  m_impl->deconvH(__image);
}
void KNNConv3D::convD(
    core::TensorDataIO<float, 3> __image) {
  m_impl->convD(__image);
}
void KNNConv3D::deconvD(
    core::TensorDataIO<float, 3> __image) {
  m_impl->deconvD(__image);
}

} // namespace openpni::experimental::node