#pragma once
#include "include/experimental/core/Image.hpp"
namespace openpni::experimental::node::impl {
void h_image_augmentation(float const *__inImage, core::MDSpan<3> __inSpan, float *__outImage,
                          core::Vector<int64_t, 3> __padding);
void h_fill_feature_matrix(float const *__paddedImage, float *__featureMatrix, core::MDBeginEndSpan<3> __featureSpan,
                           core::MDBeginEndSpan<3> __inImageInPaddedSpan, core::MDSpan<3> __paddedImageSpan);
void h_feature_matrix_normalize(float *__featureMatrix, core::MDSpan<6> __featureMatrixSpan);
void h_fill_KNN_indices(float const *__featureMatrix, core::MDSpan<6> __featureMatrixSpan,
                        core::MDBeginEndSpan<3> __searchSpan, int __knnNumbers, float *__outKNNKernel,
                        std::size_t *__outKNNTo, float __sigmaG2);
void h_knn_conv(float const *__knnValue, std::size_t const *__knnTo, int __knnNumbers, float const *__inImage,
                std::size_t __imageSize, float *__outImage);
void h_knn_deconv(float const *__knnValue, std::size_t const *__knnTo, int __knnNumbers, float const *__inImage,
                  std::size_t __imageSize, float *__outImage);
} // namespace openpni::experimental::node::impl

namespace openpni::experimental::node::impl {
void d_image_augmentation(float const *__inImage, core::MDSpan<3> __inSpan, float *__outImage,
                          core::Vector<int64_t, 3> __padding);
void d_fill_feature_matrix(float const *__paddedImage, float *__featureMatrix, core::MDBeginEndSpan<3> __featureSpan,
                           core::MDBeginEndSpan<3> __inImageInPaddedSpan, core::MDSpan<3> __paddedImageSpan);
void d_feature_matrix_normalize(float *__featureMatrix, core::MDSpan<6> __featureMatrixSpan);
void d_fill_KNN_indices(float const *__featureMatrix, core::MDSpan<6> __featureMatrixSpan,
                        core::MDBeginEndSpan<3> __searchSpan, int __knnNumbers, float *__outKNNKernel,
                        std::size_t *__outKNNTo, float __sigmaG2);
void d_knn_conv(float const *__knnValue, std::size_t const *__knnTo, int __knnNumbers, float const *__inImage,
                std::size_t __imageSize, float *__outImage);
void d_knn_deconv(float const *__knnValue, std::size_t const *__knnTo, int __knnNumbers, float const *__inImage,
                  std::size_t __imageSize, float *__outImage);
} // namespace openpni::experimental::node::impl