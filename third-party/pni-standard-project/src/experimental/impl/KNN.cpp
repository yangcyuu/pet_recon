#include "KNN.h"

#include "include/experimental/algorithms/Sort.hpp"
#include "include/experimental/tools/Parallel.hpp"
namespace openpni::experimental::node::impl {
void h_image_augmentation(
    float const *__inImage, core::MDSpan<3> __inSpan, float *__outImage, core::Vector<int64_t, 3> __padding) {
  auto outSpan = core::MDSpan<3>::create(__inSpan.dimSize + __padding * 2);
  tools::parallel_for_each(outSpan, [&](decltype(outSpan)::index_type index) {
    auto inIdx = index - __padding;
    if (__inSpan.inBounds(inIdx))
      __outImage[outSpan[index]] = __inImage[__inSpan[inIdx]];
    else
      __outImage[outSpan[index]] = 0.f;
  });
}
void h_fill_feature_matrix(
    float const *__paddedImage, float *__featureMatrix, core::MDBeginEndSpan<3> __featureSpan,
    core::MDBeginEndSpan<3> __inImageInPaddedSpan, core::MDSpan<3> __paddedImageSpan) {
  auto featureMatrixSpan =
      core::MDBeginEndSpan<6>::create(__inImageInPaddedSpan.begins.left_expand(__featureSpan.begins),
                                      __inImageInPaddedSpan.ends.left_expand(__featureSpan.ends));
  tools::parallel_for_each(featureMatrixSpan, [&](decltype(featureMatrixSpan)::index_type index) {
    auto index_in_padded_image = index.left_shrink<3>();
    auto index_in_feature = index.right_shrink<3>();
    auto feature_in_padded_image = index_in_padded_image + index_in_feature;
    __featureMatrix[featureMatrixSpan[index]] = __paddedImage[__paddedImageSpan[feature_in_padded_image]];
  });
}
void h_feature_matrix_normalize(
    float *__featureMatrix, core::MDSpan<6> __featureMatrixSpan) {
  auto imageSpan = __featureMatrixSpan.left_shrink<3>();
  auto featureSpan = __featureMatrixSpan.right_shrink<3>();

  tools::parallel_for_each(featureSpan, [&](decltype(imageSpan)::index_type feature_index) {
    const auto total_image_pixels = imageSpan.totalSize();
    float featureMean{0.f}, stddev{0.f};
    for (const auto imageIdx : imageSpan)
      featureMean += __featureMatrix[__featureMatrixSpan[imageIdx.left_expand(feature_index)]];
    featureMean /= total_image_pixels;
    for (const auto imageIdx : imageSpan)
      stddev += core::FMath<float>::fpow(
          __featureMatrix[__featureMatrixSpan[imageIdx.left_expand(feature_index)]] - featureMean, 2);
    stddev = core::FMath<float>::fsqrt(stddev / total_image_pixels);
    for (const auto imageIdx : imageSpan)
      __featureMatrix[__featureMatrixSpan[imageIdx.left_expand(feature_index)]] /= stddev;
  });
}
void h_fill_KNN_indices(
    float const *__featureMatrix, core::MDSpan<6> __featureMatrixSpan, core::MDBeginEndSpan<3> __searchSpan,
    int __knnNumbers, float *__outKNNKernel, std::size_t *__outKNNTo, float __sigmaG2) {
  auto imageSpan = __featureMatrixSpan.left_shrink<3>();
  auto featureSpan = __featureMatrixSpan.right_shrink<3>();

  auto kNNSpan = core::MDSpan<4>::create(imageSpan.dimSize.left_expand(__knnNumbers));
  tools::parallel_for_each(imageSpan, [&](decltype(imageSpan)::index_type image_index) {
    std::vector<std::pair<float, decltype(imageSpan)::index_type>> knn_distances;
    for (const auto searchIdx : __searchSpan) {
      auto search_in_image = image_index + searchIdx;
      if (!imageSpan.inBounds(search_in_image))
        continue;
      float distance = 0.f;
      for (const auto feature_idx : featureSpan) {
        distance +=
            core::FMath<float>::fpow(__featureMatrix[__featureMatrixSpan[image_index.left_expand(feature_idx)]] -
                                         __featureMatrix[__featureMatrixSpan[search_in_image.left_expand(feature_idx)]],
                                     2);
      }
      knn_distances.emplace_back(distance, search_in_image);
    }
    std::partial_sort(
        knn_distances.data(), knn_distances.data() + std::min(__knnNumbers, static_cast<int>(knn_distances.size())),
        knn_distances.data() + knn_distances.size(), [](const auto &a, const auto &b) { return a.first < b.first; });
    float sumWeight = 0.f;
    for (int i = 0; i < __knnNumbers && i < knn_distances.size(); i++) {
      auto weight = core::FMath<float>::fexp(-knn_distances[i].first / (2.0f * __sigmaG2));
      sumWeight += weight;
      __outKNNKernel[kNNSpan[image_index.left_expand(i)]] = weight;
      __outKNNTo[kNNSpan[image_index.left_expand(i)]] = imageSpan[knn_distances[i].second];
    }
    for (int i = 0; i < __knnNumbers && i < knn_distances.size(); i++)
      __outKNNKernel[kNNSpan[image_index.left_expand(i)]] /= sumWeight;
  });
}
void h_knn_conv(
    float const *__knnValue, std::size_t const *__knnTo, int __knnNumbers, float const *__inImage,
    std::size_t __imageSize, float *__outImage) {
  tools::parallel_for_each(__imageSize, [&](std::size_t index) {
    float result = 0.f;
    for (int i = 0; i < __knnNumbers; i++) {
      result += __knnValue[index * __knnNumbers + i] * __inImage[__knnTo[index * __knnNumbers + i]];
    }
    __outImage[index] = result;
  });
}
void h_knn_deconv(
    float const *__knnValue, std::size_t const *__knnTo, int __knnNumbers, float const *__inImage,
    std::size_t __imageSize, float *__outImage) {
  std::fill_n(__outImage, __imageSize, 0.f);
  tools::parallel_for_each(__imageSize, [&](std::size_t index) {
    for (int i = 0; i < __knnNumbers; i++) {
      auto toIndex = __knnTo[index * __knnNumbers + i];
      auto weight = __knnValue[index * __knnNumbers + i];
      std::atomic_ref<float>(__outImage[toIndex]) += weight * __inImage[index];
    }
  });
}
} // namespace openpni::experimental::node::impl
