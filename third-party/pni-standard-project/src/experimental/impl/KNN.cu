#include <cub/cub.cuh>
#include <thrust/transform_reduce.h>

#include "KNN.h"
#include "include/experimental/algorithms/Sort.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/tools/Loop.hpp"
#include "include/experimental/tools/Parallel.cuh"
namespace openpni::experimental::node::impl {
void d_image_augmentation(
    float const *__inImage, core::MDSpan<3> __inSpan, float *__outImage, core::Vector<int64_t, 3> __padding) {
  auto outSpan = core::MDSpan<3>::create(__inSpan.dimSize + __padding * 2);
  tools::parallel_for_each_CUDA(outSpan.totalSize(), [=] __device__(std::size_t index) {
    auto out_index = outSpan.toIndex(index);
    auto in_index = out_index - __padding;
    if (__inSpan.inBounds(in_index)) {
      __outImage[index] = __inImage[__inSpan[in_index]];
    } else {
      __outImage[index] = 0.f;
    }
  });
}
void d_fill_feature_matrix(
    float const *__paddedImage, float *__featureMatrix, core::MDBeginEndSpan<3> __featureSpan,
    core::MDBeginEndSpan<3> __inImageInPaddedSpan, core::MDSpan<3> __paddedImageSpan) {
  auto featureMatrixSpan =
      core::MDBeginEndSpan<6>::create(__inImageInPaddedSpan.begins.left_expand(__featureSpan.begins),
                                      __inImageInPaddedSpan.ends.left_expand(__featureSpan.ends));
  tools::parallel_for_each_CUDA(featureMatrixSpan, [=] __device__(decltype(featureMatrixSpan)::index_type index) {
    auto index_in_padded_image = index.left_shrink<3>();
    auto index_in_feature = index.right_shrink<3>();
    auto feature_in_padded_image = index_in_padded_image + index_in_feature;
    __featureMatrix[featureMatrixSpan[index]] = __paddedImage[__paddedImageSpan[feature_in_padded_image]];
  });
}
void d_feature_matrix_normalize(
    float *__featureMatrix, core::MDSpan<6> __featureMatrixSpan) {
  auto imageSpan = __featureMatrixSpan.left_shrink<3>();
  auto featureSpan = __featureMatrixSpan.right_shrink<3>();

  int count = 0;
  for (const auto feature_index : featureSpan) {
    auto featureMean = thrust::transform_reduce(
        thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::counting_iterator<std::size_t>(0),
        thrust::counting_iterator<std::size_t>(imageSpan.totalSize()),
        [=] __device__(std::size_t idx) -> float {
          auto image_index = imageSpan.toIndex(idx);
          return __featureMatrix[__featureMatrixSpan[image_index.left_expand(feature_index)]];
        },
        0.f, thrust::plus<float>());
    featureMean /= imageSpan.totalSize();
    auto stddev = thrust::transform_reduce(
        thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::counting_iterator<std::size_t>(0),
        thrust::counting_iterator<std::size_t>(imageSpan.totalSize()),
        [=] __device__(std::size_t idx) -> float {
          auto image_index = imageSpan.toIndex(idx);
          auto val = __featureMatrix[__featureMatrixSpan[image_index.left_expand(feature_index)]];
          return core::FMath<float>::fpow(val - featureMean, 2);
        },
        0.f, thrust::plus<float>());
    stddev = core::FMath<float>::fsqrt(stddev / imageSpan.totalSize());
    tools::parallel_for_each_CUDA(imageSpan.totalSize(), [=] __device__(std::size_t idx) {
      auto image_index = imageSpan.toIndex(idx);
      __featureMatrix[__featureMatrixSpan[image_index.left_expand(feature_index)]] /= stddev;
    });
  }
}
struct KNNItem {
  std::size_t to;
  float distance;
};
void d_fill_KNN_indices(
    float const *__featureMatrix, core::MDSpan<6> __featureMatrixSpan, core::MDBeginEndSpan<3> __searchSpan,
    int __knnNumbers, float *__outKNNKernel, std::size_t *__outKNNTo, float __sigmaG2) {
  auto imageSpan = __featureMatrixSpan.left_shrink<3>();
  auto featureSpan = __featureMatrixSpan.right_shrink<3>();

  constexpr std::size_t max_buffer_size = 1024ull * 1024ull * 128ull;
  const std::size_t batchSize = max_buffer_size / (sizeof(KNNItem) * __knnNumbers);
  cuda_sync_ptr<KNNItem> knn_buffer = make_cuda_sync_ptr<KNNItem>(batchSize * __knnNumbers, "KNN Buffer");

  auto kNNSpan = core::MDSpan<4>::create(imageSpan.dimSize.left_expand(__knnNumbers));
  for (const auto [begin, end] : tools::chunked_ranges_generator.by_max_size(0, imageSpan.totalSize(), batchSize)) {
    tools::parallel_for_each_CUDA(begin, end, [=, knn_buffer = knn_buffer.data()] __device__(std::size_t idx) {
      auto image_index = imageSpan.toIndex(idx);
      auto *knn_distances = &knn_buffer[(idx - begin) * __knnNumbers];
      for (int i = 0; i < __knnNumbers; i++) {
        knn_distances[i].to = 0;
        knn_distances[i].distance = 1e10f;
      }

      for (const auto searchIdx : __searchSpan) {
        auto search_in_image = image_index + searchIdx;
        if (!imageSpan.inBounds(search_in_image))
          continue;
        float distance = 0.f;
        for (const auto feature_idx : featureSpan) {
          distance += core::FMath<float>::fpow(
              __featureMatrix[__featureMatrixSpan[image_index.left_expand(feature_idx)]] -
                  __featureMatrix[__featureMatrixSpan[search_in_image.left_expand(feature_idx)]],
              2);
        }
        algorithms::min_heap_insert_replace(
            knn_distances, knn_distances + __knnNumbers, KNNItem{std::size_t(imageSpan[search_in_image]), distance},
            [] __device__(const KNNItem &a, const KNNItem &b) { return a.distance < b.distance; });
      }

      float sumWeight = 0.f;
      for (int i = 0; i < __knnNumbers; i++) {
        auto weight = core::FMath<float>::fexp(-knn_distances[i].distance / (2.0f * __sigmaG2));
        sumWeight += weight;
        __outKNNKernel[kNNSpan[image_index.left_expand(i)]] = weight;
        __outKNNTo[kNNSpan[image_index.left_expand(i)]] = knn_distances[i].to;
      }
      for (int i = 0; i < __knnNumbers; i++)
        __outKNNKernel[kNNSpan[image_index.left_expand(i)]] /= sumWeight;
    });
  }
}
void d_knn_conv(
    float const *__knnValue, std::size_t const *__knnTo, int __knnNumbers, float const *__inImage,
    std::size_t __imageSize, float *__outImage) {
  tools::parallel_for_each_CUDA(__imageSize, [=] __device__(std::size_t index) {
    float result = 0.f;
    for (int i = 0; i < __knnNumbers; i++) {
      result += __knnValue[index * __knnNumbers + i] * __inImage[__knnTo[index * __knnNumbers + i]];
    }
    __outImage[index] = result;
  });
}
void d_knn_deconv(
    float const *__knnValue, std::size_t const *__knnTo, int __knnNumbers, float const *__inImage,
    std::size_t __imageSize, float *__outImage) {
  example::d_parralel_fill(__outImage, 0.f, __imageSize);
  tools::parallel_for_each_CUDA(__imageSize, [=] __device__(std::size_t index) {
    for (int i = 0; i < __knnNumbers; i++) {
      auto toIndex = __knnTo[index * __knnNumbers + i];
      auto weight = __knnValue[index * __knnNumbers + i];
      atomicAdd(&__outImage[toIndex], weight * __inImage[index]);
    }
  });
}
} // namespace openpni::experimental::node::impl