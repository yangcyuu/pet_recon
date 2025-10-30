#pragma once
#include <cmath>
#include <numbers>

#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../math/Convolution.hpp"
#include "../process/Foreach.hpp"

namespace openpni::example::knn {
template <typename ImageValueType>
struct _paddingImgInitial {

  Image3DSpan<ImageValueType> __out_paddingImgSpan;
  openpni::Image3DSpan<ImageValueType> __in_OSEMimg;
  basic::Vec3<int> __featureWHSize;

  void operator()(
      std::size_t OSEMidx) const {
    std::size_t x = OSEMidx % __in_OSEMimg.geometry.voxelNum.x;
    std::size_t y = (OSEMidx / __in_OSEMimg.geometry.voxelNum.x) % __in_OSEMimg.geometry.voxelNum.y;
    std::size_t z = OSEMidx / (__in_OSEMimg.geometry.voxelNum.x * __in_OSEMimg.geometry.voxelNum.y);

    __out_paddingImgSpan
        .ptr[__out_paddingImgSpan.geometry.at(x + __featureWHSize.x, y + __featureWHSize.y, z + __featureWHSize.z)] =
        __in_OSEMimg.ptr[OSEMidx];
  }
};

template <typename ImageValueType>
struct _featureMatrix {
  ImageValueType *__featureMatrixAll; // size = nVoxelNumInAll * featureSize
  basic::Vec3<int> __featureWHSize;
  basic::Image3DGeometry __in_OSEMimgGeo;
  Image3DSpan<ImageValueType> __in_paddingImg;

  void operator()(
      std::size_t idx) const {
    const std::size_t featureSize =
        (__featureWHSize.x * 2 + 1) * (__featureWHSize.y * 2 + 1) * (__featureWHSize.z * 2 + 1);
    const std::size_t indexInImg = idx / featureSize;
    const std::size_t featureIdx = idx % featureSize;

    // 计算特征窗口内的相对坐标
    const std::size_t featureX = featureIdx % (__featureWHSize.x * 2 + 1);
    const std::size_t featureY = (featureIdx / (__featureWHSize.x * 2 + 1)) % (__featureWHSize.y * 2 + 1);
    const std::size_t featureZ = featureIdx / ((__featureWHSize.x * 2 + 1) * (__featureWHSize.y * 2 + 1));

    // 获取原始图像中的坐标
    const std::size_t x = indexInImg % __in_OSEMimgGeo.voxelNum.x;
    const std::size_t y = (indexInImg / __in_OSEMimgGeo.voxelNum.x) % __in_OSEMimgGeo.voxelNum.y;
    const std::size_t z = indexInImg / (__in_OSEMimgGeo.voxelNum.x * __in_OSEMimgGeo.voxelNum.y);

    __featureMatrixAll[idx] =
        __in_paddingImg.ptr[__in_paddingImg.geometry.at(x + featureX, y + featureY, z + featureZ)];
  }
};

template <typename ImageValueType>
struct _featureMatrixNormalization {
  ImageValueType *__featureMatrixAll;
  basic::Vec3<int> __featureWHSize;
  std::size_t __OSEMTotalVoxelNum;

  void operator()(
      std::size_t featureIdx) const {
    ImageValueType mean = 0;
    ImageValueType stddev = 0;
    const size_t featureSize = (__featureWHSize.x * 2 + 1) * (__featureWHSize.y * 2 + 1) * (__featureWHSize.z * 2 + 1);
    for (const auto voxelIdx : std::views::iota(size_t(0), __OSEMTotalVoxelNum))
      mean += __featureMatrixAll[featureIdx + featureSize * voxelIdx];
    mean /= __OSEMTotalVoxelNum;
    for (const auto voxelIdx : std::views::iota(size_t(0), __OSEMTotalVoxelNum))
      stddev += basic::FMath<ImageValueType>::fpow(__featureMatrixAll[featureIdx + featureSize * voxelIdx] - mean, 2);
    stddev = basic::FMath<ImageValueType>::fsqrt(stddev / __OSEMTotalVoxelNum);
    for (const auto voxelIdx : std::views::iota(size_t(0), __OSEMTotalVoxelNum))
      __featureMatrixAll[featureIdx + featureSize * voxelIdx] /= stddev;
  }
};

template <typename ImageValueType>
struct _calKNNDistance {
  ImageValueType *__out_KNNKernel;
  size_t *__out_KNNto;
  ImageValueType *__featureMatrixAll; // size = nVoxelNumInAll * featureSize
  basic::Image3DGeometry __in_OSEMimgGeo;
  basic::Vec3<int> __searchWHSize;
  std::size_t __featureSize;
  float __sigmaG2;
  int __KNNnumbers;

  void operator()(
      std::size_t OSEMvoxelIdx) const {
    std::size_t x = OSEMvoxelIdx % __in_OSEMimgGeo.voxelNum.x;
    std::size_t y = (OSEMvoxelIdx / __in_OSEMimgGeo.voxelNum.x) % __in_OSEMimgGeo.voxelNum.y;
    std::size_t z = OSEMvoxelIdx / (__in_OSEMimgGeo.voxelNum.x * __in_OSEMimgGeo.voxelNum.y);

    std::size_t searchSize = (__searchWHSize.x * 2 + 1) * (__searchWHSize.y * 2 + 1) * (__searchWHSize.z * 2 + 1);
    std::vector<std::pair<ImageValueType, int>> searchInfo;

    for (const auto zz : std::views::iota(basic::max<int>(0, z - __searchWHSize.z),
                                          basic::min<int>(__in_OSEMimgGeo.voxelNum.z, z + __searchWHSize.z + 1)))
      for (const auto yy : std::views::iota(basic::max<int>(0, y - __searchWHSize.y),
                                            basic::min<int>(__in_OSEMimgGeo.voxelNum.y, y + __searchWHSize.y + 1)))
        for (const auto xx : std::views::iota(basic::max<int>(0, x - __searchWHSize.x),
                                              basic::min<int>(__in_OSEMimgGeo.voxelNum.x, x + __searchWHSize.x + 1))) {
          ImageValueType distance = 0;
          std::size_t searchIndexInImg = __in_OSEMimgGeo.at(xx, yy, zz);
          for (const auto featureIdx : std::views::iota(size_t(0), __featureSize))
            distance += basic::FMath<ImageValueType>::fpow(
                __featureMatrixAll[featureIdx + __featureSize * OSEMvoxelIdx] -
                    __featureMatrixAll[featureIdx + __featureSize * searchIndexInImg],
                2);
          searchInfo.emplace_back(distance, searchIndexInImg);
        }
    // sort distance to cal nearest KNN number position
    std::ranges::partial_sort(searchInfo.begin(), searchInfo.begin() + __KNNnumbers, searchInfo.end(),
                              [](const auto &a, const auto &b) { return a.first < b.first; });
    for (auto knnIdx : std::views::iota(0, __KNNnumbers)) {
      __out_KNNKernel[knnIdx + __KNNnumbers * OSEMvoxelIdx] = exp(-searchInfo[knnIdx].first / 2 / __sigmaG2);
      __out_KNNto[knnIdx + __KNNnumbers * OSEMvoxelIdx] = searchInfo[knnIdx].second;
    }
  }
};

template <typename ImageValueType>
struct _KNNNormalization {
  ImageValueType *__out_KNNKernel;
  int __KNNnumbers;
  void operator()(
      std::size_t OSEMvoxelIdx) const {
    ImageValueType kRowAll = 0;
    for (auto knnIdx : std::views::iota(0, __KNNnumbers))
      kRowAll += __out_KNNKernel[knnIdx + __KNNnumbers * OSEMvoxelIdx];
    for (auto knnIdx : std::views::iota(0, __KNNnumbers))
      __out_KNNKernel[knnIdx + __KNNnumbers * OSEMvoxelIdx] /= kRowAll;
  }
};

template <typename ImageValueType>
struct _generateKNNKernel {
  ImageValueType *__out_KNNKernel;
  size_t *__out_KNNto;
  openpni::Image3DSpan<ImageValueType> __in_OSEMimg;
  basic::Vec3<int> __featureWHSize;
  basic::Vec3<int> __searchWHSize;
  float __sigmaG2;
  int __KNNnumbers;

  void operator()() const {
    std::size_t featureSize = (__featureWHSize.x * 2 + 1) * (__featureWHSize.y * 2 + 1) * (__featureWHSize.z * 2 + 1);
    basic::Image3DGeometry paddingImgGeo{{__in_OSEMimg.geometry.voxelSize},
                                         {__in_OSEMimg.geometry.imgBegin - __featureWHSize},
                                         {__in_OSEMimg.geometry.voxelNum + __featureWHSize * 2}};
    std::unique_ptr<ImageValueType[]> paddingImg = std::make_unique<ImageValueType[]>(paddingImgGeo.totalVoxelNum());
    // padding img initial
    Image3DSpan<ImageValueType> paddingImgSpan{paddingImgGeo, paddingImg.get()};
    process::for_each(__in_OSEMimg.geometry.totalVoxelNum(),
                      _paddingImgInitial<ImageValueType>{paddingImgSpan, __in_OSEMimg, __featureWHSize});
    // generate feature matrix
    std::unique_ptr<ImageValueType[]> featureMatrix =
        std::make_unique<ImageValueType[]>(__in_OSEMimg.geometry.totalVoxelNum() * featureSize);
    process::for_each(__in_OSEMimg.geometry.totalVoxelNum() * featureSize,
                      _featureMatrix{featureMatrix.get(), __featureWHSize, __in_OSEMimg.geometry,
                                     Image3DSpan<ImageValueType>{paddingImgGeo, paddingImg.get()}});
    // featureMatrix normalization
    process::for_each(featureSize, _featureMatrixNormalization{featureMatrix.get(), __featureWHSize,
                                                               __in_OSEMimg.geometry.totalVoxelNum()});
    // cal KNN
    process::for_each(__in_OSEMimg.geometry.totalVoxelNum(),
                      _calKNNDistance{__out_KNNKernel, __out_KNNto, featureMatrix.get(), __in_OSEMimg.geometry,
                                      __searchWHSize, featureSize, __sigmaG2, __KNNnumbers});
    // KNN normalization
    process::for_each(__in_OSEMimg.geometry.totalVoxelNum(), _KNNNormalization{__out_KNNKernel, __KNNnumbers});
  }
};

template <typename knnKernelPrecision>
struct _KNNConvolution {
  knnKernelPrecision *__KNNkernel; // pre generate KNN kernel,size = nVoxelNumInAll * KNNnumbers
  size_t *__KNNto;                 // pre generate KNN point to img position,size = nVoxelNumInAll * KNNnumbers
  basic::Vec3<int> __kernelWHSize;
  int __kernelNumber;

  template <typename ImageValueType>
  __PNI_CUDA_MACRO__ ImageValueType convolution_impl(
      const ImageValueType *__in_Img3D, const int imgIndx) const {
    ImageValueType sum = 0;
    for (int k = 0; k < __kernelNumber; k++)
      sum += __KNNkernel[k + __kernelNumber * imgIndx] * __in_Img3D[__KNNto[k + __kernelNumber * imgIndx]];
    return sum;
  }
  template <typename ImageValueType>
  __PNI_CUDA_MACRO__ ImageValueType deconvolution_impl(
      const ImageValueType *__in_Img3D, const int imgIdx, const int convIdx) const {
    return __KNNkernel[convIdx] * __in_Img3D[imgIdx];
  }
};

template <typename ImageValueType>
inline void knnConvolution3d(
    const _KNNConvolution<ImageValueType> &knnKernel, openpni::Image3DIOSpan<ImageValueType> __convImg) {
  process::for_each(__convImg.geometry.totalVoxelNum(), [&](std::size_t imgIndx) {
    __convImg.ptr_out[imgIndx] = knnKernel.convolution_impl(__convImg.ptr_in, imgIndx);
  });
}
template <typename ImageValueType>
inline void knnDeconvolution3d(
    const _KNNConvolution<ImageValueType> &knnKernel, openpni::Image3DIOSpan<ImageValueType> __deconvImg) {

  process::for_each(__deconvImg.geometry.totalVoxelNum() * knnKernel.__kernelNumber, [&](std::size_t convIdx) {
    std::size_t imgIdx = convIdx / knnKernel.__kernelNumber;
    __deconvImg.ptr_out[knnKernel.__KNNto[convIdx]] +=
        knnKernel.deconvolution_impl(__deconvImg.ptr_in, imgIdx, convIdx);
  });
}

} // namespace openpni::example::knn

// namespace openpni::test {
// template <typename ImageValueType>
// struct KNNConvolution_Test {
//   int KNNnumbers;
//   float sigmaG2;
//   template <typename ImageValueType, typename FeatureSize, typename SearchSize>
//   inline void convolution3d(
//       openpni::Image3DIOSpan<ImageValueType> __convImg) {
//     ImageValueType mean[FeatureSize::total_size()];
//     ImageValueType standard[FeatureSize::total_size()];
//     const auto shiftMDSpan = basic::MDBeginEndSpan<3>::create(-FeatureSize::extent() / 2, FeatureSize::extent() / 2);
//     for (const auto [sx, sy, sz, idx] : shiftMDSpan) {
//       Vector<int64_t, 3> voxelNum = Vector<int64_t, 3>::create(
//           __convImg.geometry.voxelNum.x, __convImg.geometry.voxelNum.y, __convImg.geometry.voxelNum.z);
//       Vector<int64_t, 3> shift = Vector<int64_t, 3>::create(sx, sy, sz);
//       auto beginEndSpan = basic::MDBeginEndSpan<3>::create(shift, shift + voxelNum);
//       mean[idx] = process::sum_CUDA(__convImg.input_span(), beginEndSpan) / __convImg.geometry.totalVoxelNum();
//       standard[idx] =
//           process::squaredSum_CUDA(__convImg.input_span(), beginEndSpan) / __convImg.geometry.totalVoxelNum();
//     }

//     for_each_CUDA(__convImg.geometry.totalVoxelNum(), [=] __device__(basic::Vector<int64_t, 3> idx) {
//       struct VoxelMark {
//         float distance;
//         basic::Vector<int64_t, 3> voxelId;
//       } searchInfo[SearchSize::total_size()];
//       for (const auto [sx, sy, sz, sidx] :
//            basic::MDBeginEndSpan<3>::create(-SearchSize::extent() / 2, SearchSize::extent() / 2)) {
//         basic::Vector<int64_t, 3> shift = basic::Vector<int64_t, 3>::create(sx, sy, sz);
//         basic::Vector<int64_t, 3> searchVoxel = idx + shift;
//         if (!__convImg.geometry.in(searchVoxel))
//           continue;
//         float distance = 0;
//         for (const auto [fx, fy, fz, fidx] :
//              basic::MDBeginEndSpan<3>::create(-FeatureSize::extent() / 2, FeatureSize::extent() / 2)) {
//           basic::Vector<int64_t, 3> featureShift = basic::Vector<int64_t, 3>::create(fx, fy, fz);
//           basic::Vector<int64_t, 3> featureVoxel = searchVoxel + featureShift;
//           ImageValueType featureValue;
//           if (!__convImg.geometry.in(featureVoxel))
//             featureValue = 0;
//           else
//             featureValue =
//                 ((__convImg.ptr_in[__convImg.geometry.at(featureVoxel)] - mean[FeatureSize::mdspan()[featureShift]])
//                 /
//                      standard[FeatureSize::mdspan()[featureShift]] -
//                  (__convImg.ptr_in[__convImg.geometry.at(idx)] - mean[FeatureSize::mdspan()[featureShift]]) /
//                      standard[FeatureSize::mdspan()[featureShift]]);
//           distance += featureValue * featureValue;
//         }
//         searchInfo[sidx].voxelId = searchVoxel;

//         math::bubble_sort(searchInfo, searchInfo + SearchSize::total_size(),
//                           [] __device__(const VoxelMark &a, const VoxelMark &b) { return a.distance < b.distance; });
//         float coef[SearchSize::total_size()];
//         float coefSum = 0;
//         for (int i = 0; i < math::min(SearchSize::total_size(), KNNnumbers); i++) {
//           coef[i] = math::FMath<float>::fexp(-searchInfo[i].distance / 2 / sigmaG2);
//           coefSum += coef[i];
//         }
//         for (int i = 0; i < math::min(SearchSize::total_size(), KNNnumbers); i++)
//           coef[i] /= coefSum;
//         __convImg.ptr_out[idx] = 0;
//         for (int i = 0; i < math::min(SearchSize::total_size(), KNNnumbers); i++)
//           if (__convImg.geometry.in(searchInfo[i].voxelId))
//             __convImg.ptr_out[idx] += coef[i] * __convImg.ptr_in[__convImg.geometry.at(searchInfo[i].voxelId)];
//       }
//     });
//   }
// };
// } // namespace openpni::test