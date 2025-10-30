#pragma once
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "../basic/Vector.hpp"
namespace openpni::process {
template <typename Func>
inline void for_each_CUDA(
    std::size_t __max, Func &&__func, cudaStream_t __stream) {
  thrust::for_each(thrust::cuda::par.on(__stream), thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(__max), __func);
}
// template <typename Func>
// inline void for_each_CUDA(
//     basic::MDBeginEndSpan<3> __span, Func &&__func) {
//   const auto size = __span.ends - __span.begins;
//   thrust::for_each(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(__span.totalSize()),
//                    [=] __device__(std::size_t index) {
//                      auto index3D = basic::Vector<int64_t, 3>::create(index % size[0], (index / size[0]) % size[1],
//                                                                       index / (size[0] * size[1])) +
//                                     __span.begins;
//                      __func(index3D);
//                    });
// }
// template <typename ValueType>
// inline ValueType sum_CUDA(
//     Image3DInputSpan<ValueType> __imgSpan, basic::MDBeginEndSpan<3> __roiSpan) {
//   std::size_t count = __roiSpan.totalSize();
//   auto getValueFunc = [=] __device__(std::size_t index) -> ValueType {
//     auto index3D =
//         basic::Vector<int64_t, 3>::create(
//             index % (__roiSpan.ends[0] - __roiSpan.begins[0]),
//             (index / (__roiSpan.ends[0] - __roiSpan.begins[0])) % (__roiSpan.ends[1] - __roiSpan.begins[1]),
//             index / ((__roiSpan.ends[0] - __roiSpan.begins[0]) * (__roiSpan.ends[1] - __roiSpan.begins[1]))) +
//         __roiSpan.begins;
//     if (!__imgSpan.geometry.in(index3D.data[0], index3D.data[1], index3D.data[2]))
//       return 0;
//     return __imgSpan.ptr[__imgSpan.geometry.at(index3D.data[0], index3D.data[1], index3D.data[2])];
//   };
//   auto first = thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), getValueFunc);
//   auto last = thrust::make_transform_iterator(thrust::counting_iterator<size_t>(count), getValueFunc);
//   return thrust::reduce(first, last, ValueType(0), thrust::plus<ValueType>());
// }
// template <typename ValueType>
// inline ValueType squaredSum_CUDA(
//     Image3DInputSpan<ValueType> __imgSpan, basic::MDBeginEndSpan<3> __roiSpan) {
//   std::size_t count = __roiSpan.totalSize();
//   auto getValueFunc = [=] __device__(std::size_t index) -> ValueType {
//     auto index3D =
//         basic::Vector<int64_t, 3>::create(
//             index % (__roiSpan.ends[0] - __roiSpan.begins[0]),
//             (index / (__roiSpan.ends[0] - __roiSpan.begins[0])) % (__roiSpan.ends[1] - __roiSpan.begins[1]),
//             index / ((__roiSpan.ends[0] - __roiSpan.begins[0]) * (__roiSpan.ends[1] - __roiSpan.begins[1]))) +
//         __roiSpan.begins;
//     if (!__imgSpan.geometry.in(index3D.data[0], index3D.data[1], index3D.data[2]))
//       return 0;
//     auto value = __imgSpan.ptr[__imgSpan.geometry.at(index3D.data[0], index3D.data[1], index3D.data[2])];
//     return value * value;
//   };
//   auto first = thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), getValueFunc);
//   auto last = thrust::make_transform_iterator(thrust::counting_iterator<size_t>(count), getValueFunc);
//   return thrust::reduce(first, last, ValueType(0), thrust::plus<ValueType>());
// }
} // namespace openpni::process
