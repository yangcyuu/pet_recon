#pragma once
#include "Transforms.hpp"
namespace openpni::process {
template <typename InputValueType, typename OutputValueType,
          _impl::IsInterpolationMethod2D InterpolationMethod, typename... Actions>
  requires(_impl::IsTransformAction<Actions> && ...)
void transform2D_cuda(const InputValueType *__input, OutputValueType *__output,
                      const basic::Image2DGeometry &__ig,
                      const basic::Image2DGeometry &__og,
                      const InterpolationMethod &__ipMethod, std::size_t maxThreadNum,
                      const Actions &...actions) {
  dim3 block(16, 16);
  dim3 grid((__og.voxelNum.x + block.x - 1) / block.x,
            (__og.voxelNum.y + block.y - 1) / block.y);
  _impl::transform2D_kernel<<<grid, block>>>(__input, __output, __ig, __og, __ipMethod,
                                             (-actions)...);
  cudaDeviceSynchronize();
}
} // namespace openpni::process
