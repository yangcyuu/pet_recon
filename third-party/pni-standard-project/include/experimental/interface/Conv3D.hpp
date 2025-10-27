#pragma once
#include "../core/Image.hpp"
namespace openpni::experimental::interface {
class Conv3D {
public:
  Conv3D() = default;
  virtual ~Conv3D() = default;

public:
  virtual void convH(core::TensorDataIO<float, 3> __image) = 0;
  virtual void deconvH(core::TensorDataIO<float, 3> __image) = 0;
  virtual void convD(core::TensorDataIO<float, 3> __image) = 0;
  virtual void deconvD(core::TensorDataIO<float, 3> __image) = 0;
};
} // namespace openpni::experimental::interface