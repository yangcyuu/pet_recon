#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <utility>

#include "utils.h"

#define TEXTURES_OPERATORS(type, op)                                                                                   \
  inline type &operator op##=(const type & other) {                                                                    \
    this->_data op## = other._data;                                                                                    \
    return *this;                                                                                                      \
  }                                                                                                                    \
                                                                                                                       \
  inline type operator op(const type &other) const {                                                                   \
    type result = this->clone();                                                                                       \
    result op## = other;                                                                                               \
    return result;                                                                                                     \
  }

#define TEXTURE_SCALAR_OPERATORS(type, op)                                                                             \
  template<typename T>                                                                                                 \
  type &operator op##=(const T & value) {                                                                              \
    this->_data op## = value;                                                                                          \
    return *this;                                                                                                      \
  }                                                                                                                    \
                                                                                                                       \
  template<typename T>                                                                                                 \
  type operator op(const T &value) const {                                                                             \
    type result = this->clone();                                                                                       \
    result op## = value;                                                                                               \
    return result;                                                                                                     \
  }

#define SCALAR_TEXTURE_OPERATORS(type, op)                                                                             \
  template<typename T>                                                                                                 \
  friend type operator op(const T &value, const type &texture) {                                                       \
    type result(value op texture._data);                                                                               \
    return result;                                                                                                     \
  }

#define TEXTURE_ALL_OPERATORS(type, op)                                                                                \
  TEXTURES_OPERATORS(type, op)                                                                                         \
  TEXTURE_SCALAR_OPERATORS(type, op)                                                                                   \
  SCALAR_TEXTURE_OPERATORS(type, op)

class Texture2D {
public:
  Texture2D() = default;

  Texture2D(torch::Tensor tensor) : _data(std::move(tensor)) {
    // tensor layout: [C, H, W]
    if (_data.dim() != 3) {
      ERROR_AND_EXIT("Texture2D tensor must be 3-dimensional");
    }
  }

  Texture2D(size_t width, size_t height, size_t channel, const torch::TensorOptions &options = {}) {
    _data = torch::empty({static_cast<int64_t>(channel), static_cast<int64_t>(height), static_cast<int64_t>(width)},
                         options);
  }

  Texture2D(const void *data, size_t width, size_t height, size_t channel, const torch::TensorOptions &options = {}) {
    // data layout: [H, W, C]
    _data = torch::from_blob(const_cast<void *>(data),
                             {static_cast<int64_t>(height), static_cast<int64_t>(width), static_cast<int64_t>(channel)},
                             options)
                .permute({2, 0, 1})
                .clone(); // to [C, H, W]
  }

  template<typename T>
    requires(!std::is_pointer_v<T>)
  Texture2D(const T &value, size_t width, size_t height, size_t channel, const torch::TensorOptions &options = {}) {
    _data = torch::full({static_cast<int64_t>(channel), static_cast<int64_t>(height), static_cast<int64_t>(width)},
                        value, options);
  }

  TEXTURE_ALL_OPERATORS(Texture2D, +)
  TEXTURE_ALL_OPERATORS(Texture2D, -)
  TEXTURE_ALL_OPERATORS(Texture2D, *)
  TEXTURE_ALL_OPERATORS(Texture2D, /)

  size_t width() const { return _data.size(2); }
  size_t height() const { return _data.size(1); }
  size_t channel() const { return _data.size(0); }

  template<typename T>
  Texture2D clamp_min(const T &value) const {
    Texture2D result;
    result._data = torch::clamp_min(_data, value);
    return result;
  }

  template<typename T>
  Texture2D clamp_max(const T &value) const {
    Texture2D result;
    result._data = torch::clamp_max(_data, value);
    return result;
  }

  Texture2D log() const {
    Texture2D result;
    result._data = torch::log(_data);
    return result;
  }

  torch::Tensor eval_coords(const at::Tensor &coords) const {
    // coord: [N, 2], range [-1, 1], (x, y)
    auto options = torch::nn::functional::GridSampleFuncOptions()
                       .padding_mode(torch::kZeros)
                       .mode(torch::kBilinear)
                       .align_corners(true);
    torch::Tensor grid = coords.unsqueeze(0).unsqueeze(2); // [1, N, 1, 2]
    torch::Tensor input = _data.unsqueeze(0); // [1, C, H, W]
    torch::Tensor output = torch::nn::functional::grid_sample(input, grid,
                                                              options); // [1, C, N, 1]
    output = output.squeeze(0).squeeze(2); // [C, N]
    return output.transpose(0, 1); // [N, C]
  }

  torch::Tensor eval_coord(const at::Tensor &coord) const {
    // coord: [2], range [-1, 1], (x, y)
    auto options = torch::nn::functional::GridSampleFuncOptions()
                       .padding_mode(torch::kZeros)
                       .mode(torch::kBilinear)
                       .align_corners(true);
    torch::Tensor grid = coord.unsqueeze(0).unsqueeze(0).unsqueeze(0); // [1, 1, 1, 2]
    torch::Tensor input = _data.unsqueeze(0); // [1, C, H, W]
    torch::Tensor output = torch::nn::functional::grid_sample(input, grid,
                                                              options); // [1, C, 1, 1]
    output = output.squeeze(0).squeeze(2).squeeze(1); // [C]
    return output;
  }

  void assign(const torch::Tensor &rows, const torch::Tensor &cols, const torch::Tensor &channels,
              const torch::Tensor &values) {
    _data.index_put_({channels, rows, cols}, values);
  }

  void add_assign(const torch::Tensor &rows, const torch::Tensor &cols, const torch::Tensor &channels,
                  const torch::Tensor &values) {
    MARK_AS_UNUSED(_data.index_put_({channels, rows, cols}, values, true));
  }

  const torch::Tensor &tensor() const { return _data; }

  void set_requires_grad(bool requires_grad) const { MARK_AS_UNUSED(_data.set_requires_grad(requires_grad)); }

  void zero_grad() const {
    if (_data.grad().defined()) {
      MARK_AS_UNUSED(_data.grad().zero_());
    }
  }


  Texture2D clone() const { return Texture2D(_data.clone()); }

  void save(const std::string &filename) const;

private:
  torch::Tensor _data;
};

class Texture3D {
public:
  Texture3D() = default;

  Texture3D(torch::Tensor tensor) : _data(std::move(tensor)) {
    // tensor layout: [C, D, H, W]
    if (_data.dim() != 4) {
      ERROR_AND_EXIT("Texture3D tensor must be 4-dimensional");
    }
  }
  Texture3D(size_t width, size_t height, size_t depth, size_t channel, const torch::TensorOptions &options = {}) {
    _data = torch::empty({static_cast<int64_t>(channel), static_cast<int64_t>(depth), static_cast<int64_t>(height),
                          static_cast<int64_t>(width)},
                         options);
  }

  Texture3D(const void *data, size_t width, size_t height, size_t depth, size_t channel,
            const torch::TensorOptions &options = {}) {
    // data layout: [D, H, W, C]
    _data = torch::from_blob(const_cast<void *>(data),
                             {static_cast<int64_t>(depth), static_cast<int64_t>(height), static_cast<int64_t>(width),
                              static_cast<int64_t>(channel)},
                             options)
                .permute({3, 0, 1, 2})
                .clone(); // to [C, D, H, W]
  }

  template<typename T>
    requires(!std::is_pointer_v<T>)
  Texture3D(const T &value, size_t width, size_t height, size_t depth, size_t channel,
            const torch::TensorOptions &options = {}) {
    _data = torch::full({static_cast<int64_t>(channel), static_cast<int64_t>(depth), static_cast<int64_t>(height),
                         static_cast<int64_t>(width)},
                        value, options);
  }

  TEXTURE_ALL_OPERATORS(Texture3D, +)
  TEXTURE_ALL_OPERATORS(Texture3D, -)
  TEXTURE_ALL_OPERATORS(Texture3D, *)
  TEXTURE_ALL_OPERATORS(Texture3D, /)

  size_t width() const { return _data.size(3); }
  size_t height() const { return _data.size(2); }
  size_t depth() const { return _data.size(1); }
  size_t channel() const { return _data.size(0); }

  template<typename T>
  Texture3D clamp_min(const T &value) const {
    Texture3D result;
    result._data = torch::clamp_min(_data, value);
    return result;
  }

  template<typename T>
  Texture3D clamp_max(const T &value) const {
    Texture3D result;
    result._data = torch::clamp_max(_data, value);
    return result;
  }

  template<typename T>
  Texture3D clamp(const T &min_value, const T &max_value) const {
    Texture3D result;
    result._data = torch::clamp(_data, min_value, max_value);
    return result;
  }

  Texture3D log() const {
    Texture3D result;
    result._data = torch::log(_data);
    return result;
  }

  Texture3D &add_(const Texture3D &other) const {
    MARK_AS_UNUSED(this->_data.add_(other._data));
    return const_cast<Texture3D &>(*this);
  }

  Texture3D &sub_(const Texture3D &other) const {
    MARK_AS_UNUSED(this->_data.sub_(other._data));
    return const_cast<Texture3D &>(*this);
  }

  Texture3D &mul_(const Texture3D &other) const {
    MARK_AS_UNUSED(this->_data.mul_(other._data));
    return const_cast<Texture3D &>(*this);
  }

  Texture3D &div_(const Texture3D &other) const {
    MARK_AS_UNUSED(this->_data.div_(other._data));
    return const_cast<Texture3D &>(*this);
  }

  template<typename T>
  Texture3D &clamp_(const T &min_value, const T &max_value) const {
    MARK_AS_UNUSED(this->_data.clamp_(min_value, max_value));
    return const_cast<Texture3D &>(*this);
  }

  template<typename T>
  Texture3D &clamp_min_(const T &value) const {
    MARK_AS_UNUSED(this->_data.clamp_min_(value));
    return const_cast<Texture3D &>(*this);
  }

  template<typename T>
  Texture3D &clamp_max_(const T &value) const {
    MARK_AS_UNUSED(this->_data.clamp_max_(value));
    return const_cast<Texture3D &>(*this);
  }

  torch::Tensor eval_coords(const at::Tensor &coords) const {
    // coord: [N, 3], range [-1, 1], (x, y, z)
    auto options = torch::nn::functional::GridSampleFuncOptions()
                       .padding_mode(torch::kZeros)
                       .mode(torch::kBilinear)
                       .align_corners(true);
    torch::Tensor grid = coords.unsqueeze(0).unsqueeze(2).unsqueeze(2); // [1, N, 1, 1, 3]
    torch::Tensor input = _data.unsqueeze(0); // [1, C, D, H, W]
    torch::Tensor output = torch::nn::functional::grid_sample(input, grid,
                                                              options); // [1, C, N, 1, 1]
    output = output.squeeze(0).squeeze(2).squeeze(2); // [C, N]
    return output.transpose(0, 1); // [N, C]
  }

  torch::Tensor eval_coord(const at::Tensor &coord) const {
    // coord: [3], range [-1, 1], (x, y, z)
    auto options = torch::nn::functional::GridSampleFuncOptions()
                       .padding_mode(torch::kZeros)
                       .mode(torch::kBilinear)
                       .align_corners(true);
    torch::Tensor grid = coord.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0); // [1, 1, 1, 1, 3]
    torch::Tensor input = _data.unsqueeze(0); // [1, C, D, H, W]
    torch::Tensor output = torch::nn::functional::grid_sample(input, grid,
                                                              options); // [1, C, 1, 1]
    output = output.squeeze(0).squeeze(2).squeeze(1); // [C]
    return output;
  }

  void assign(const torch::Tensor &depths, const torch::Tensor &rows, const torch::Tensor &cols,
              const torch::Tensor &channels, const torch::Tensor &values) {
    // depths, rows, cols, channels, values: [N]
    _data.index_put_({channels, depths, rows, cols}, values);
  }

  void add_assign(const torch::Tensor &depths, const torch::Tensor &rows, const torch::Tensor &cols,
                  const torch::Tensor &channels, const torch::Tensor &values) {
    // depths, rows, cols, channels, values: [N]
    MARK_AS_UNUSED(_data.index_put_({channels, depths, rows, cols}, values, true));
  }

  const torch::Tensor &tensor() const { return _data; }

  void set_requires_grad(bool requires_grad) const { MARK_AS_UNUSED(_data.set_requires_grad(requires_grad)); }

  void zero_grad() const {
    if (_data.grad().defined()) {
      MARK_AS_UNUSED(_data.grad().zero_());
    }
  }

  Texture3D grad() const {
    Texture3D result;
    if (_data.grad().defined()) {
      result._data = _data.grad();
    } else {
      result._data = torch::zeros_like(_data);
    }
    return result;
  }


  Texture3D clone() const { return Texture3D(_data.clone()); }

  void save(const std::string &filename) const;

  void save_rawdata(std::string_view filename) const;

private:
  torch::Tensor _data;
};
