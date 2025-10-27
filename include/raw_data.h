#pragma once

#include <fstream>
#include <memory>
#include <span>
#include <vector>

#include <opencv2/opencv.hpp>

#include "texture.h"
#include "utils.h"

struct RawPETDataParameters {
  size_t num_bins = 0;
  size_t num_views = 0;
  size_t num_slices = 0;
};

template<typename T>
class RawPETData;
using F64RawPETData = RawPETData<double>;
using F32RawPETData = RawPETData<float>;

template<typename T>
class RawPETData {
public:
  explicit RawPETData(const RawPETDataParameters &parameters) :
      _num_bins(parameters.num_bins), _num_views(parameters.num_views), _num_slices(parameters.num_slices) {
    if (_num_bins == 0 || _num_views == 0 || _num_slices == 0) {
      ERROR_AND_EXIT("RawPETData::RawPETData: invalid parameters");
    }
    _data = std::make_unique_for_overwrite<T[]>(_num_bins * _num_views * _num_slices);
  }

  RawPETData(size_t num_bins, size_t num_views, size_t num_slices) :
      _num_bins(num_bins), _num_views(num_views), _num_slices(num_slices) {
    if (_num_bins == 0 || _num_views == 0 || _num_slices == 0) {
      ERROR_AND_EXIT("RawPETData::RawPETData: invalid parameters");
    }
    _data = std::make_unique_for_overwrite<T[]>(_num_bins * _num_views * _num_slices);
  }

  RawPETData(const RawPETData &other) :
      _num_bins(other._num_bins), _num_views(other._num_views), _num_slices(other._num_slices) {
    if (other._data) {
      _data = std::make_unique_for_overwrite<T[]>(_num_bins * _num_views * _num_slices);
      std::copy(other._data.get(), other._data.get() + _num_bins * _num_views * _num_slices, _data.get());
    }
  }

  RawPETData(RawPETData &&other) noexcept :
      _num_bins(other._num_bins), _num_views(other._num_views), _num_slices(other._num_slices),
      _data(std::move(other._data)) {
    other._num_bins = 0;
    other._num_views = 0;
    other._num_slices = 0;
  }

  RawPETData &operator=(const RawPETData &other) {
    if (this != &other) {
      _num_bins = other._num_bins;
      _num_views = other._num_views;
      _num_slices = other._num_slices;
      if (other._data) {
        _data = std::make_unique_for_overwrite<T[]>(_num_bins * _num_views * _num_slices);
        std::ranges::copy(other._data.get(), other._data.get() + _num_bins * _num_views * _num_slices, _data.get());
      } else {
        _data = nullptr;
      }
    }
    return *this;
  }

  RawPETData &operator=(RawPETData &&other) noexcept {
    if (this != &other) {
      _num_bins = other._num_bins;
      _num_views = other._num_views;
      _num_slices = other._num_slices;
      _data = std::move(other._data);
      other._num_bins = 0;
      other._num_views = 0;
      other._num_slices = 0;
    }
    return *this;
  }

  T &operator()(size_t bin, size_t view, size_t slice) {
    if (bin >= _num_bins || view >= _num_views || slice >= _num_slices) {
      ERROR_AND_EXIT("RawPETData::operator(): index out of range");
    }
    return _data[slice * _num_bins * _num_views + view * _num_bins + bin];
  }

  const T &operator()(size_t bin, size_t view, size_t slice) const {
    if (bin >= _num_bins || view >= _num_views || slice >= _num_slices) {
      ERROR_AND_EXIT("RawPETData::operator(): index out of range");
    }
    return _data[slice * _num_bins * _num_views + view * _num_bins + bin];
  }

  T &operator[](size_t index) {
    if (index >= _num_bins * _num_views * _num_slices) {
      ERROR_AND_EXIT("RawPETData::operator[]: index out of range");
    }
    return _data[index];
  }

  const T &operator[](size_t index) const {
    if (index >= _num_bins * _num_views * _num_slices) {
      ERROR_AND_EXIT("RawPETData::operator[]: index out of range");
    }
    return _data[index];
  }

  static RawPETData from_file(const std::string_view path, const RawPETDataParameters &parameters,
                              const std::ifstream::off_type offset = 0) {
    RawPETData raw_data(parameters);
    raw_data.read_file(path, offset);
    return raw_data;
  }

  void read_file(std::string_view path, std::ifstream::off_type offset = 0);

  const T *data() const { return _data.get(); }
  T *data() { return _data.get(); }

  size_t num_bins() const { return _num_bins; }
  size_t num_views() const { return _num_views; }
  size_t num_slices() const { return _num_slices; }

  std::span<T> slice(const size_t slice_index) {
    return std::span<T>(_data.get() + slice_index * _num_bins * _num_views, _num_bins * _num_views);
  }

  std::span<const T> slice(const size_t slice_index) const {
    return std::span<const T>(_data.get() + slice_index * _num_bins * _num_views, _num_bins * _num_views);
  }

  cv::Mat slice_cvimage(const size_t slice_index) const {
    return cv::Mat(static_cast<int>(_num_views), static_cast<int>(_num_bins), cv::DataType<T>::type,
                   _data.get() + slice_index * _num_bins * _num_views);
  }

  Texture2D slice_texture(const size_t slice_index, const torch::TensorOptions &options = {}) const {
    return Texture2D(_data.get() + slice_index * _num_bins * _num_views, _num_bins, _num_views, 1, options);
  }

private:
  size_t _num_bins = 0;
  size_t _num_views = 0;
  size_t _num_slices = 0;
  std::unique_ptr<T[]> _data = nullptr;
};

// implementation

template<typename T>
void RawPETData<T>::read_file(const std::string_view path, const std::ifstream::off_type offset) {

  if (!_data) {
    ERROR_AND_EXIT("RawPETData::read_file: data not initialized");
  }
  std::ifstream file(path.data(), std::ios::binary);
  if (!file.is_open()) {
    ERROR_AND_EXIT("RawPETData::read_file: cannot open file {}", path);
  }
  if (const auto file_size = file.seekg(0, std::ios::end).tellg();
      file_size != sizeof(T) * _num_bins * _num_views * _num_slices + offset) {
    ERROR_AND_EXIT("RawPETData::read_file: file size mismatch");
  }
  file.seekg(offset, std::ios::beg);
  file.read(reinterpret_cast<std::ifstream::char_type *>(_data.get()),
            sizeof(T) * _num_bins * _num_views * _num_slices);
  file.close();
}
