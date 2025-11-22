#pragma once

#include <filesystem>
#include <fstream>

#include "texture.h"
#include "utils.h"

template<typename T, size_t N>
  requires(N >= 2)
class Image {
public:
  Image() = default;

  Image(T *data_ptr, const int64_t (&dims)[N]) {
    size_t total_size = 1;
    for (size_t i = 0; i < N; ++i) {
      total_size *= dims[i];
      _dimensions[i] = dims[i];
    }
    _data = std::make_unique<T[]>(total_size);
    std::copy(data_ptr, data_ptr + total_size, _data.get());
  }

  Image(T *data_ptr, int64_t width, int64_t height) {
    static_assert(N == 2, "Image constructor with width and height is only for 2D images.");
    _data = std::make_unique<T[]>(width * height);
    std::copy(data_ptr, data_ptr + width * height, _data.get());
    _dimensions[0] = width;
    _dimensions[1] = height;
  }

  Image(T *data_ptr, int64_t width, int64_t height, int64_t depth) {
    static_assert(N == 3, "Image constructor with width, height and depth is only for 3D images.");
    _data = std::make_unique<T[]>(width * height * depth);
    std::copy(data_ptr, data_ptr + width * height * depth, _data.get());
    _dimensions[0] = width;
    _dimensions[1] = height;
    _dimensions[2] = depth;
  }

  template<typename... Dims>
    requires(sizeof...(Dims) + 3 == N)
  Image(T *data_ptr, int64_t width, int64_t height, int64_t depth, Dims... dims) {
    static_assert(N >= 4, "Image constructor with additional dimensions is only for images with 4 or more dimensions.");
    const size_t total_size = width * height * depth;
    ((total_size *= dims), ...);
    _data = std::make_unique<T[]>(total_size);
    std::copy(data_ptr, data_ptr + total_size, _data.get());
    _dimensions[0] = width;
    _dimensions[1] = height;
    _dimensions[2] = depth;
    size_t idx = 3;
    ((_dimensions[idx++] = dims), ...);
  }

  Image(const Image &other) {
    size_t total_size = 1;
    for (size_t i = 0; i < N; ++i) {
      total_size *= other._dimensions[i];
      _dimensions[i] = other._dimensions[i];
    }
    _data = std::make_unique<T[]>(total_size);
    std::copy(other._data.get(), other._data.get() + total_size, _data.get());
  }

  Image(Image &&other) noexcept : _data(std::move(other._data)) {
    std::copy(std::begin(other._dimensions), std::end(other._dimensions), std::begin(_dimensions));
  }

  static Image from_file(const std::string &filename, const int64_t (&dims)[N], std::fstream::off_type offset = 0) {
    Image img;
    size_t total_size = 1;
    for (size_t i = 0; i < N; ++i) {
      total_size *= dims[i];
      img._dimensions[i] = dims[i];
    }
    img._data = std::make_unique<T[]>(total_size);
    img.read_file(filename, offset);
    return img;
  }

  static Image from_file(const std::string &filename, int64_t width, int64_t height,
                         std::fstream::off_type offset = 0) {
    static_assert(N == 2, "Image::from_file with width and height is only for 2D images.");
    Image img;
    img._dimensions[0] = width;
    img._dimensions[1] = height;
    img._data = std::make_unique<T[]>(width * height);
    img.read_file(filename, offset);
    return img;
  }

  static Image from_file(const std::string &filename, int64_t width, int64_t height, int64_t depth,
                         std::fstream::off_type offset = 0) {
    static_assert(N == 3, "Image::from_file with width, height and depth is only for 3D images.");
    Image img;
    img._dimensions[0] = width;
    img._dimensions[1] = height;
    img._dimensions[2] = depth;
    img._data = std::make_unique<T[]>(width * height * depth);
    img.read_file(filename, offset);
    return img;
  }

  Image &operator=(const Image &other) {
    if (this != &other) {
      size_t total_size = 1;
      for (size_t i = 0; i < N; ++i) {
        total_size *= other._dimensions[i];
        _dimensions[i] = other._dimensions[i];
      }
      _data = std::make_unique<T[]>(total_size);
      std::copy(other._data.get(), other._data.get() + total_size, _data.get());
    }
    return *this;
  }

  Image &operator=(Image &&other) noexcept {
    if (this != &other) {
      _data = std::move(other._data);
      std::copy(std::begin(other._dimensions), std::end(other._dimensions), std::begin(_dimensions));
    }
    return *this;
  }

  size_t size() const {
    size_t total_size = 1;
    for (size_t i = 0; i < N; ++i) {
      total_size *= _dimensions[i];
    }
    return total_size;
  }

  int64_t width() const { return _dimensions[0]; }

  int64_t height() const { return _dimensions[1]; }

  int64_t depth() const {
    static_assert(N >= 3, "Image must have at least 3 dimensions to get depth.");
    return _dimensions[2];
  }

  auto texture(const torch::TensorOptions &options = {}) const {
    if constexpr (N == 2) {
      if (options.device().is_cpu()) {
        return Texture2D(_data.get(), width(), height(), 1, options);
      }
      if (options.device().is_cuda()) {
        return Texture2D::cuda(_data.get(), width(), height(), 1, options.device(torch::kCPU));
      }
      ERROR_AND_EXIT("Unsupported device type for Image::texture");
    } else if constexpr (N == 3) {
      if (options.device().is_cpu()) {
        return Texture3D(_data.get(), width(), height(), depth(), 1, options);
      }
      if (options.device().is_cuda()) {
        return Texture3D::cuda(_data.get(), width(), height(), depth(), 1, options.device(torch::kCPU));
      }
      ERROR_AND_EXIT("Unsupported device type for Image::texture");
    } else {
      ERROR_AND_EXIT("Image texture conversion is only supported for 2D and 3D images.");
    }
  }

  void read_file(const std::string &filename, std::fstream::off_type offset) {
    std::filesystem::path filepath = std::filesystem::absolute(filename);
    if (!std::filesystem::exists(filepath)) {
      ERROR_AND_EXIT("File does not exist: {}", filepath.string());
    }
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
      ERROR_AND_EXIT("Failed to open file for reading: {}", filepath.string());
    }
    ifs.seekg(0, std::ios::end);
    size_t file_size = ifs.tellg();
    if (sizeof(T) * size() + offset != file_size) {
      ERROR_AND_EXIT("File size does not match image size: {}", filepath.string());
    }
    ifs.seekg(offset, std::ios::beg);
    ifs.read(reinterpret_cast<char *>(_data.get()), sizeof(T) * size());
    ifs.close();
  }

  T *data() const { return _data.get(); }

  T *release() { return _data.release(); }

private:
  std::unique_ptr<T[]> _data;
  int64_t _dimensions[N];
};

template<typename T>
using Image2D = Image<T, 2>;

template<typename T>
using Image3D = Image<T, 3>;
