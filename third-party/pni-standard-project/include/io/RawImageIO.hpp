#pragma once
#include <cstring>
#include <fstream>

#include "../basic/Image.hpp"
#include "../misc/HeaderWithSizeReserved.hpp"
#include "../misc/ThirdParty.hpp"
namespace openpni::io::rawimage {
constexpr int RAW_IMAGE_FILE_HEADER_SIZE = 512; // RawImage文件头大小
#pragma pack(push, 1)
struct RawImageFileHeader {
  char elementType[16]{0};
  unsigned xNum;
  unsigned yNum;
  unsigned zNum;
  float xBegin;
  float yBegin;
  float zBegin;
  float xPixSize;
  float yPixSize;
  float zPixSize;
};
#pragma pack(pop)

template <typename RawType>
class Raw3D {
public:
  using value_type = RawType;
  using index_type = int;
  using index = std::tuple<index_type, index_type, index_type>;

public:
  Raw3D();
  ~Raw3D() = default;
  Raw3D(index_type x, index_type y, index_type z);
  Raw3D(index __size);
  Raw3D(basic::Vec3<index_type> __size);
  Raw3D(const basic::Image3DGeometry &__geometry);
  Raw3D(const Raw3D &) = default;

public:
  basic::Image3DGeometry imageGeometry() const noexcept;

public:
  RawType *data() const noexcept;
  inline RawType *begin() const noexcept { return data(); }
  uint64_t elements() const noexcept;
  inline RawType *end() const noexcept { return data() + elements(); }
  uint64_t bytes() const noexcept;
  RawType &operator()(index_type x, index_type y, index_type z) const noexcept;
  RawType &operator()(index) const noexcept;
  RawType &operator[](basic::Vec3<index_type> __index) const noexcept;
  bool isIndexValid(index_type x, index_type y, index_type z) const noexcept;
  bool isIndexValid(index) const noexcept;
  bool isIndexValid(basic::Vec3<index_type> __index) const noexcept;
  bool operator==(const Raw3D &other) const noexcept;
  Raw3D &realloc();
  operator bool() const noexcept;
  std::span<RawType> span() const noexcept;
  std::span<const RawType> cspan() const noexcept;

public:
  bool saveToFile(std::string path) noexcept;
  static tl::expected<Raw3D, std::string> loadFromFile(std::string path) noexcept;

private:
  float xBegin{};
  float yBegin{};
  float zBegin{};
  float xPixSize{};
  float yPixSize{};
  float zPixSize{};
  index_type xNum{};
  index_type yNum{};
  index_type zNum{};
  std::shared_ptr<RawType[]> value;

public:
  float getXBegin() const noexcept { return xBegin; }
  auto &setXBegin(
      float value) noexcept {
    xBegin = value;
    return *this;
  }
  float getYBegin() const noexcept { return yBegin; }
  auto &setYBegin(
      float value) noexcept {
    yBegin = value;
    return *this;
  }
  float getZBegin() const noexcept { return zBegin; }
  auto &setZBegin(
      float value) noexcept {
    zBegin = value;
    return *this;
  }
  float getXPixSize() const noexcept { return xPixSize; }
  auto &setXPixSize(
      float value) noexcept {
    xPixSize = value;
    return *this;
  }
  float getYPixSize() const noexcept { return yPixSize; }
  auto &setYPixSize(
      float value) noexcept {
    yPixSize = value;
    return *this;
  }
  float getZPixSize() const noexcept { return zPixSize; }
  auto &setZPixSize(
      float value) noexcept {
    zPixSize = value;
    return *this;
  }
  index_type getXNum() const noexcept { return xNum; }
  auto &setXNum(
      index_type value) noexcept {
    xNum = value;
    return *this;
  }
  index_type getYNum() const noexcept { return yNum; }
  auto &setYNum(
      index_type value) noexcept {
    yNum = value;
    return *this;
  }
  index_type getZNum() const noexcept { return zNum; }
  auto &setZNum(
      index_type value) noexcept {
    zNum = value;
    return *this;
  }

  basic::Vec3<float> getBegin() const noexcept { return basic::make_vec3<float>(xBegin, yBegin, zBegin); }
  auto &setBegin(
      basic::Vec3<float> __begin) noexcept {
    xBegin = __begin.x;
    yBegin = __begin.y;
    zBegin = __begin.z;
    return *this;
  }
  basic::Vec3<float> getPixSize() const noexcept { return basic::make_vec3<float>(xPixSize, yPixSize, zPixSize); }
  auto &setPixSize(
      basic::Vec3<float> __pixSize) noexcept {
    xPixSize = __pixSize.x;
    yPixSize = __pixSize.y;
    zPixSize = __pixSize.z;
    return *this;
  }
  basic::Vec3<index_type> getNum() const noexcept { return basic::make_vec3<index_type>(xNum, yNum, zNum); }
  auto &setNum(
      basic::Vec3<index_type> __size) noexcept {
    xNum = __size.x;
    yNum = __size.y;
    zNum = __size.z;
    return *this;
  }
};
} // namespace openpni::io::rawimage
namespace openpni::io::rawimage {

template <typename RawType>
inline Raw3D<RawType>::Raw3D() {}

template <typename RawType>
inline Raw3D<RawType>::Raw3D(
    index_type x, index_type y, index_type z) {
  value = std::shared_ptr<RawType[]>(new RawType[x * y * z]);
  xNum = x;
  yNum = y;
  zNum = z;
}

template <typename RawType>
inline Raw3D<RawType>::Raw3D(
    index __size)
    : Raw3D<RawType>(std::get<0>(__size), std::get<1>(__size), std::get<2>(__size)) {}

template <typename RawType>
inline Raw3D<RawType>::Raw3D(
    basic::Vec3<index_type> __size)
    : Raw3D<RawType>(__size.x, __size.y, __size.z) {}
template <typename RawType>
inline Raw3D<RawType>::Raw3D(
    const basic::Image3DGeometry &__geometry) {
  xNum = __geometry.voxelNum.x;
  yNum = __geometry.voxelNum.y;
  zNum = __geometry.voxelNum.z;
  xBegin = __geometry.imgBegin.x;
  yBegin = __geometry.imgBegin.y;
  zBegin = __geometry.imgBegin.z;
  xPixSize = __geometry.voxelSize.x;
  yPixSize = __geometry.voxelSize.y;
  zPixSize = __geometry.voxelSize.z;
  value = std::shared_ptr<RawType[]>(new RawType[elements()]);
}

template <typename RawType>
inline basic::Image3DGeometry Raw3D<RawType>::imageGeometry() const noexcept {
  basic::Image3DGeometry result;
  result.voxelNum = basic::make_vec3<int>(xNum, yNum, zNum);
  result.imgBegin = basic::make_vec3<float>(xBegin, yBegin, zBegin);
  result.voxelSize = basic::make_vec3<float>(xPixSize, yPixSize, zPixSize);
  return result;
}

template <typename RawType>
inline RawType *Raw3D<RawType>::data() const noexcept {
  return value.get();
}

template <typename RawType>
inline uint64_t Raw3D<RawType>::elements() const noexcept {
  return static_cast<uint64_t>(xNum) * yNum * zNum;
}

template <typename RawType>
inline uint64_t Raw3D<RawType>::bytes() const noexcept {
  return elements() * sizeof(RawType);
}

template <typename RawType>
inline RawType &Raw3D<RawType>::operator()(
    index_type x, index_type y, index_type z) const noexcept {
  return value[x + y * xNum + z * xNum * yNum];
}

template <typename RawType>
inline RawType &Raw3D<RawType>::operator()(
    index __index) const noexcept {
  return (*this)(std::get<0>(__index), std::get<1>(__index), std::get<2>(__index));
}

template <typename RawType>
inline RawType &Raw3D<RawType>::operator[](
    basic::Vec3<index_type> __index) const noexcept {
  return (*this)(__index.x, __index.y, __index.z);
}

template <typename RawType>
inline bool Raw3D<RawType>::isIndexValid(
    index_type x, index_type y, index_type z) const noexcept {
  return x < xNum && y < yNum && z < zNum;
}

template <typename RawType>
inline bool Raw3D<RawType>::isIndexValid(
    index __index) const noexcept {
  return isIndexValid(std::get<0>(__index), std::get<1>(__index), std::get<2>(__index));
}

template <typename RawType>
inline bool Raw3D<RawType>::isIndexValid(
    basic::Vec3<index_type> __index) const noexcept {
  return isIndexValid(__index.x, __index.y, __index.z);
}

template <typename RawType>
inline bool Raw3D<RawType>::operator==(
    const Raw3D<RawType> &other) const noexcept {
  // Only need to compare data pointer.
  return value == other.value;
}

template <typename RawType>
inline Raw3D<RawType>::operator bool() const noexcept {
  return value != nullptr;
}

template <typename RawType>
inline std::span<RawType> Raw3D<RawType>::span() const noexcept {
  return std::span<RawType>(value.get(), elements());
}

template <typename RawType>
inline std::span<const RawType> Raw3D<RawType>::cspan() const noexcept {
  return std::span<const RawType>(value.get(), elements());
}

template <typename RawType>
inline Raw3D<RawType> &Raw3D<RawType>::realloc() {
  if (elements() == 0)
    value.reset();
  else
    value = std::shared_ptr<RawType[]>(new RawType[elements()]);
  return *this;
}

template <typename RawType>
inline bool Raw3D<RawType>::saveToFile(
    std::string path) noexcept {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs.is_open())
    return false;

  misc::HeaderWithSizeReserved<RawImageFileHeader, RAW_IMAGE_FILE_HEADER_SIZE> _H_;
  auto &header = _H_.header;
  ::strncpy(header.elementType, typeid(RawType).name(), sizeof(header.elementType) - 1);
  header.xNum = xNum;
  header.yNum = yNum;
  header.zNum = zNum;
  header.xBegin = xBegin;
  header.yBegin = yBegin;
  header.zBegin = zBegin;
  header.xPixSize = xPixSize;
  header.yPixSize = yPixSize;
  header.zPixSize = zPixSize;

  ofs.write(reinterpret_cast<const char *>(&_H_), sizeof(_H_));
  ofs.write(reinterpret_cast<const char *>(value.get()), bytes());
  ofs.close();
  return true;
}
template <typename RawType>
inline tl::expected<Raw3D<RawType>, std::string> Raw3D<RawType>::loadFromFile(
    std::string path) noexcept {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open())
    return tl::unexpected("Failed to open file: " + path);

  misc::HeaderWithSizeReserved<RawImageFileHeader, RAW_IMAGE_FILE_HEADER_SIZE> _H_;
  ifs.read(reinterpret_cast<char *>(&_H_), sizeof(_H_));
  if (!ifs.good())
    return tl::unexpected("Failed to read header from file: " + path);

  auto &header = _H_.header;
  if (::strncmp(header.elementType, typeid(RawType).name(), sizeof(header.elementType)) != 0)
    return tl::unexpected("Element type mismatch: tl::expected " + std::string(typeid(RawType).name()) + ", got " +
                          std::string(header.elementType));

  const uint64_t elementExpected = static_cast<uint64_t>(header.xNum) * header.yNum * header.zNum;
  const auto fileSize = static_cast<uint64_t>(ifs.seekg(0, std::ios::end).tellg());
  if (fileSize != sizeof(_H_) + elementExpected * sizeof(RawType))
    return tl::unexpected("File size mismatch: tl::expected " +
                          std::to_string(sizeof(_H_) + elementExpected * sizeof(RawType)) + ", got " +
                          std::to_string(fileSize));
  ifs.seekg(sizeof(_H_), std::ios::beg);
  Raw3D<RawType> result(header.xNum, header.yNum, header.zNum);
  ifs.read(reinterpret_cast<char *>(result.data()), elementExpected * sizeof(RawType));
  if (!ifs.good())
    return tl::unexpected("Failed to read data from file: " + path);

  result.xBegin = header.xBegin;
  result.yBegin = header.yBegin;
  result.zBegin = header.zBegin;
  result.xPixSize = header.xPixSize;
  result.yPixSize = header.yPixSize;
  result.zPixSize = header.zPixSize;

  return result;
}
} // namespace openpni::io::rawimage
// namespace openpni::io::rawimage
