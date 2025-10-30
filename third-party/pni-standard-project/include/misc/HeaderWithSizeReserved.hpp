#pragma once
namespace openpni::misc {
#pragma pack(push, 1)
template <typename HeaderType, int SizeConstraint>
struct HeaderWithSizeReserved {
  HeaderType header;
  static_assert(sizeof(HeaderType) < SizeConstraint, "Header size exceeds the constraint");
  char __RESERVED_DONT_USE[SizeConstraint - sizeof(header)]{0}; // 填充到指定大小

  HeaderWithSizeReserved() = default;
  HeaderWithSizeReserved(
      const HeaderType &header)
      : header(header) {}

  template <typename OutStream>
  static bool writeToStream(
      OutStream &ofs, const HeaderType &header) {
    HeaderWithSizeReserved<HeaderType, SizeConstraint> H;
    H.header = header;
    ofs.write(reinterpret_cast<const char *>(&H), sizeof(H));
    return ofs.good();
  }
  template <typename InStream>
  static std::optional<HeaderType> readFromStream(
      InStream &ifs) {
    HeaderWithSizeReserved<HeaderType, SizeConstraint> H;
    ifs.read(reinterpret_cast<char *>(&H), sizeof(H));
    if (!ifs.good())
      return std::nullopt;
    return H.header;
  }
  template <typename InStream>
  static bool readFromStream(
      InStream &ifs, HeaderType &header) {
    auto result = readFromStream(ifs);
    if (!result)
      return false;
    header = result.value();
    return true;
  }

  static std::size_t copyToMemory(
      void *dest, const HeaderType &header) {
    HeaderWithSizeReserved<HeaderType, SizeConstraint> H;
    H.header = header;
    memcpy(dest, &H, sizeof(H));
    return sizeof(H);
  }
};
#pragma pack(pop)

} // namespace openpni::misc
