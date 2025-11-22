#include "BDM2.hpp"

#include <format>

#include "include/Exceptions.hpp"
#include "include/experimental/tools/Parallel.hpp"
#include "include/misc/HeaderWithSizeReserved.hpp"
#include "src/autogen/autogen_xml.hpp"
#include "src/common/FileFormat.hpp"
namespace openpni::experimental::node::impl {
using namespace openpni::device::bdm2;
openpni::device::bdm2::CalibrationTable read_bdm2_calibration_table(
    std::string filename) { // 从文件中读取二进制数据
  using CaliFileHeader =
      misc::HeaderWithSizeReserved<openpni::autogen::GeneralFileHeader, openpni::autogen::GeneralFileHeaderSize>;
  auto stream = openpni::autogen::readBinaryFile(filename);
  if (!stream)
    throw openpni::exceptions::file_cannot_access();

  CaliFileHeader header;
  stream->seekg(0, std::ios_base::beg);
  stream->read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!stream->good())
    throw openpni::exceptions::file_format_incorrect();
  // Skip sizeof(CaliFileHeader) bytes from the beginning of the stream
  auto stream_skip_header = openpni::autogen::subStream(*stream, sizeof(CaliFileHeader), stream->str().size());

#define try_assert(condition)                                                                                          \
  if (!(condition)) {                                                                                                  \
    throw openpni::exceptions::file_format_incorrect();                                                                \
  }

  CalibrationTable result;

  try {
    if (header.header.version == 1) {
      auto caliFile =
          openpni::autogen::binary::struct_cast<openpni::autogen::binary::BDM2CaliFileV1>(stream_skip_header);
      result.energyCoef = std::make_unique<std::array<EnergyCoefs_t, BLOCK_NUM>>();
      result.positionTable = std::make_unique<std::array<PositionTable_t, BLOCK_NUM>>();

      try_assert(caliFile.EnergyCoefs.size() == BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        std::copy(caliFile.EnergyCoefs.begin() + i * CRYSTAL_NUM_ONE_BLOCK,
                  caliFile.EnergyCoefs.begin() + (i + 1) * CRYSTAL_NUM_ONE_BLOCK, (*result.energyCoef)[i].begin());
      }

      try_assert(caliFile.PositionTable.size() == BLOCK_NUM * 256 * 256);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        std::copy(caliFile.PositionTable.begin() + i * 256 * 256, caliFile.PositionTable.begin() + (i + 1) * 256 * 256,
                  (*result.positionTable)[i].begin());
      }

      return result;
    }
  } catch (const std::exception &e) {
    throw openpni::exceptions::file_format_incorrect();
  }
  throw openpni::exceptions::file_unknown_version();
}
void h_r2s(
    uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length, uint16_t channelIndex,
    BDM2ConvertSingleContext h_ctx, interface::LocalSingle *h_out, uint64_t count,
    uint32_t const *h_packetInclusiveSum) {
  tools::parallel_for_each(count, [=](uint64_t index) {
    auto *outBegin = h_out + (index == 0 ? 0 : h_packetInclusiveSum[index - 1]) * SINGLE_NUM_PER_PACKET;
    auto *outEnd = h_out + h_packetInclusiveSum[index] * SINGLE_NUM_PER_PACKET;
    if (outBegin == outEnd)
      return;
    auto *raw = h_raw + h_offset[index];
    auto length = h_length[index];
    impl_r2s(raw, length, channelIndex, h_ctx, outBegin, index);
  });
}
void h_r2s_converged(
    uint8_t const *h_raw, uint64_t const *h_offset, uint16_t const *h_length, uint16_t const *h_channel,
    BDM2ConvertSingleContext const *h_ctx, uint16_t const *h_channelIndicesMap, interface::LocalSingle *h_out,
    uint64_t count, uint32_t const *h_packetInclusiveSum) {
  tools::parallel_for_each(count, [=](uint64_t index) {
    auto *outBegin = h_out + (index == 0 ? 0 : h_packetInclusiveSum[index - 1]) * SINGLE_NUM_PER_PACKET;
    auto *outEnd = h_out + h_packetInclusiveSum[index] * SINGLE_NUM_PER_PACKET;
    if (outBegin == outEnd)
      return;
    auto *raw = h_raw + h_offset[index];
    auto length = h_length[index];
    auto channelIndex = h_channel[index];
    auto ctxIndex = h_channelIndicesMap[channelIndex];
    if (ctxIndex == 0xFFFF)
      return;
    impl_r2s(raw, length, channelIndex, h_ctx[ctxIndex], outBegin, index);
  });
}
} // namespace openpni::experimental::node::impl
