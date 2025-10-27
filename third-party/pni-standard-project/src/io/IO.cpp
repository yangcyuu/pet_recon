#include "include/io/IO.hpp"
#include "ListmodeFileInputImpl.hpp"
#include "ListmodeFileOutputImpl.hpp"
#include "RawFileInputImpl.hpp"
#include "RawFileOutputImpl.hpp"
#include "SingleFileInputImpl.hpp"
#include "SingleFileOutputImpl.hpp"
#include "include/io/RawDataIO.hpp"
namespace openpni::io {
RawFileInput::RawFileInput() noexcept {
  impl = std::make_unique<RawFileInputImpl>();
}
RawFileInput::~RawFileInput() noexcept {}
void RawFileInput::open(std::string path) {
  impl->open(path);
}
uint32_t RawFileInput::segmentNum() const noexcept {
  return impl->segmentNum();
}
void RawFileOutput::setChannelNum(uint16_t channelNum) noexcept {
  impl->setChannelNum(channelNum);
}

rawdata::RawdataHeader RawFileInput::header() const noexcept {
  return impl->getHeader();
}
rawdata::SegmentHeader RawFileInput::segmentHeader(uint32_t segmentIndex) const noexcept {
  return impl->getSegmentHeader(segmentIndex);
}
rawdata::RawdataSegment RawFileInput::readSegment(uint32_t segmentIndex,
                                                  uint32_t prefetchIndex) const noexcept {
  return impl->getSegment(segmentIndex, prefetchIndex);
}

void RawFileOutput::setReservedBytes(uint64_t r) noexcept {
  impl->setReservedBytes(r);
}

RawFileOutput::RawFileOutput() noexcept {
  impl = std::make_unique<RawFileOutputImpl>();
}
RawFileOutput::~RawFileOutput() noexcept {}
void RawFileOutput::open(std::string path) {
  impl->open(path);
}
bool RawFileOutput::appendSegment(process::RawDataView rawData) noexcept {
  return impl->appendSegment(rawData);
}
rawdata::RawDataView
rawdata::RawdataSegment::view(const RawdataHeader &fileHeader,
                              const SegmentHeader &segmentHeader) const {
  rawdata::RawDataView result;
  result.channel = channel.get();
  result.channelNum = fileHeader.channelNum;
  result.clock_ms = segmentHeader.clock;
  result.count = segmentHeader.count;
  result.data = data.get();
  result.length = length.get();
  result.offset = offset.get();
  result.duration_ms = segmentHeader.duration;
  return result;
}

SingleFileOutput::SingleFileOutput() noexcept {
  impl = std::make_unique<SingleFileOutput_impl>();
}
SingleFileOutput::~SingleFileOutput() noexcept {}
void SingleFileOutput::open(std::string path) {
  impl->open(path);
}
bool SingleFileOutput::setBytes4CrystalIndex(CrystalIndexType type) noexcept {
  return impl->setBytes4CrystalIndex(type);
}
bool SingleFileOutput::setBytes4TimeValue(TimeValueType type) noexcept {
  return impl->setBytes4TimeValue(type);
}
bool SingleFileOutput::setBytes4Energy(EnergyType type) noexcept {
  return impl->setBytes4Energy(type);
}
void SingleFileOutput::setTotalCrystalNum(uint32_t num) noexcept {
  impl->setTotalCrystalNum(num);
}
void SingleFileOutput::setReservedBytes(uint64_t r) noexcept {
  impl->setReservedBytes(r);
}
bool SingleFileOutput::appendSegment(const basic::GlobalSingle_t *data, uint64_t count,
                                     uint64_t clock, uint32_t duration) {
  return impl->appendSegment(data, count, clock, duration);
}
single::SingleSegmentHeader
SingleFileInput::segmentHeader(uint32_t segmentIndex) const noexcept {
  return impl->segmentHeader(segmentIndex);
}
SingleFileInput::SingleFileInput() noexcept {
  impl = std::make_unique<SingleFileInput_impl>();
}
SingleFileInput::~SingleFileInput() noexcept {}
void SingleFileInput::open(std::string path) {
  impl->open(path);
}
single::SingleFileHeader SingleFileInput::header() const noexcept {
  return impl->header();
}
uint32_t SingleFileInput::segmentNum() const noexcept {
  return impl->segmentNum();
}
single::SingleSegmentBytes
SingleFileInput::readSegment(uint32_t segmentIndex,
                             uint32_t prefetchIndex) const noexcept {
  return impl->readSegment(segmentIndex, prefetchIndex);
}

ListmodeFileOutput::ListmodeFileOutput() noexcept {
  impl = std::make_unique<ListmodeFileOutput_impl>();
}
ListmodeFileOutput::~ListmodeFileOutput() noexcept {}
void ListmodeFileOutput::open(std::string path) {
  impl->open(path);
}
bool ListmodeFileOutput::appendSegment(const basic::Listmode_t *data, uint64_t count,
                                       uint64_t clock, uint32_t duration) noexcept {
  return impl->appendSegment(data, count, clock, duration);
}
void ListmodeFileOutput::setReservedBytes(uint64_t r) noexcept {
  impl->setReservedBytes(r);
}

bool ListmodeFileOutput::setBytes4CrystalIndex1(CrystalIndexType type) noexcept {
  return impl->setBytes4CrystalIndex1(type);
}
bool ListmodeFileOutput::setBytes4CrystalIndex2(CrystalIndexType type) noexcept {
  return impl->setBytes4CrystalIndex2(type);
}
bool ListmodeFileOutput::setBytes4TimeValue1_2(TimeValue1_2Type type) noexcept {
  return impl->setBytes4TimeValue1_2(type);
}
void ListmodeFileOutput::setTotalCrystalNum(uint32_t num) noexcept {
  impl->setTotalCrystalNum(num);
}
ListmodeFileInput::ListmodeFileInput() noexcept {
  impl = std::make_unique<ListmodeFileInput_impl>();
}
ListmodeFileInput::~ListmodeFileInput() noexcept {}
void ListmodeFileInput::open(std::string path) {
  impl->open(path);
}
uint32_t ListmodeFileInput::segmentNum() const noexcept {
  return impl->segmentNum();
}
listmode::ListmodeFileHeader ListmodeFileInput::header() const noexcept {
  return impl->header();
}
listmode::ListmodeSegmentHeader
ListmodeFileInput::segmentHeader(uint32_t segmentIndex) const noexcept {
  return impl->segmentHeader(segmentIndex);
}
listmode::ListmodeSegmentBytes
ListmodeFileInput::readSegment(uint32_t segmentIndex,
                               uint32_t prefetchIndex) const noexcept {
  return impl->readSegment(segmentIndex, prefetchIndex);
}

std::string RawFileInput::typeNameOfChannel(uint16_t channelIndex) const noexcept {
  return impl->typeNameOfChannel(channelIndex);
}

void RawFileOutput::setTypeNameOfChannel(uint16_t channelIndex,
                                         const std::string &typeName) noexcept {
  impl->setTypeNameOfChannel(channelIndex, typeName);
}
} // namespace openpni::io