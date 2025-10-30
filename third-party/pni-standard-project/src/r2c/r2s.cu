#include <cub/device/device_reduce.cuh>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/remove.h>

#include "include/detector/Detectors.hpp"
#include "include/process/ListmodeProcessing.hpp"
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
using PacketPositionInfo = openpni::device::PacketPositionInfo;
namespace openpni::process {

template <typename T, typename V>
__global__ void kernel_vector_add(
    const T *d_src, T *d_dst, uint64_t size, V number) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    d_dst[idx] = d_src[idx] + number;
  }
}
template <typename T, typename V>
__global__ void kernel_vector_sub(
    const T *d_src, T *d_dst, uint64_t size, V number) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    d_dst[idx] = d_src[idx] - number;
  }
}

template <typename T>
struct OpEqual {
  __host__ __device__ bool operator()(
      const T &value) const {
    return _value == value;
  }
  T _value;
};

template <typename T>
struct OpNotEqual {
  __host__ __device__ bool operator()(
      const T &value) const {
    return _value != value;
  }
  T _value;
};

struct OpSameChannel {
  __host__ __device__ bool operator()(
      const PacketPositionInfo &value) const {
    return _channel == value.channel;
  }
  uint16_t _channel;
};

__global__ void kernel_copy_packetPosition(
    PacketPositionInfo *d_dst, const uint64_t *d_offset, const uint16_t *d_length, const uint16_t *d_channel,
    uint64_t num) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    d_dst[idx].offset = d_offset[idx] - d_offset[0];
    d_dst[idx].length = d_length[idx];
    d_dst[idx].channel = d_channel[idx];
  }
}

struct OpValidLocalSingle {
  __host__ __device__ bool operator()(
      const basic::LocalSingle_t &value) const {
    return !(value.crystalIndex == 0xffff);
  }
};
LocalSinglesOfEachChannel rtos(
    process::RawDataView d_rawdata, device::DetectorBase *const *h_detectors) {
  using PacketPositionInfoOfEachChannel = std::vector<cuda_sync_ptr<PacketPositionInfo>>;
  PacketPositionInfoOfEachChannel info(d_rawdata.channelNum);
  if (d_rawdata.count == 0)
    return std::vector<cuda_sync_ptr<basic::LocalSingle_t>>(d_rawdata.channelNum);

  // Calculate the number of packets in each channel
  {
    thrust::device_vector<PacketPositionInfo> d_packetPosition(d_rawdata.count);
    kernel_copy_packetPosition<<<(d_rawdata.count + 255) / 256, 256>>>(
        d_packetPosition.data().get(), d_rawdata.offset, d_rawdata.length, d_rawdata.channel, d_rawdata.count);
    for (uint16_t i = 0; i < d_rawdata.channelNum; ++i) {
      const auto begin = d_packetPosition.begin();
      const auto end = thrust::partition(begin, d_packetPosition.end(), OpSameChannel{i});
      info[i] = make_cuda_sync_ptr_from_dcopy<PacketPositionInfo>(
          std::span<const PacketPositionInfo>(thrust::raw_pointer_cast(&*begin), thrust::raw_pointer_cast(&*end)));
    }
  }

  // Convert 2 singles for each channel
  std::vector<cuda_sync_ptr<basic::LocalSingle_t>> result(d_rawdata.channelNum);
  std::vector<thrust::device_vector<basic::LocalSingle_t>> temp_singles(d_rawdata.channelNum);
  for (uint16_t i = 0; i < d_rawdata.channelNum; ++i) {
    const auto &p_runtime = h_detectors[i];
    temp_singles[i].resize(info[i].elements() * p_runtime->detectorUnchangable().maxSingleNumPerPacket);
    p_runtime->r2s_cuda(d_rawdata.data, info[i].data(), info[i].elements(), temp_singles[i].data().get());

    const auto begin = temp_singles[i].begin();
    const auto end = thrust::partition(begin, temp_singles[i].end(), OpValidLocalSingle());
    result[i] = make_cuda_sync_ptr_from_dcopy(
        std::span<const basic::LocalSingle_t>(thrust::raw_pointer_cast(&*begin), thrust::raw_pointer_cast(&*end)));
  }

  return result;
}
} // namespace openpni::process
#endif