// #include "BDM50100Array_impl.hpp"
// #include <iostream>
// #include <stdexcept>
// #include <fstream>

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
// #include <cuda_runtime.h>
// #include <cub/cub.cuh>
// #include "BDM50100_kernel.cuh"
// #include "BDM50100Array_kernel.cuh"

// #define CALL_AND_RETURN_IF_CUDA_ERR(func, err) \
//     if (err)                                   \
//     {                                          \
//         if (func)                              \
//         {                                      \
//             func("err at file: ");             \
//             func(std::string(__FILE__));       \
//             func("err at line: ");             \
//             func(std::to_string(__LINE__));    \
//             func("err string: ");              \
//             func(cudaGetErrorString(err));     \
//         }                                      \
//         return err;                            \
//     }

// #define CALL_IF(func, s) \
//     if (func)            \
//     {                    \
//         func(s);         \
//     }

// #define THREADNUM 128

// #define DEBUG_MODE false

// namespace openpni::device::deviceArray::bdm50100Array
// {
//     using namespace openpni::device::bdm50100;
//     using namespace openpni::device::bdm50100::algo;

//     struct energyEventSortOp
//     {
//         __device__ bool operator()(const caliCoef::EnergyEvent_t &a,
//                                    const caliCoef::EnergyEvent_t &b) const
//         {
//             if (index_process::toGlobalBlockId(a.bdmId, a.channelId) !=
//                 index_process::toGlobalBlockId(b.bdmId, b.channelId))
//             {
//                 return index_process::toGlobalBlockId(a.bdmId, a.channelId) <
//                        index_process::toGlobalBlockId(b.bdmId, b.channelId);
//             }
//             else
//             {
//                 return a.T0 < b.T0;
//             }
//         }
//     };

//     struct timeEventSortOp
//     {
//         __device__ bool operator()(const caliCoef::TimeEvent_t &a,
//                                    const caliCoef::TimeEvent_t &b) const
//         {
//             if (index_process::toGlobalBlockId(a.bdmId, a.channelId) !=
//                 index_process::toGlobalBlockId(b.bdmId, b.channelId))
//             {
//                 return index_process::toGlobalBlockId(a.bdmId, a.channelId) <
//                        index_process::toGlobalBlockId(b.bdmId, b.channelId);
//             }
//             else
//             {
//                 return a.T0 < b.T0;
//             }
//         }
//     };

//     struct rawSingleSortOp
//     {
//         __device__ bool operator()(const caliCoef::RawGlobalSingle50100_CUDA_t &a,
//                                    const caliCoef::RawGlobalSingle50100_CUDA_t &b) const
//         {
//             if (a.valid != b.valid)
//             {
//                 // 如果有效性不同，优先保留有效的
//                 if (a.valid == 0)
//                     return false; // a无效，b有效
//                 if (b.valid == 0)
//                     return true; // b无效，a有效
//             }
//             return a.timeValue_nano < b.timeValue_nano;
//         }
//     };

//     std::size_t BDM50100Array_impl::getBufferSizeNeeded(const uint64_t __packetCount,
//                                                         const uint64_t __bdmNum) noexcept
//     {
//         std::size_t totalBufferSize = 0;

//         std::size_t maxCubBufferSize = 0;
//         std::size_t cubBufferSizeTemp = 0;

//         // 1. 可能产生的最大能量事件和事件事件字节数, 等价于包的有效字节数
//         totalBufferSize += __packetCount * bdm50100::SINGLE_BYTES_PER_PACKET * 2;

//         // 2. 每个包的能量、时间个数统计，用于求前缀和
//         totalBufferSize += (__packetCount + 1) * sizeof(uint32_t) * 2;

//         // 3. CUB的临时缓冲区大小（前缀和）
//         {
//             uint32_t *ptr = nullptr;
//             cub::DeviceScan::ExclusiveSum(nullptr,
//                                           cubBufferSizeTemp,
//                                           ptr, // 输入指针
//                                           ptr, // 输出指针
//                                           __packetCount + 1,
//                                           cudaStreamDefault);

//             maxCubBufferSize = max(cubBufferSizeTemp, maxCubBufferSize);
//         }

//         // 4. CUB的临时缓冲区大小（能量、时间排序）
//         {
//             caliCoef::EnergyEvent_t *energyPtr = nullptr;
//             cub::DeviceMergeSort::StableSortKeys(nullptr,
//                                                  cubBufferSizeTemp,
//                                                  energyPtr, // 键指针
//                                                  bdm50100::MAX_ENERGY_EVENT_NUM_PER_PACKET * __packetCount,
//                                                  energyEventSortOp(),
//                                                  cudaStreamDefault);
//             maxCubBufferSize = max(cubBufferSizeTemp, maxCubBufferSize);
//         }

//         {
//             caliCoef::TimeEvent_t *timePtr = nullptr;
//             cub::DeviceMergeSort::StableSortKeys(nullptr,
//                                                  cubBufferSizeTemp,
//                                                  timePtr, // 键指针
//                                                  bdm50100::MAX_TIME_EVENT_NUM_PER_PACKET * __packetCount,
//                                                  timeEventSortOp(),
//                                                  cudaStreamDefault);
//             maxCubBufferSize = max(cubBufferSizeTemp, maxCubBufferSize);
//         }

//         // 5. block的时间能量个数统计值，用于求前缀和
//         totalBufferSize += (bdm50100::BLOCK_NUM * __bdmNum + 1) * 2 * sizeof(uint32_t);

//         // 6. cub前缀和
//         {
//             uint32_t *outputPtr = nullptr;
//             cub::DeviceScan::ExclusiveSum(nullptr,
//                                           cubBufferSizeTemp,
//                                           outputPtr, // 输入指针
//                                           outputPtr, // 输出指针
//                                           bdm50100::BLOCK_NUM * __bdmNum + 1,
//                                           cudaStreamDefault);
//         }

//         // 7. 单事件的数据大小
//         totalBufferSize += __packetCount *
//                            bdm50100::MAX_SINGLE_NUM_PER_PACKET *
//                            sizeof(caliCoef::RawGlobalSingle50100_CUDA_t);

//         // 8. CUB为单事件排序的临时缓冲区大小
//         {
//             caliCoef::RawGlobalSingle50100_CUDA_t *rawSinglePtr = nullptr;
//             cub::DeviceMergeSort::StableSortKeys(nullptr,
//                                                  cubBufferSizeTemp,
//                                                  rawSinglePtr, // 键指针
//                                                  __packetCount * bdm50100::MAX_SINGLE_NUM_PER_PACKET,
//                                                  rawSingleSortOp(),
//                                                  cudaStreamDefault);
//             maxCubBufferSize = max(cubBufferSizeTemp, maxCubBufferSize);
//         }

//         // 9. RAW单事件掩码
//         totalBufferSize += (__packetCount * bdm50100::MAX_SINGLE_NUM_PER_PACKET + 1) * sizeof(uint32_t);

//         // 10. CUB的临时缓冲区大小（单事件掩码）
//         {
//             uint32_t *ptr = nullptr;
//             cub::DeviceScan::ExclusiveSum(nullptr,
//                                           cubBufferSizeTemp,
//                                           ptr, // 输入指针
//                                           ptr, // 输出指针
//                                           __packetCount * bdm50100::MAX_SINGLE_NUM_PER_PACKET + 1,
//                                           cudaStreamDefault);

//             maxCubBufferSize = max(cubBufferSizeTemp, maxCubBufferSize);
//         }

//         // 11 总量
//         totalBufferSize += maxCubBufferSize;

//         return totalBufferSize;
//     }

//     cudaError_t BDM50100Array_impl::selectConvertRaw2Global(const caliCoef::RawGlobalSingle50100_CUDA_t
//     *d_rawGlobalSingle,
//                                                             basic::GlobalSingle_t *d_selectedSingle,
//                                                             const uint32_t *d_prefixedMask,
//                                                             const uint32_t rawGlobalSingleNum,
//                                                             const BDM50100ArrayR2SParams *__params,
//                                                             cudaStream_t __stream,
//                                                             std::function<void(std::string)> __callBackFunc)
//     {
//         deviceArray::DeviceModel model = __params->deviceModel;
//         const uint32_t blockNum = (rawGlobalSingleNum - 1) / THREADNUM + 1;

//         switch (model)
//         {
//         case deviceArray::DeviceModel::DIGITMI_930:
//         {
//             // 调用CUDA内核进行选择和转换
//             arrayAlgo::selectConvertRaw2Global_kernel<deviceArray::DeviceModel::DIGITMI_930>
//                 <<<blockNum, THREADNUM, 0, __stream>>>(d_rawGlobalSingle,
//                                                        d_selectedSingle,
//                                                        d_prefixedMask,
//                                                        rawGlobalSingleNum);
//             break;
//         }
//         default:
//         {
//             CALL_IF(__callBackFunc,
//                     "Unsupported device model for this version: " +
//                         std::to_string(static_cast<int>(model)));
//             return cudaErrorInvalidValue; // 不支持的设备型号
//             break;
//         }
//         }
//         cudaStreamSynchronize(__stream);
//         return cudaPeekAtLastError();
//     }

//     cudaError_t BDM50100Array_impl::selectConvertRaw2Rigional(const caliCoef::RawGlobalSingle50100_CUDA_t
//     *d_rawGlobalSingle,
//                                                               basic::RigionalSingle_t *d_selectedSingle,
//                                                               const uint32_t *d_prefixedMask,
//                                                               const uint32_t rawGlobalSingleNum,
//                                                               cudaStream_t __stream,
//                                                               std::function<void(std::string)> __callBackFunc)
//     {

//         const uint32_t blockNum = (rawGlobalSingleNum - 1) / THREADNUM + 1;

//         // 调用CUDA内核进行选择和转换
//         arrayAlgo::selectConvertRaw2Rigional_kernel<<<blockNum, THREADNUM, 0, __stream>>>(d_rawGlobalSingle,
//                                                                                           d_selectedSingle,
//                                                                                           d_prefixedMask,
//                                                                                           rawGlobalSingleNum);

//         cudaStreamSynchronize(__stream);
//         return cudaPeekAtLastError();
//     }

//     // 将探测器阵列加载到GPU
//     cudaError_t BDM50100Array_impl::loadDetectorToGpu(std::vector<DetectorChangable> &DetectorChangable,
//                                                       std::vector<std::string> &caliFiles,
//                                                       const int deviceId)
//     {
//         cudaError_t err = cudaSuccess;
//         int originalDeviceId = 0;

//         // 保存当前设备ID并切换到目标设备
//         err = cudaGetDevice(&originalDeviceId);
//         if (err != cudaSuccess)
//             return err;

//         err = cudaSetDevice(deviceId);
//         if (err != cudaSuccess)
//             return err;

//         try
//         {
//             // 首先在CPU上加载数据
//             if (!loadDetector(DetectorChangable, caliFiles))
//             {
//                 cudaSetDevice(originalDeviceId);
//                 return cudaErrorInvalidValue;
//             }

//             // 将每个探测器加载到GPU
//             for (size_t i = 0; i < m_detectorImpls.size(); ++i)
//             {
//                 err = m_detectorImpls[i]->loadDetectorToGpu(DetectorChangable[i], caliFiles[i], deviceId);
//                 if (err != cudaSuccess)
//                 {
//                     std::cerr << "Failed to load detector " << i << " to GPU: "
//                               << cudaGetErrorString(err) << std::endl;
//                     cudaSetDevice(originalDeviceId);
//                     return err;
//                 }
//             }

//             // 创建GPU上的校准数据视图数组
//             size_t numDetectors = m_detectorImpls.size();
//             if (numDetectors > 0)
//             {
//                 // 分配GPU内存存储各探测器的校准数据指针
//                 m_d_tdcEnergyCoefsView = basic::cuda::make_cuda_unique_ptr<caliCoef::TDCArray_S *>(numDetectors);
//                 m_d_tdcTimeCoefsView = basic::cuda::make_cuda_unique_ptr<caliCoef::TDCArray_S *>(numDetectors);
//                 m_d_energyCoefsView = basic::cuda::make_cuda_unique_ptr<caliCoef::EnergyCoef *>(numDetectors);
//                 m_d_timeCoefsView = basic::cuda::make_cuda_unique_ptr<caliCoef::TimeCoef *>(numDetectors);

//                 // 收集各探测器的GPU校准数据指针
//                 std::vector<caliCoef::TDCArray_S *> h_tdcEnergyCoefsViews(numDetectors);
//                 std::vector<caliCoef::TDCArray_S *> h_tdcTimeCoefsViews(numDetectors);
//                 std::vector<caliCoef::EnergyCoef *> h_energyCoefsViews(numDetectors);
//                 std::vector<caliCoef::TimeCoef *> h_timeCoefsViews(numDetectors);

//                 for (size_t i = 0; i < numDetectors; ++i)
//                 {
//                     auto caliView = m_detectorImpls[i]->getCaliView();
//                     h_tdcEnergyCoefsViews[i] = caliView.tdcEnergyCoefs;
//                     h_tdcTimeCoefsViews[i] = caliView.tdcTimeCoefs;
//                     h_energyCoefsViews[i] = caliView.energyCoefs;
//                     h_timeCoefsViews[i] = caliView.timeCoefs;
//                 }

//                 // 将指针数组复制到GPU
//                 err = cudaMemcpy(m_d_tdcEnergyCoefsView.get(),
//                                  h_tdcEnergyCoefsViews.data(),
//                                  numDetectors * sizeof(caliCoef::TDCArray_S *),
//                                  cudaMemcpyHostToDevice);
//                 if (err != cudaSuccess)
//                 {
//                     cudaSetDevice(originalDeviceId);
//                     return err;
//                 }

//                 err = cudaMemcpy(m_d_tdcTimeCoefsView.get(),
//                                  h_tdcTimeCoefsViews.data(),
//                                  numDetectors * sizeof(caliCoef::TDCArray_S *),
//                                  cudaMemcpyHostToDevice);
//                 if (err != cudaSuccess)
//                 {
//                     cudaSetDevice(originalDeviceId);
//                     return err;
//                 }

//                 err = cudaMemcpy(m_d_energyCoefsView.get(),
//                                  h_energyCoefsViews.data(),
//                                  numDetectors * sizeof(caliCoef::EnergyCoef *),
//                                  cudaMemcpyHostToDevice);
//                 if (err != cudaSuccess)
//                 {
//                     cudaSetDevice(originalDeviceId);
//                     return err;
//                 }

//                 err = cudaMemcpy(m_d_timeCoefsView.get(),
//                                  h_timeCoefsViews.data(),
//                                  numDetectors * sizeof(caliCoef::TimeCoef *),
//                                  cudaMemcpyHostToDevice);
//                 if (err != cudaSuccess)
//                 {
//                     cudaSetDevice(originalDeviceId);
//                     return err;
//                 }
//             }

//             m_isDetectorLoadedToGpu = true;
//             m_currentDeviceId = deviceId;
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "Exception in loadDetectorToGpu: " << e.what() << std::endl;
//             err = cudaErrorUnknown;
//         }

//         // 恢复原设备ID
//         cudaSetDevice(originalDeviceId);
//         return err;
//     }

//     // 检查探测器是否已加载到GPU
//     bool BDM50100Array_impl::isDetectorLoadedToGpu() const noexcept
//     {
//         return m_isDetectorLoadedToGpu && m_currentDeviceId >= 0;
//     }

//     // CUDA版本的原始数据到单事件转换
//     cudaError_t BDM50100Array_impl::r2s_cuda(const void *__d_raw,
//                                              const PacketPositionInfo *__d_position,
//                                              uint64_t __count,
//                                              basic::RigionalSingle_t *__d_out,
//                                              uint64_t &__outSinglesNum,
//                                              uint64_t &__bufferSize,
//                                              void *__d_buffer,
//                                              const BDM50100ArrayR2SParams *__params,
//                                              cudaStream_t __stream,
//                                              std::function<void(std::string)> __callBackFunc) const noexcept
//     {
//         if (!isDetectorLoadedToGpu())
//         {
//             CALL_IF(__callBackFunc, "Detector not loaded to GPU in r2s_cuda");
//             return cudaErrorNotReady;
//         }

//         cudaError_t err = cudaSuccess;
//         int originalDeviceId = 0;

//         // 切换到正确的设备
//         err = cudaGetDevice(&originalDeviceId);
//         if (err != cudaSuccess)
//             return err;

//         err = cudaSetDevice(m_currentDeviceId);
//         if (err != cudaSuccess)
//             return err;

//         // 0. 若__d_buffer为空，则返回缓冲区大小
//         if (__d_buffer == nullptr)
//         {
//             __bufferSize = getBufferSizeNeeded(__count, m_detectorImpls.size());
//             cudaSetDevice(originalDeviceId);
//             return cudaSuccess;
//         }

//         // 1. 检查输入参数
//         {
//             if (__d_raw == nullptr ||
//                 __d_position == nullptr ||
//                 __d_out == nullptr ||
//                 __count == 0 ||
//                 __params == nullptr)
//             {
//                 CALL_IF(__callBackFunc, "Invalid input parameters in r2s_cuda");
//                 cudaSetDevice(originalDeviceId);
//                 return cudaErrorInvalidValue;
//             }

//             std::size_t expectedBufferSize = getBufferSizeNeeded(__count, m_detectorImpls.size());
//             if (__bufferSize < expectedBufferSize)
//             {
//                 CALL_IF(__callBackFunc, "Insufficient buffer size in r2s_cuda");
//                 cudaSetDevice(originalDeviceId);
//                 return cudaErrorInvalidValue;
//             }
//         }

//         std::size_t cubBuffersize = __bufferSize;

//         // 2. 获得指针
//         caliCoef::EnergyEvent_t *d_energyEvents = nullptr;
//         caliCoef::TimeEvent_t *d_timeEvents = nullptr;
//         uint32_t *d_energyCount = nullptr;
//         uint32_t *d_timeCount = nullptr;
//         uint32_t *d_energyBlockCount = nullptr;
//         uint32_t *d_timeBlockCount = nullptr;
//         caliCoef::RawGlobalSingle50100_CUDA_t *d_rawSingles = nullptr;
//         uint32_t *d_rawSinglesMask = nullptr;
//         uint8_t *d_cubBuffer = nullptr;

//         {
//             // 根据getBufferSizeNeeded的内存布局进行指针偏移赋值
//             uint8_t *bufferPtr = static_cast<uint8_t *>(__d_buffer);
//             size_t offset = 0;
//             const size_t bdmNum = m_detectorImpls.size();

//             // 1. 能量事件数组 (最大能量事件数)
//             d_energyEvents = reinterpret_cast<caliCoef::EnergyEvent_t *>(bufferPtr + offset);
//             offset += __count * bdm50100::SINGLE_BYTES_PER_PACKET;

//             // 2. 时间事件数组 (最大时间事件数)
//             d_timeEvents = reinterpret_cast<caliCoef::TimeEvent_t *>(bufferPtr + offset);
//             offset += __count * bdm50100::SINGLE_BYTES_PER_PACKET;

//             // 3. 每个包的能量事件计数 (包含前缀和所需的+1项)
//             d_energyCount = reinterpret_cast<uint32_t *>(bufferPtr + offset);
//             offset += (__count + 1) * sizeof(uint32_t);

//             // 4. 每个包的时间事件计数
//             d_timeCount = reinterpret_cast<uint32_t *>(bufferPtr + offset);
//             offset += (__count + 1) * sizeof(uint32_t);

//             // 5. 每个block的能量和时间事件计数 (包含前缀和所需的+1项)
//             d_energyBlockCount = reinterpret_cast<uint32_t *>(bufferPtr + offset);
//             offset += (bdm50100::BLOCK_NUM * bdmNum + 1) * sizeof(uint32_t);

//             d_timeBlockCount = reinterpret_cast<uint32_t *>(bufferPtr + offset);
//             offset += (bdm50100::BLOCK_NUM * bdmNum + 1) * sizeof(uint32_t);

//             // 6. 原始单事件数组
//             d_rawSingles = reinterpret_cast<caliCoef::RawGlobalSingle50100_CUDA_t *>(bufferPtr + offset);
//             offset += __count * bdm50100::MAX_SINGLE_NUM_PER_PACKET * sizeof(caliCoef::RawGlobalSingle50100_CUDA_t);

//             // 7. 原始单事件掩码 (包含前缀和所需的+1项)
//             d_rawSinglesMask = reinterpret_cast<uint32_t *>(bufferPtr + offset);
//             offset += (__count * bdm50100::MAX_SINGLE_NUM_PER_PACKET + 1) * sizeof(uint32_t);

//             // 8. CUB临时缓冲区（放在最后，使用剩余空间）
//             d_cubBuffer = bufferPtr + offset;
//             cubBuffersize -= offset;
//         }

//         // 3. 统计每个包上的能量和时间事件个数
//         {
//             const int blockNum = (__count - 1) / THREADNUM + 1;

//             countGroupBDM50100_kernel<<<blockNum, THREADNUM, 0, __stream>>>(static_cast<const char *>(__d_raw),
//                                                                             __d_position,
//                                                                             __count,
//                                                                             d_energyCount,
//                                                                             d_timeCount);
//             cudaStreamSynchronize(__stream);
//             err = cudaPeekAtLastError();
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

// #if DEBUG_MODE
//             // ================================debug mode===========================================
//             std::unique_ptr<uint32_t[]> energyCountHost(new uint32_t[__count + 1]);
//             std::unique_ptr<uint32_t[]> timeCountHost(new uint32_t[__count + 1]);
//             cudaMemcpyAsync(energyCountHost.get(),
//                             d_energyCount,
//                             (__count + 1) * sizeof(uint32_t),
//                             cudaMemcpyDeviceToHost,
//                             __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//             cudaMemcpyAsync(timeCountHost.get(),
//                             d_timeCount,
//                             (__count + 1) * sizeof(uint32_t),
//                             cudaMemcpyDeviceToHost,
//                             __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             std::ofstream debugFile("/media/lenovo/1TB/a_new_envir/v4_coin/data/Z50100ArrayDeBug/energy_count.bin",
//             std::ios::binary);

//             if (debugFile.is_open())
//             {
//                 debugFile.write(reinterpret_cast<const char *>(energyCountHost.get()), (__count + 1) *
//                 sizeof(uint32_t)); debugFile.close();
//             }
//             else
//             {
//                 std::cerr << "Failed to open energy_count.bin for writing." << std::endl;
//             }

//             std::ofstream debugFile2("/media/lenovo/1TB/a_new_envir/v4_coin/data/Z50100ArrayDeBug/time_count.bin",
//             std::ios::binary); if (debugFile2.is_open())
//             {
//                 debugFile2.write(reinterpret_cast<const char *>(timeCountHost.get()), (__count + 1) *
//                 sizeof(uint32_t)); debugFile2.close();
//             }
//             else
//             {
//                 std::cerr << "Failed to open time_count.bin for writing." << std::endl;
//             }
//             uint64_t totalEnergyCount = 0;
//             uint64_t totalTimeCount = 0;

//             for (int i = 0; i < __count; i++)
//             {
//                 int size = energyCountHost[i] * sizeof(EnergyPacket50100) +
//                            timeCountHost[i] * sizeof(TimePacket50100);
//                 if (size != 1280)
//                 {
//                     std::cout << i << std::endl;
//                 }
//                 totalEnergyCount += energyCountHost[i];
//                 totalTimeCount += timeCountHost[i];
//             }
//             std::cout << "Total Energy Count: " << totalEnergyCount << std::endl;
//             std::cout << "Total Time Count: " << totalTimeCount << std::endl;
//             // ================================debug mode===========================================
// #endif
//         }

//         // 4. 前缀和能量和时间事件计数
//         basic::cuda::cuda_pinned_unique_ptr<uint32_t> energyEventTotalCount =
//             basic::cuda::make_cuda_pinned_unique_ptr<uint32_t>(1);
//         basic::cuda::cuda_pinned_unique_ptr<uint32_t> timeEventTotalCount =
//             basic::cuda::make_cuda_pinned_unique_ptr<uint32_t>(1);

//         {
//             err = cub::DeviceScan::ExclusiveSum(d_cubBuffer,
//                                                 cubBuffersize,
//                                                 d_energyCount,
//                                                 d_energyCount,
//                                                 __count + 1,
//                                                 __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             err = cudaMemcpyAsync(energyEventTotalCount.get(),
//                                   d_energyCount + __count,
//                                   sizeof(uint32_t),
//                                   cudaMemcpyDeviceToHost,
//                                   __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             err = cub::DeviceScan::ExclusiveSum(d_cubBuffer,
//                                                 cubBuffersize,
//                                                 d_timeCount,
//                                                 d_timeCount,
//                                                 __count + 1,
//                                                 __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             err = cudaMemcpyAsync(timeEventTotalCount.get(),
//                                   d_timeCount + __count,
//                                   sizeof(uint32_t),
//                                   cudaMemcpyDeviceToHost,
//                                   __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//         }
//         cudaStreamSynchronize(__stream);
//         CALL_IF(__callBackFunc, "Energy Event Total Count: " + std::to_string(energyEventTotalCount.at(0)) +
//                                     ", Time Event Total Count: " + std::to_string(timeEventTotalCount.at(0)));
// #if DEBUG_MODE
//         // =================debug mode===========================
//         std::cout << "Energy Event Total Count: " << energyEventTotalCount.at(0) << std::endl;
//         std::cout << "Time Event Total Count: " << timeEventTotalCount.at(0) << std::endl;
//         // =================debug mode===========================
// #endif

//         // 5. 拆解包
//         {
//             const int blockNum = (__count - 1) / THREADNUM + 1;

//             arrayAlgo::resolveGroupBDM50100Array_kernel<ResloveMode::complete>
//                 <<<blockNum, THREADNUM, 0, __stream>>>(static_cast<const char *>(__d_raw),
//                                                        __d_position,
//                                                        __count,
//                                                        m_d_tdcEnergyCoefsView.get(),
//                                                        m_d_tdcTimeCoefsView.get(),
//                                                        __params->energyThresholds,
//                                                        d_energyCount,
//                                                        d_energyEvents,
//                                                        d_timeCount,
//                                                        d_timeEvents);
//             cudaStreamSynchronize(__stream);
//             err = cudaPeekAtLastError();
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
// #if DEBUG_MODE
//             //================================debug mode================================

//             for (size_t i = 0; i < 13; ++i)
//             {
//                 std::cout << __params->energyThresholds.data[i] << " ";
//             }
//             std::cout << std::endl;

//             std::unique_ptr<caliCoef::EnergyEvent_t[]> energyEventsHost(new
//             caliCoef::EnergyEvent_t[energyEventTotalCount.at(0)]); std::unique_ptr<caliCoef::TimeEvent_t[]>
//             timeEventsHost(new caliCoef::TimeEvent_t[timeEventTotalCount.at(0)]);
//             cudaMemcpyAsync(energyEventsHost.get(),
//                             d_energyEvents,
//                             energyEventTotalCount.at(0) * sizeof(caliCoef::EnergyEvent_t),
//                             cudaMemcpyDeviceToHost,
//                             __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             cudaMemcpyAsync(timeEventsHost.get(),
//                             d_timeEvents,
//                             timeEventTotalCount.at(0) * sizeof(caliCoef::TimeEvent_t),
//                             cudaMemcpyDeviceToHost,
//                             __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//             std::ofstream debugFile("/media/lenovo/1TB/a_new_envir/v4_coin/data/Z50100ArrayDeBug/energy_events.bin",
//             std::ios::binary); if (debugFile.is_open())
//             {
//                 debugFile.write(reinterpret_cast<const char *>(energyEventsHost.get()),
//                                 energyEventTotalCount.at(0) * sizeof(caliCoef::EnergyEvent_t));
//                 debugFile.close();
//             }
//             else
//             {
//                 std::cerr << "Failed to open energy_events.bin for writing." << std::endl;
//             }

//             std::ofstream debugFile2("/media/lenovo/1TB/a_new_envir/v4_coin/data/Z50100ArrayDeBug/time_events.bin",
//             std::ios::binary); if (debugFile2.is_open())
//             {
//                 debugFile2.write(reinterpret_cast<const char *>(timeEventsHost.get()),
//                                  timeEventTotalCount.at(0) * sizeof(caliCoef::TimeEvent_t));
//                 debugFile2.close();
//             }
//             else
//             {
//                 std::cerr << "Failed to open time_events.bin for writing." << std::endl;
//             }
//             for (size_t i = 0; i < 10; i++)
//             {
//                 std::cout << "Energy Event: BDM ID: " << energyEventsHost[i].bdmId
//                           << ", Channel ID: " << energyEventsHost[i].channelId
//                           << ", T0: " << energyEventsHost[i].T0
//                           << ", Energy: " << energyEventsHost[i].Energy << std::endl;
//             }

//             for (size_t i = 0; i < 10; i++)
//             {
//                 std::cout << "Time Event: BDM ID: " << timeEventsHost[i].bdmId
//                           << ", Channel ID: " << timeEventsHost[i].channelId
//                           << ", T0: " << timeEventsHost[i].T0
//                           << ", Time: " << timeEventsHost[i].deltaT << std::endl;
//             }

//             // ================================debug mode================================
// #endif
//         }

//         // 6. count每个block上的能量和时间事件个数
//         {
//             err = cudaMemsetAsync(d_energyBlockCount,
//                                   0,
//                                   (bdm50100::BLOCK_NUM * m_detectorImpls.size() + 1) * sizeof(uint32_t),
//                                   __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             err = cudaMemsetAsync(d_timeBlockCount,
//                                   0,
//                                   (bdm50100::BLOCK_NUM * m_detectorImpls.size() + 1) * sizeof(uint32_t),
//                                   __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             const int eventNum = max(energyEventTotalCount.at(0), timeEventTotalCount.at(0));
//             const int blockNum = (eventNum - 1) / THREADNUM + 1;
//             arrayAlgo::countGlobalBlockEvent_kernel<<<blockNum, THREADNUM, 0, __stream>>>(d_energyEvents,
//                                                                                           d_timeEvents,
//                                                                                           d_energyBlockCount,
//                                                                                           d_timeBlockCount,
//                                                                                           energyEventTotalCount.at(0),
//                                                                                           timeEventTotalCount.at(0));
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//         }

//         // 7. 将能量和时间事件排序
//         {
//             err = cub::DeviceMergeSort::StableSortKeys(d_cubBuffer,
//                                                        cubBuffersize,
//                                                        d_energyEvents,
//                                                        energyEventTotalCount.at(0),
//                                                        energyEventSortOp(),
//                                                        __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             err = cub::DeviceMergeSort::StableSortKeys(d_cubBuffer,
//                                                        cubBuffersize,
//                                                        d_timeEvents,
//                                                        timeEventTotalCount.at(0),
//                                                        timeEventSortOp(),
//                                                        __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
// #if DEBUG_MODE
//             // =================debug mode================================
//             std::unique_ptr<caliCoef::EnergyEvent_t[]> energyEventsHost(new
//             caliCoef::EnergyEvent_t[energyEventTotalCount.at(0)]); std::unique_ptr<caliCoef::TimeEvent_t[]>
//             timeEventsHost(new caliCoef::TimeEvent_t[timeEventTotalCount.at(0)]); err =
//             cudaMemcpyAsync(energyEventsHost.get(),
//                                   d_energyEvents,
//                                   energyEventTotalCount.at(0) * sizeof(caliCoef::EnergyEvent_t),
//                                   cudaMemcpyDeviceToHost,
//                                   __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             err = cudaMemcpyAsync(timeEventsHost.get(),
//                                   d_timeEvents,
//                                   timeEventTotalCount.at(0) * sizeof(caliCoef::TimeEvent_t),
//                                   cudaMemcpyDeviceToHost,
//                                   __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             for (size_t i = 0; i < 10; i++)
//             {
//                 std::cout << "Sorted Energy Event: BDM ID: " << energyEventsHost[i].bdmId
//                           << ", Channel ID: " << energyEventsHost[i].channelId
//                           << ", T0: " << energyEventsHost[i].T0
//                           << ", Energy: " << energyEventsHost[i].Energy << std::endl;
//             }

//             for (size_t i = 0; i < 10; i++)
//             {
//                 std::cout << "Sorted Time Event: BDM ID: " << timeEventsHost[i].bdmId
//                           << ", Channel ID: " << timeEventsHost[i].channelId
//                           << ", T0: " << timeEventsHost[i].T0
//                           << ", Time: " << timeEventsHost[i].deltaT << std::endl;
//             }
//             // =================debug mode================================
// #endif
//         }

//         // 8. 将block统计求前缀和，得到偏移
//         {
//             err = cub::DeviceScan::ExclusiveSum(d_cubBuffer,
//                                                 cubBuffersize,
//                                                 d_energyBlockCount,
//                                                 d_energyBlockCount,
//                                                 bdm50100::BLOCK_NUM * m_detectorImpls.size() + 1,
//                                                 __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             err = cub::DeviceScan::ExclusiveSum(d_cubBuffer,
//                                                 cubBuffersize,
//                                                 d_timeBlockCount,
//                                                 d_timeBlockCount,
//                                                 bdm50100::BLOCK_NUM * m_detectorImpls.size() + 1,
//                                                 __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

// #if DEBUG_MODE
//             // =================debug mode================================
//             uint32_t energyBlockSum;
//             uint32_t timeBlockSum;
//             err = cudaMemcpyAsync(&energyBlockSum,
//                                   d_energyBlockCount + bdm50100::BLOCK_NUM * m_detectorImpls.size(),
//                                   sizeof(uint32_t),
//                                   cudaMemcpyDeviceToHost,
//                                   __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//             err = cudaMemcpyAsync(&timeBlockSum,
//                                   d_timeBlockCount + bdm50100::BLOCK_NUM * m_detectorImpls.size(),
//                                   sizeof(uint32_t),
//                                   cudaMemcpyDeviceToHost,
//                                   __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//             std::cout << "Energy Block Total Count: " << energyBlockSum << std::endl;
//             std::cout << "Time Block Total Count: " << timeBlockSum << std::endl;
//             // =================debug mode================================
// #endif
//         }

//         // 9. 将单事件转换
//         {
//             const int blockNum = bdm50100::BLOCK_NUM * m_detectorImpls.size();
//             const int threadNum = THREADNUM;
//             arrayAlgo::eventMatchArray_Kernel<<<blockNum, threadNum, 0, __stream>>>(
//                 d_energyEvents,
//                 d_timeEvents,
//                 d_energyBlockCount,
//                 d_timeBlockCount,
//                 __params->matchXTalkEnabled,
//                 __params->timeWindow,
//                 __params->timeShift,
//                 m_d_energyCoefsView.get(),
//                 m_d_timeCoefsView.get(),
//                 d_rawSingles);
//             cudaStreamSynchronize(__stream);
//             err = cudaPeekAtLastError();
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//         }

//         // 10. 单事件时间排序
//         {
//             // 能量事件数 == 单事件数
//             err = cub::DeviceMergeSort::StableSortKeys(d_cubBuffer,
//                                                        cubBuffersize,
//                                                        d_rawSingles,
//                                                        energyEventTotalCount.at(0),
//                                                        rawSingleSortOp(),
//                                                        __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

// #if DEBUG_MODE
//             // =================debug mode================================
//             std::unique_ptr<caliCoef::RawGlobalSingle50100_CUDA_t[]> rawSinglesHost(
//                 new caliCoef::RawGlobalSingle50100_CUDA_t[energyEventTotalCount.at(0)]);
//             err = cudaMemcpyAsync(rawSinglesHost.get(),
//                                   d_rawSingles,
//                                   energyEventTotalCount.at(0) * sizeof(caliCoef::RawGlobalSingle50100_CUDA_t),
//                                   cudaMemcpyDeviceToHost,
//                                   __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//             int count = 0;
//             for (size_t i = 0; i < energyEventTotalCount.at(0); i++)
//             {
//                 if (rawSinglesHost[i].valid)
//                 {
//                     std::cout << "Raw Single " << i << ": "
//                               << "timeValue_nano: " << rawSinglesHost[i].timeValue_nano
//                               << ", energy: " << rawSinglesHost[i].energy
//                               << ", bdmId: " << rawSinglesHost[i].bdmId
//                               << ", cryId: " << rawSinglesHost[i].cryId
//                               << ", valid: " << (int)rawSinglesHost[i].valid
//                               << ", matchCount: " << (int)rawSinglesHost[i].matchCount
//                               << ", isXtalk: " << (int)rawSinglesHost[i].isXtalk
//                               << ", xtalkCount: " << (int)rawSinglesHost[i].xtalkCount
//                               << std::endl;
//                     count++;
//                     if (count >= 10)
//                     {
//                         break; // 只打印前10个
//                     }
//                 }
//             }
//             count = 0;
//             for (size_t i = 0; i < energyEventTotalCount.at(0); i++)
//             {
//                 if (rawSinglesHost[i].valid)
//                 {
//                     count++;
//                 }
//             }

//             std::cout << "Total valid singles: " << count << std::endl;

//             // =================debug mode================================
// #endif
//         }

//         // 11. xtalk校正
//         if (__params->crossTalkEnabled)
//         {
//             // 暂时写法，后续可修改为模板函数调用
//             const int searchRange = 32;
//             const int threadGroupSearchRange = searchRange * THREADNUM;
//             const int blockNum = (energyEventTotalCount.at(0) - 1) / threadGroupSearchRange + 1;

//             arrayAlgo::xtalkCorrection_kernel<<<blockNum, THREADNUM, 0, __stream>>>(
//                 d_rawSingles,
//                 m_d_energyCoefsView.get(),
//                 __params->crossTalkTimeWindow,
//                 searchRange,
//                 energyEventTotalCount.at(0));
//             err = cudaPeekAtLastError();
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//         }

//         // 12. 利用掩码筛选单事件
//         {
//             const int blockNum = (energyEventTotalCount.at(0) - 1) / THREADNUM + 1;

//             algo::maskingValidRawSingle_kernel<<<blockNum, THREADNUM, 0, __stream>>>(
//                 d_rawSingles,
//                 d_rawSinglesMask,
//                 energyEventTotalCount.at(0));
//             cudaStreamSynchronize(__stream);
//             err = cudaPeekAtLastError();
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             err = cub::DeviceScan::ExclusiveSum(d_cubBuffer,
//                                                 cubBuffersize,
//                                                 d_rawSinglesMask,
//                                                 d_rawSinglesMask,
//                                                 energyEventTotalCount.at(0) + 1,
//                                                 __stream);
//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);

//             err = selectConvertRaw2Rigional(d_rawSingles,
//                                             __d_out,
//                                             d_rawSinglesMask,
//                                             energyEventTotalCount.at(0),
//                                             __stream,
//                                             __callBackFunc);

//             cudaStreamSynchronize(__stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//         }

//         // 13. 设置输出参数
//         {
//             basic::cuda::cuda_pinned_unique_ptr<uint32_t> singlesNumPtr =
//                 basic::cuda::make_cuda_pinned_unique_ptr<uint32_t>(1);
//             err = cudaMemcpyAsync(singlesNumPtr.get(),
//                                   d_rawSinglesMask + energyEventTotalCount.at(0),
//                                   sizeof(uint32_t),
//                                   cudaMemcpyDeviceToHost,
//                                   __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__callBackFunc, err);
//             cudaStreamSynchronize(__stream);
//             __outSinglesNum = singlesNumPtr.at(0);
//         }

//         CALL_IF(__callBackFunc, "r2s_cuda completed successfully. Singles Num: " + std::to_string(__outSinglesNum));

//         // 恢复原设备ID
//         cudaSetDevice(originalDeviceId);
//         return err;
//     }

//     void BDM50100Array_impl::test()
//     {
//         // 测试函数，实际使用中可以删除或替换为其他测试逻辑
//         std::cout << "BDM50100Array_impl test function called." << std::endl;
//         std::cout << "for 930 system with 48 bdm and 1024 * 1024 pactet" << std::endl;
//         std::cout << "getBufferSizeNeeded: " << getBufferSizeNeeded(1024 * 1024, 48) << " bytes" << std::endl;
//         std::cout << "is GB" << (getBufferSizeNeeded(1024 * 1024, 48) / (1024.0 * 1024.0 * 1024.0)) << " GB" <<
//         std::endl;
//     }
// }

// #undef CALL_AND_RETURN_IF_CUDA_ERR
// #undef THREADNUM
// #undef CALL_IF
// #undef DEBUG_MODE

// #endif
