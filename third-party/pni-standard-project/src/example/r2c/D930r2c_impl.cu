// #include "include/example/D930r2c_impl.cuh"
// #include "include/example/Bdm50100ArrayLayout.cuh"
// #include "include/r2cHandle/r2cTools.cuh"

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

// #define CALL_IF(func, msg) \
//     if (func)              \
//     {                      \
//         func(msg);         \
//     }
// #define CRYMAP_LAYER 4
// #define DEBUG_MODE false

// #if DEBUG_MODE
// #include <fstream>
// #endif

// namespace openpni::example::r2c_example::D930
// {
//     using namespace openpni::basic;
//     using namespace openpni::device;
//     using namespace openpni::device::bdm50100;
//     using namespace openpni::device::deviceArray;
//     using namespace openpni::process::s2c_any;

//     static __global__ void convertRigionalToGlobal_kernel(const basic::RigionalSingle_t *d_rigionalSingles,
//                                                           basic::GlobalSingle_t *d_globalSingles,
//                                                           const unsigned singlesNum)
//     {
//         unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
//         if (idx < singlesNum)
//         {
//             d_globalSingles[idx].globalCrystalIndex =
//                 bdm50100Array::layout::D930::getGlobalCryId(d_rigionalSingles[idx].bdmIndex,
//                                                             d_rigionalSingles[idx].crystalIndex);
//             d_globalSingles[idx].timeValue_pico = d_rigionalSingles[idx].timeValue_pico;
//             d_globalSingles[idx].energy = d_rigionalSingles[idx].energy;
//         }
//     }

//     // 使用BDM50100中的常量
//     using openpni::device::bdm50100::MAX_SINGLE_NUM_PER_PACKET;

//     struct isGoodPairOp
//     {
//         uint32_t BDM_ONE_RING_NUM;
//         uint32_t RANGE;
//         __host__ __device__ __forceinline__ isGoodPairOp() {}

//         __host__ __device__ __forceinline__ isGoodPairOp(uint32_t bdm_one_ring_num,
//                                                          uint32_t range)
//             : BDM_ONE_RING_NUM(bdm_one_ring_num), RANGE(range) {}

//         __host__ __device__ __forceinline__ bool operator()(const basic::RigionalSingle_t &a,
//                                                             const basic::RigionalSingle_t &b) const
//         {
//             int32_t bdmIdInRingA = a.bdmIndex % BDM_ONE_RING_NUM;
//             int32_t bdmIdInRingB = b.bdmIndex % BDM_ONE_RING_NUM;
//             int32_t def = abs(bdmIdInRingA - bdmIdInRingB);

//             if ((def < RANGE) || (def > (BDM_ONE_RING_NUM - RANGE)))
//             {
//                 return false;
//             }

//             return true;
//         }
//     };

//     struct singleCryMapOp
//     {
//         __host__ __device__ __forceinline__ uint32_t operator()(const basic::RigionalSingle_t &a) const
//         {
//             return a.bdmIndex * openpni::device::bdm50100::CRYSTAL_NUM + a.crystalIndex;
//         }
//     };

//     // 构造函数
//     D930r2c_impl::D930r2c_impl() : m_maxFrameNum(0),
//                                    m_deviceId(0),
//                                    m_isAvailable(false),
//                                    m_bufferSize(0),
//                                    m_validSinglesNum(0)
//     {
//     }

//     // 析构函数
//     D930r2c_impl::~D930r2c_impl()
//     {
//     }

//     // 初始化函数
//     cudaError_t D930r2c_impl::init(const uint64_t maxFrameNum,
//                                    const uint32_t deviceId,
//                                    std::vector<DetectorChangable> &DetectorChangable,
//                                    std::vector<std::string> &caliFiles,
//                                    std::function<void(std::string)> logFunc)
//     {
//         cudaError_t err = cudaSuccess;
//         // 重置可用状态
//         m_isAvailable = false;

//         // 保存参数
//         m_maxFrameNum = static_cast<uint32_t>(maxFrameNum);
//         m_deviceId = deviceId;

//         // 设置CUDA设备
//         err = cudaSetDevice(deviceId);
//         CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

//         try
//         {
//             // 初始化BDM阵列
//             m_d_bdmArray.loadDetectorToGpu(DetectorChangable, caliFiles, deviceId);
//             CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

//             // 分配GPU内存缓冲区
//             {
//                 size_t bufferSize = 0;
//                 BDM50100ArrayR2SParams r2sParams;
//                 uint64_t maxSinglesPerFrame = MAX_SINGLE_NUM_PER_PACKET * maxFrameNum; // 临时数量
//                 cudaError_t err = m_d_bdmArray.r2s_cuda(m_d_dataFrame.get(),           // d_raw
//                                                         m_d_packetPositionInfo.get(),  // d_packetPositionInfo
//                                                         maxFrameNum,                   //
//                                                         m_d_rigionalSingle.get(),
//                                                         maxSinglesPerFrame,
//                                                         bufferSize,
//                                                         nullptr,
//                                                         &r2sParams,
//                                                         cudaStreamDefault, // streamId
//                                                         logFunc);
//                 cudaDeviceSynchronize();
//                 CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);
//                 m_bufferSize = bufferSize;
//                 m_d_buffer = cuda::make_cuda_unique_ptr<char>(m_bufferSize);
//                 CALL_IF(logFunc, "Buffer Size: " + std::to_string(m_bufferSize));
//             }

//             // 分配数据帧内存
//             m_d_dataFrame = cuda::make_cuda_unique_ptr<DataFrame50100_t>(maxFrameNum);
//             m_d_packetPositionInfo = cuda::make_cuda_unique_ptr<PacketPositionInfo>(maxFrameNum);

//             // 分配单事件内存
//             {
//                 const size_t maxSinglesPerFrame = MAX_SINGLE_NUM_PER_PACKET * maxFrameNum; // 临时数量
//                 m_d_rigionalSingle = cuda::make_cuda_unique_ptr<basic::RigionalSingle_t>(maxSinglesPerFrame);
//             }

//             // 分配晶体映射表内存
//             {
//                 int temp = constants::RING_NUM *
//                            constants::BDM_PER_RING *
//                            bdm50100::CRYSTAL_NUM *
//                            CRYMAP_LAYER;
//                 m_d_cryMap = cuda::make_cuda_unique_ptr<unsigned int>(temp);
//             }

//             // 分配符合事件内存
//             {
//                 const uint32_t maxPromptNumExcepted =
//                 float(openpni::process::s2c_any::constants::MAX_ENERGY_RATE_EXPECTED *
//                                                             openpni::process::s2c_any::constants::MAX_PROMPT_COIN_RATE_EXPECTED
//                                                             * MAX_SINGLE_NUM_PER_PACKET * maxFrameNum) +
//                                                       1;

//                 const uint32_t maxDelayNumExcepted =
//                 float(openpni::process::s2c_any::constants::MAX_ENERGY_RATE_EXPECTED *
//                                                            openpni::process::s2c_any::constants::MAX_DELAY_COIN_RATE_EXPECTED
//                                                            * MAX_SINGLE_NUM_PER_PACKET * maxFrameNum) +
//                                                      1;
//                 CALL_IF(logFunc, "Max Prompt Coin Expected: " + std::to_string(maxPromptNumExcepted));
//                 CALL_IF(logFunc, "Max Delay Coin Expected: " + std::to_string(maxDelayNumExcepted));

//                 m_d_promptCoinSingles = cuda::make_cuda_unique_ptr<basic::RigionalSingle_t>(maxPromptNumExcepted);
//                 m_d_delayCoinSingles = cuda::make_cuda_unique_ptr<basic::RigionalSingle_t>(maxDelayNumExcepted);

//                 m_d_promptListmode = cuda::make_cuda_unique_ptr<basic::Listmode_t>(maxPromptNumExcepted / 2 + 1);
//                 m_d_delayListmode = cuda::make_cuda_unique_ptr<basic::Listmode_t>(maxDelayNumExcepted / 2 + 1);
//             }
//         }
//         catch (const std::exception &e)
//         {
//             CALL_IF(logFunc, std::string("Exception: ") + e.what());
//             return cudaErrorMemoryAllocation;
//         }

//         // 设置可用状态
//         m_isAvailable = true;
//         return cudaSuccess;
//     }

//     // 同步执行函数
//     cudaError_t D930r2c_impl::excSync(const DataFrame50100_t *h_p_dataFrame,
//                                       const PacketPositionInfo *h_p_packetPositionInfo,
//                                       const uint32_t frameNum,
//                                       const D930r2cParams_t params,
//                                       basic::Listmode_t *h_p_d_promptListmode,
//                                       basic::Listmode_t *h_p_d_delayListmode,
//                                       uint32_t &promptNum,
//                                       uint32_t &delayNum,
//                                       cudaStream_t stream,
//                                       std::function<void(std::string)> logFunc)
//     {
//         cudaError_t err = cudaSuccess;

//         // 设置CUDA设备
//         err = cudaSetDevice(m_deviceId);
//         CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

//         // 检查输入参数
//         if (!h_p_dataFrame || !h_p_packetPositionInfo || frameNum == 0 || frameNum > m_maxFrameNum)
//         {
//             CALL_IF(logFunc, "Invalid input parameters");
//             return cudaErrorInvalidValue;
//         }

//         // 复制输入数据到GPU
//         err = cudaMemcpyAsync(m_d_dataFrame.get(),
//                               h_p_dataFrame,
//                               frameNum * sizeof(DataFrame50100_t),
//                               cudaMemcpyHostToDevice,
//                               stream);
//         CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

//         err = cudaMemcpyAsync(m_d_packetPositionInfo.get(),
//                               h_p_packetPositionInfo,
//                               frameNum * sizeof(PacketPositionInfo),
//                               cudaMemcpyHostToDevice,
//                               stream);
//         CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

//         // 步骤1: Raw to Singles (R2S)
//         uint64_t singlesNum = 0;

//         {
//             uint64_t bufferSize = m_bufferSize;
//             BDM50100ArrayR2SParams r2sParams;
//             // 从D930参数中初始化R2S参数
//             r2sParams.crossTalkEnabled = params.crossTalkEnabled;
//             r2sParams.energyThresholds = params.energyThresholds;
//             r2sParams.matchXTalkEnabled = params.matchXTalkEnabled;
//             r2sParams.timeWindow = params.timeWindow;
//             r2sParams.timeShift = params.timeShift;
//             r2sParams.crossTalkTimeWindow = params.crossTalkTimeWindow;
//             r2sParams.deviceModel = params.deviceModel;

//             err = m_d_bdmArray.r2s_cuda(m_d_dataFrame.get(),
//                                         m_d_packetPositionInfo.get(),
//                                         frameNum,
//                                         m_d_rigionalSingle.get(),
//                                         singlesNum,
//                                         bufferSize,
//                                         m_d_buffer.get(),
//                                         &r2sParams,
//                                         stream,
//                                         logFunc);
//             cudaStreamSynchronize(stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

// #if DEBUG_MODE
//             std::cout << "R2S completed, singlesNum: " << singlesNum << std::endl;
//             std::unique_ptr<basic::RigionalSingle_t[]> h_rawGlobalSingle =
//                 std::make_unique<basic::RigionalSingle_t[]>(singlesNum);
//             err = cudaMemcpyAsync(h_rawGlobalSingle.get(),
//                                   m_d_rigionalSingle.get(),
//                                   singlesNum * sizeof(basic::RigionalSingle_t),
//                                   cudaMemcpyDeviceToHost,
//                                   stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);
//             cudaStreamSynchronize(stream);
//             std::ofstream ofs("/media/cmx/K1/v4_coin/data/Z50100ArrayDeBugNew/singlesR.bin", std::ios::binary);
//             if (ofs.is_open())
//             {
//                 ofs.write(reinterpret_cast<const char *>(h_rawGlobalSingle.get()),
//                           singlesNum * sizeof(basic::RigionalSingle_t));
//                 ofs.close();
//             }
//             else
//             {
//                 CALL_IF(logFunc, "Failed to open output file for writing singles data");
//             }
// #endif
//         }

//         // 记录R2S结果
//         m_validSinglesNum = static_cast<uint32_t>(singlesNum);

//         // 步骤2: Singles to Coincidences (S2C)
//         unsigned promptCoinSinglesNum = 0;
//         unsigned delayCoinSinglesNum = 0;

//         // 使用s2cHandle进行符合处理
//         {
//             isGoodPairOp coinOperator(constants::BDM_PER_RING, params.range);
//             int cryMapSize = constants::RING_NUM * constants::BDM_PER_RING * bdm50100::CRYSTAL_NUM;
//             uint64_t bufferSize = m_bufferSize;
//             err = m_d_s2cHandle.exc<basic::RigionalSingle_t, isGoodPairOp, singleCryMapOp, true, false>(
//                 m_d_rigionalSingle.get(),
//                 m_d_promptCoinSingles.get(),
//                 m_d_delayCoinSingles.get(),
//                 m_d_cryMap.get(),
//                 params.energyLowKev,
//                 params.energyHighKev,
//                 params.timeWindowPicosec,
//                 params.delayTimePicosec,
//                 cryMapSize,
//                 singlesNum,
//                 promptCoinSinglesNum,
//                 delayCoinSinglesNum,
//                 bufferSize,
//                 m_d_buffer.get(),
//                 coinOperator,
//                 singleCryMapOp(),
//                 m_deviceId,
//                 stream,
//                 logFunc);
//             cudaStreamSynchronize(stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

//             {
//                 basic::GlobalSingle_t *d_globalSingles =
//                     reinterpret_cast<basic::GlobalSingle_t *>(m_d_buffer.get());

//                 // 调用转换函数
//                 err = convertRigionalToGlobal(m_d_promptCoinSingles.get(),
//                                               d_globalSingles,
//                                               promptCoinSinglesNum * 2,
//                                               stream,
//                                               logFunc);
//                 CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

//                 err = openpni::process::r2cTools::compressCoin(d_globalSingles,
//                                                                m_d_promptListmode.get(),
//                                                                promptCoinSinglesNum,
//                                                                0,
//                                                                stream);
//                 cudaStreamSynchronize(stream);
//                 CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);
//             }

//             {
//                 basic::GlobalSingle_t *d_globalSingles =
//                     reinterpret_cast<basic::GlobalSingle_t *>(m_d_buffer.get());

//                 // 调用转换函数
//                 err = convertRigionalToGlobal(m_d_delayCoinSingles.get(),
//                                               d_globalSingles,
//                                               delayCoinSinglesNum * 2,
//                                               stream,
//                                               logFunc);
//                 CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

//                 err = openpni::process::r2cTools::compressCoin(d_globalSingles,
//                                                                m_d_delayListmode.get(),
//                                                                delayCoinSinglesNum,
//                                                                params.delayTimePicosec,
//                                                                stream);
//                 cudaStreamSynchronize(stream);
//                 CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);
//             }

//             // err = s2cHandle::listModeCompress(m_d_delayCoinSingles.get(),
//             //                                   m_d_delayListmode.get(),
//             //                                   delayCoinSinglesNum,
//             //                                   params.delayTimePicosec,
//             //                                   m_deviceId,
//             //                                   stream,
//             //                                   logFunc);
//             // cudaStreamSynchronize(stream);
//             // CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);
//         }

//         // 复制结果到主机
//         promptNum = promptCoinSinglesNum;
//         delayNum = delayCoinSinglesNum;
//         if (h_p_d_promptListmode && h_p_d_delayListmode)
//         {
//             err = cudaMemcpyAsync(h_p_d_promptListmode,
//                                   m_d_promptListmode.get(),
//                                   promptCoinSinglesNum * sizeof(basic::Listmode_t),
//                                   cudaMemcpyDeviceToHost,
//                                   stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);

//             err = cudaMemcpyAsync(h_p_d_delayListmode,
//                                   m_d_delayListmode.get(),
//                                   delayCoinSinglesNum * sizeof(basic::Listmode_t),
//                                   cudaMemcpyDeviceToHost,
//                                   stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(logFunc, err);
//         }

//         cudaStreamSynchronize(stream);
//         return cudaSuccess;
//     }

//     cudaError_t D930r2c_impl::getLastValidSingleAsync(basic::GlobalSingle_t *__h_p_d_globalSingles,
//                                                       uint32_t &__singlesNum,
//                                                       cudaStream_t __stream,
//                                                       std::function<void(std::string)> __logFunc)
//     {
//         cudaError_t err = cudaSuccess;
//         if (!__h_p_d_globalSingles)
//         {
//             CALL_IF(__logFunc, "Invalid input parameters for getLastValidSingleAsync");
//             return cudaErrorInvalidValue;
//         }

//         if (!m_isAvailable)
//         {
//             CALL_IF(__logFunc, "D930r2c_impl is not available");
//             return cudaErrorNotReady;
//         }

//         int originDeviceId = 0;
//         // 获取当前CUDA设备ID
//         err = cudaGetDevice(&originDeviceId);
//         CALL_AND_RETURN_IF_CUDA_ERR(__logFunc, err);

//         // 设置CUDA设备
//         err = cudaSetDevice(m_deviceId);
//         CALL_AND_RETURN_IF_CUDA_ERR(__logFunc, err);

//         if (!__h_p_d_globalSingles)
//         {
//             CALL_IF(__logFunc, "Invalid input parameters for getLastValidSingleAsync");
//             return cudaErrorInvalidValue;
//         }

//         if (m_validSinglesNum == 0)
//         {
//             CALL_IF(__logFunc, "No valid singles found");
//             return cudaSuccess; // 没有有效的单事件，直接返回成功
//         }

//         {
//             size_t singlesSize = m_validSinglesNum * sizeof(basic::GlobalSingle_t);
//             if (m_bufferSize < singlesSize)
//             {
//                 CALL_IF(__logFunc, "Buffer size is not enough for singles data");
//                 return cudaErrorInvalidValue; // 缓冲区大小不足
//             }
//             basic::GlobalSingle_t *d_globalSingles = reinterpret_cast<basic::GlobalSingle_t *>(m_d_buffer.get());

//             // 调用转换函数
//             err = convertRigionalToGlobal(m_d_rigionalSingle.get(),
//                                           d_globalSingles,
//                                           m_validSinglesNum,
//                                           __stream,
//                                           __logFunc);
//             CALL_AND_RETURN_IF_CUDA_ERR(__logFunc, err);

//             // 从GPU复制数据到主机
//             err = cudaMemcpyAsync(__h_p_d_globalSingles,
//                                   d_globalSingles,
//                                   m_validSinglesNum * sizeof(basic::GlobalSingle_t),
//                                   cudaMemcpyDeviceToHost,
//                                   __stream);
//             CALL_AND_RETURN_IF_CUDA_ERR(__logFunc, err);
//             __singlesNum = m_validSinglesNum;
//         }

//         // 同步流
//         err = cudaStreamSynchronize(__stream);
//         CALL_AND_RETURN_IF_CUDA_ERR(__logFunc, err);

//         // 恢复原设备
//         err = cudaSetDevice(originDeviceId);
//         CALL_AND_RETURN_IF_CUDA_ERR(__logFunc, err);

//         return cudaSuccess;
//     }

//     // 获取晶体映射表
//     cudaError_t D930r2c_impl::getCryMap(std::vector<unsigned> &cryMap)
//     {
//         std::lock_guard<std::mutex> lock(m_mutex);

//         cudaError_t err = cudaSuccess;

//         // 设置CUDA设备
//         err = cudaSetDevice(m_deviceId);
//         if (err != cudaSuccess)
//             return err;

//         if (!m_d_cryMap)
//         {
//             return cudaErrorInvalidValue;
//         }

//         // 计算晶体映射表大小
//         const size_t cryMapSize = constants::RING_NUM *
//                                   constants::BDM_PER_RING *
//                                   bdm50100::CRYSTAL_NUM *
//                                   CRYMAP_LAYER;

//         // 调整输出vector的大小
//         cryMap.resize(cryMapSize);

//         // 从GPU复制数据到主机
//         err = cudaMemcpy(cryMap.data(),
//                          m_d_cryMap.get(),
//                          cryMapSize * sizeof(unsigned),
//                          cudaMemcpyDeviceToHost);

//         if (err != cudaSuccess)
//         {
//             cryMap.clear();
//             return err;
//         }

//         return cudaSuccess;
//     }

//     // 重置晶体映射表
//     cudaError_t D930r2c_impl::resetCryMap()
//     {
//         std::lock_guard<std::mutex> lock(m_mutex);

//         cudaError_t err = cudaSuccess;

//         // 设置CUDA设备
//         err = cudaSetDevice(m_deviceId);
//         if (err != cudaSuccess)
//             return err;

//         if (!m_d_cryMap)
//         {
//             return cudaErrorInvalidValue;
//         }

//         // 计算晶体映射表大小
//         const size_t cryMapSize = constants::RING_NUM *
//                                   constants::BDM_PER_RING *
//                                   bdm50100::CRYSTAL_NUM *
//                                   CRYMAP_LAYER;

//         // 将GPU内存清零
//         err = cudaMemset(m_d_cryMap.get(), 0, cryMapSize * sizeof(unsigned));
//         if (err != cudaSuccess)
//             return err;

//         return cudaSuccess;
//     }

//     // 检查是否可用
//     bool D930r2c_impl::isAvailable() const noexcept
//     {
//         return m_isAvailable;
//     }

//     // 加锁
//     void D930r2c_impl::lock() noexcept
//     {
//         m_mutex.lock();
//     }

//     // 尝试加锁
//     bool D930r2c_impl::tryLock() noexcept
//     {
//         return m_mutex.try_lock();
//     }

//     // 解锁
//     void D930r2c_impl::unlock() noexcept
//     {
//         m_mutex.unlock();
//     }

//     cudaError_t D930r2c_impl::convertRigionalToGlobal(const basic::RigionalSingle_t *__d_rigionalSingles,
//                                                       basic::GlobalSingle_t *__d_globalSingles,
//                                                       const uint32_t __singlesNum,
//                                                       cudaStream_t __stream,
//                                                       std::function<void(std::string)> __logFunc)
//     {
//         const int blockSize = 256;
//         const int numBlocks = (__singlesNum - 1) / blockSize + 1;

//         // 设置CUDA设备
//         cudaError_t err = cudaSetDevice(m_deviceId);
//         CALL_AND_RETURN_IF_CUDA_ERR(__logFunc, err);

//         // 调用CUDA内核函数
//         convertRigionalToGlobal_kernel<<<numBlocks, blockSize, 0, __stream>>>(__d_rigionalSingles,
//                                                                               __d_globalSingles,
//                                                                               __singlesNum);

//         // 检查CUDA内核执行错误
//         cudaStreamSynchronize(__stream);
//         err = cudaPeekAtLastError();
//         CALL_AND_RETURN_IF_CUDA_ERR(__logFunc, err);

//         return cudaSuccess;
//     }
// }

// #undef CALL_AND_RETURN_IF_CUDA_ERR
// #undef CALL_IF
// #undef DEBUG_MODE
// #undef CRYMAP_LAYER