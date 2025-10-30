#pragma once
#include <fstream>

#include "../basic/CudaPtr.hpp"
#include "../basic/Image.hpp"
#include "../example/ConvolutionKernel.hpp"
#include "../io/IO.hpp"
#include "../misc/CycledBuffer.hpp"
#include "../misc/ListmodeBuffer.hpp"
#include "../process/EM.cuh"
#include "../process/Scatter.hpp"
#include "PolygonalSystem.hpp"
namespace openpni::example {
inline auto generatePairs(
    std::size_t __max, std::size_t __count) -> std::vector<std::pair<std::size_t, std::size_t>> {
  std::vector<std::size_t> indices;
  indices.reserve(__count + 1); // 性能优化
  std::vector<std::pair<std::size_t, std::size_t>> result;
  result.reserve(__count); // 性能优化

  for (std::size_t i = 0; i < __count; i++)
    indices.push_back(i * __max / __count);
  indices.push_back(__max);

  for (std::size_t i = 0; i < __count; i++)
    result.push_back(std::make_pair(indices[i], indices[i + 1]));

  return result;
};
template <typename T>
inline bool readFile(
    const std::string &path, std::vector<T> &Data, int offset = 0) {
  std::ifstream FileInput(path, std::ios_base::binary);
  if (!FileInput.is_open()) {
    std::cout << "Cannot open file " + path + " !";
    return false;
  }
  // 文件指针移到文件尾
  FileInput.seekg(0, std::ios::end);
  // 获取文件指针的位置，此时就相当于文件大小了
  size_t filesize = FileInput.tellg();
  filesize -= offset;
  // 文件指针移到文件开头+offset字节
  FileInput.seekg(offset, std::ios::beg);
  if (filesize != size_t(Data.size()) * sizeof(T)) {
    std::cout << "file size length mismatch! at " << path << std::endl;
    std::cout << "flilesize: " << filesize * 2 << "     Data size:" << Data.size() << std::endl;
    FileInput.close();
    return false;
  }
  FileInput.read((char *)Data.data(), filesize);
  FileInput.close();
  return true;
}

inline basic::Image3DGeometry defaultImgGeometry(
    int imageWidth = 320, int imageHeight = 320, int imageDepth = 400) {
  openpni::basic::Vec3<float> imageLegnth{160.f, 160.f, 200.f};
  auto voxelSize =
      openpni::basic::Vec3<float>{imageLegnth.x / imageWidth, imageLegnth.y / imageHeight, imageLegnth.z / imageDepth};
  auto imgBegin = openpni::basic::Vec3<float>{-imageLegnth.x / 2, -imageLegnth.y / 2, -imageLegnth.z / 2};
  return basic::Image3DGeometry{voxelSize, imgBegin, {imageWidth, imageHeight, imageDepth}};
}

inline basic::Image3DGeometry make_ImgGeometry(
    basic::Vec3<int> ImageVoxelNum, basic::Vec3<float> ImageVoxelSize) // FOV = ImageVoxelNum*ImageVoxelSize
{
  auto FOV = ImageVoxelNum * ImageVoxelSize;
  auto imgBegin = openpni::basic::Vec3<float>{-FOV.x / 2, -FOV.y / 2, -FOV.z / 2};
  return basic::Image3DGeometry{ImageVoxelSize, imgBegin, ImageVoxelNum};
}
inline std::vector<openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float>>
createMichDataSets(
    float *mich, polygon::PolygonModel &model, int subsetNum, int binCut) {
  auto indexr = model.michSubsetIndexers(subsetNum, binCut);
  std::vector<openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float>> MichDataViews(
      subsetNum);
  for (const int subsetId : std::views::iota(0, subsetNum)) {
    MichDataViews[subsetId].qtyValue = mich;
    MichDataViews[subsetId].crystalGeometry = model.crystalGeometry().data();
    MichDataViews[subsetId].indexer.scanner = model.polygonSystem();
    MichDataViews[subsetId].indexer.detector = model.detectorInfo().geometry;
    MichDataViews[subsetId].indexer.subsetId = subsetId;
    MichDataViews[subsetId].indexer.subsetNum = subsetNum;
    MichDataViews[subsetId].indexer.binCut = 0; // No
  }
  return MichDataViews;
}
inline std::vector<openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float>>
createMichDataSets_CUDA(
    float *mich, polygon::PolygonModel &model, const int subsetNum, const int binCut) {
  std::size_t michSize = model.michSize();
  if (mich == nullptr)
    michSize = 0;
  auto d_mich = make_cuda_sync_ptr_from_hcopy(std::span<const float>{mich, michSize});
  // auto d_mich = make_cuda_sync_ptr_from_hcopy<float>(std::span{mich, michSize});
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(model.crystalGeometry());
  std::vector<openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float>> MichDataViews(
      subsetNum);
  for (const int subsetId : std::views::iota(0, subsetNum)) {
    MichDataViews[subsetId].qtyValue = d_mich.get();
    MichDataViews[subsetId].crystalGeometry = d_crystalGeometry;
    MichDataViews[subsetId].indexer.scanner = model.polygonSystem();
    MichDataViews[subsetId].indexer.detector = model.detectorInfo().geometry;
    MichDataViews[subsetId].indexer.subsetId = subsetId;
    MichDataViews[subsetId].indexer.subsetNum = subsetNum;
    MichDataViews[subsetId].indexer.binCut = binCut; // No
  }
  return MichDataViews;
}
inline std::vector<openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float>>
createMichDataSets(
    float *mich, basic::CrystalGeometry *crystalGeometry, polygon::PolygonModel &model, const int subsetNum,
    const int binCut) {
  std::size_t michSize = model.michSize();
  if (mich == nullptr)
    michSize = 0;
  std::vector<openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float>> MichDataViews(
      subsetNum);
  for (const int subsetId : std::views::iota(0, subsetNum)) {
    MichDataViews[subsetId].qtyValue = mich;
    MichDataViews[subsetId].crystalGeometry = crystalGeometry;
    MichDataViews[subsetId].indexer.scanner = model.polygonSystem();
    MichDataViews[subsetId].indexer.detector = model.detectorInfo().geometry;
    MichDataViews[subsetId].indexer.subsetId = subsetId;
    MichDataViews[subsetId].indexer.subsetNum = subsetNum;
    MichDataViews[subsetId].indexer.binCut = binCut; // No
  }
  return MichDataViews;
}
inline void createSenmap_CUDA(
    std::vector<openpni::cuda_sync_ptr<float>> &d_senmaps, polygon::PolygonModel &model,
    const basic::Image3DGeometry &imgGeo, int subsetNum, int binCut, float *Corr_Mul_mich = nullptr) {
  using michDefaultIndexer = openpni::example::polygon::IndexerOfSubsetForMich;
  auto d_convolutionKernel = make_cuda_sync_ptr_from_hcopy(example::gaussianKernel<9>(1.5f));

  // auto senMapDataViews = createMichDataSets_CUDA(Corr_Mul_mich, model, subsetNum, 0);
  // test
  std::size_t michSize = model.michSize();
  if (Corr_Mul_mich == nullptr)
    michSize = 0;
  auto d_mich = make_cuda_sync_ptr_from_hcopy(std::span<const float>{Corr_Mul_mich, michSize});
  // auto d_mich = make_cuda_sync_ptr_from_hcopy<float>(std::span{mich, michSize});
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(model.crystalGeometry());
  std::vector<openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float>> MichDataViews(
      subsetNum);
  for (const int subsetId : std::views::iota(0, subsetNum)) {
    MichDataViews[subsetId].qtyValue = d_mich.get();
    MichDataViews[subsetId].crystalGeometry = d_crystalGeometry;
    MichDataViews[subsetId].indexer.scanner = model.polygonSystem();
    MichDataViews[subsetId].indexer.detector = model.detectorInfo().geometry;
    MichDataViews[subsetId].indexer.subsetId = subsetId;
    MichDataViews[subsetId].indexer.subsetNum = subsetNum;
    MichDataViews[subsetId].indexer.binCut = binCut; // No
  }
  //
  for (const auto dataView : MichDataViews) {
    d_senmaps.push_back(openpni::make_cuda_sync_ptr<float>(imgGeo.totalVoxelNum()));
    auto projectionMethod = openpni::math::ProjectionMethodUniform();
    projectionMethod.sampler.setSampleRatio(2.5f);
    openpni::process::calSenmap_CUDA(dataView, imgGeo, d_senmaps.back().get(), d_convolutionKernel.data(),
                                     projectionMethod);
    openpni::process::fixSenmap_simple_CUDA(d_senmaps.back().get(), imgGeo, 0.05f);
  }
}
inline void forwardProjection_CUDA(
    float *out_fwdMich, float *in_ImgData, example::polygon::PolygonModel &model, int bincut = 0) {
  using michDefaultIndexer = openpni::example::polygon::IndexerOfSubsetForMich;
  basic::Image3DGeometry imgGeo{{0.5f, 0.5f, 0.5f}, {-80.f, -80.f, -100.f}, {320, 320, 400}};

  auto d_in_ImgData = make_cuda_sync_ptr_from_hcopy<float>(std::span{in_ImgData, imgGeo.totalVoxelNum()});
  auto d_out_fwdMich = make_cuda_sync_ptr<float>(model.michSize());
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(model.crystalGeometry());

  Image3DIOSpan<float> IO_d_Image3dSpan{imgGeo, d_in_ImgData.data(), d_out_fwdMich.data()};

  std::cout << "prepare dataView" << std::endl;
  openpni::basic::DataViewQTY<michDefaultIndexer, float> dataView;
  dataView.qtyValue = nullptr;
  dataView.crystalGeometry = d_crystalGeometry;
  dataView.indexer.scanner = model.polygonSystem();
  dataView.indexer.detector = model.detectorInfo().geometry;
  dataView.indexer.subsetId = 0;
  dataView.indexer.subsetNum = 1;
  dataView.indexer.binCut = bincut;
  process::EMSum_CUDA(IO_d_Image3dSpan, IO_d_Image3dSpan.geometry.roi(), dataView, math::ProjectionMethodSiddon());
  d_out_fwdMich.allocator().copy_from_device_to_host(out_fwdMich, d_out_fwdMich.cspan());
}
inline void backwardProjection_CUDA(
    float *out_ImgData, float *in_fwdMich, example::polygon::PolygonModel &model, int bincut = 0) {
  std::cout << "backwardProjection_CUDA" << std::endl;
  basic::Image3DGeometry imgGeo{{0.5f, 0.5f, 0.5f}, {-80.f, -80.f, -100.f}, {320, 320, 400}};

  auto d_in_fwdMich = make_cuda_sync_ptr_from_hcopy<float>(std::span{in_fwdMich, model.michSize()});
  auto d_out_ImgData = make_cuda_sync_ptr<float>(imgGeo.totalVoxelNum());
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(model.crystalGeometry());
  openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float> dataView;
  dataView.qtyValue = d_in_fwdMich;
  dataView.crystalGeometry = d_crystalGeometry;
  dataView.indexer.scanner = model.polygonSystem();
  dataView.indexer.detector = model.detectorInfo().geometry;
  dataView.indexer.subsetId = 0;
  dataView.indexer.subsetNum = 1;
  dataView.indexer.binCut = bincut;

  process::EMDistribute_CUDA(d_in_fwdMich.data(), d_out_ImgData.data(), imgGeo, imgGeo.roi(), dataView,
                             math::ProjectionMethodSiddon());
  d_out_ImgData.allocator().copy_from_device_to_host(out_ImgData, d_out_ImgData.cspan());
}

struct OSEM_params {
  basic::Vec3<int> OSEMImgVoxelNum = {320, 320, 400};
  basic::Vec3<float> OSEMImgVoxelSize = {0.5, 0.5, 0.5};
  int subsetNum = 12;
  int iterNum = 4;
  int binCut = 0;
  float hfwhm = 0.33f;
  double tofBinWidth = 0;
  int tofBinNum = 0;
};
inline void OSEM_CUDA(
    float *out_OSEMImg, float *in_mich, OSEM_params params, polygon::PolygonModel &model,
    float *Corr_Add_mich = nullptr, float *Corr_Mul_mich = nullptr) {

  using michDefaultIndexer = openpni::example::polygon::IndexerOfSubsetForMich;
  using multType = openpni::basic::_FactorAdaptorMich<openpni::basic::FactorType::Multiply, michDefaultIndexer>;
  using addType = openpni::basic::_FactorAdaptorMich<openpni::basic::FactorType::Addition, michDefaultIndexer>;
  using EMSumUpdatorType = openpni::process::EMSumUpdator_CUDA<multType, addType>;

  // imgGeo
  auto OSEMimg = make_ImgGeometry(params.OSEMImgVoxelNum, params.OSEMImgVoxelSize);
  // cal senmap
  std::vector<openpni::cuda_sync_ptr<float>> d_senmaps;
  createSenmap_CUDA(d_senmaps, model, OSEMimg, params.subsetNum, params.binCut, Corr_Mul_mich);
  auto d_senmapPtr = d_senmaps | std::views::transform([](const auto &ptr) { return ptr.get(); });
  auto d_senmapPtrVectors = std::vector<float *>(d_senmapPtr.begin(), d_senmapPtr.end());

  // OSEM
  std::cout << "OSEM_CUDA start" << std::endl;
  auto d_convolutionKernel = make_cuda_sync_ptr_from_hcopy(example::gaussianKernel<9>(1.5f));
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(model.crystalGeometry());
  auto d_out_Img3D = openpni::make_cuda_sync_ptr<float>(OSEMimg.totalVoxelNum());
  auto d_michData = make_cuda_sync_ptr_from_hcopy(std::span<const float>{in_mich, model.michSize()});
  std::vector<openpni::basic::DataViewQTY<michDefaultIndexer, float>> dataSetsForRecon(params.subsetNum);
  for (const int subsetId : std::views::iota(0, params.subsetNum)) {
    dataSetsForRecon[subsetId].qtyValue = d_michData.get();
    dataSetsForRecon[subsetId].crystalGeometry = d_crystalGeometry;
    dataSetsForRecon[subsetId].indexer.scanner = model.polygonSystem();
    dataSetsForRecon[subsetId].indexer.detector = model.detectorInfo().geometry;
    dataSetsForRecon[subsetId].indexer.subsetId = subsetId;
    dataSetsForRecon[subsetId].indexer.subsetNum = params.subsetNum;
    dataSetsForRecon[subsetId].indexer.binCut = params.binCut;
  }
  auto d_Corr_Add_mich = Corr_Add_mich
                             ? make_cuda_sync_ptr_from_hcopy(std::span<const float>{Corr_Add_mich, model.michSize()})
                             : cuda_sync_ptr<float>{}; // 空的 cuda_sync_ptr

  auto d_Corr_Mul_mich = Corr_Mul_mich
                             ? make_cuda_sync_ptr_from_hcopy(std::span<const float>{Corr_Mul_mich, model.michSize()})
                             : cuda_sync_ptr<float>{};
  std::vector<multType> d_multiFctAdptor(params.subsetNum);
  std::vector<addType> d_addFctAdptor(params.subsetNum);
  std::vector<EMSumUpdatorType> EMSumUpdatorSubsets(params.subsetNum);
  for (const int subsetId : std::views::iota(0, params.subsetNum)) {
    d_multiFctAdptor[subsetId].factor = d_Corr_Mul_mich.get(); // or nullPtr
    d_multiFctAdptor[subsetId].indexer = michDefaultIndexer();
    d_multiFctAdptor[subsetId].indexer.scanner = model.polygonSystem();
    d_multiFctAdptor[subsetId].indexer.detector = model.detectorInfo().geometry;
    d_multiFctAdptor[subsetId].indexer.subsetId = subsetId;
    d_addFctAdptor[subsetId].factor = d_Corr_Add_mich.get(); // or nullPtr
    d_addFctAdptor[subsetId].indexer = michDefaultIndexer();
    d_addFctAdptor[subsetId].indexer.scanner = model.polygonSystem();
    d_addFctAdptor[subsetId].indexer.detector = model.detectorInfo().geometry;
    d_addFctAdptor[subsetId].indexer.subsetId = subsetId;
    EMSumUpdatorSubsets[subsetId].additionFactorAdapter = d_addFctAdptor[subsetId];
    EMSumUpdatorSubsets[subsetId].multiplyFactorAdapter = d_multiFctAdptor[subsetId];
  }

  auto projectionMethod = openpni::math::ProjectionMethodUniform();
  projectionMethod.sampler.setSampleRatio(1.0f);

  openpni::cuda_sync_ptr<char> d_buffer;
  std::size_t bufferSize = 0;

  while (!openpni::process::SEM_simple_CUDA(dataSetsForRecon, OSEMimg, d_out_Img3D.get(), d_convolutionKernel.get(),
                                            d_senmapPtrVectors, 4, d_buffer.get(), bufferSize, projectionMethod,
                                            EMSumUpdatorSubsets, openpni::process::ImageSimpleUpdate_CUDA())) {
    bufferSize += 20 * 1024 * 1024;
    std::cout << "Resize buffer to " << bufferSize << " bytes." << std::endl;
    d_buffer = openpni::make_cuda_sync_ptr<char>(bufferSize);
  }
  std::cout << "OSEM_CUDA done" << std::endl;
  d_out_Img3D.allocator().copy_from_device_to_host(out_OSEMImg, d_out_Img3D.cspan());
}

inline void OSEM_listMode_CUDA(
    float *out_osem, OSEM_params params, std::string listmode_path, polygon::PolygonModel &model,
    unsigned long long size_GB, float *Corr_Add_mich = nullptr, float *Corr_Mul_mich = nullptr) {
  auto lortest = openpni::example::polygon::calLORIDFromCrystalUniformID(model.polygonSystem(),
                                                                         model.detectorInfo().geometry, 220, 221);

  auto OSEMimg = make_ImgGeometry(params.OSEMImgVoxelNum, params.OSEMImgVoxelSize);
  // cal senmap here only generate one senmap
  std::vector<openpni::cuda_sync_ptr<float>> d_senmaps;
  createSenmap_CUDA(d_senmaps, model, OSEMimg, 1, params.binCut, Corr_Mul_mich);
  thrust::transform(thrust::device_pointer_cast(d_senmaps[0].get()),
                    thrust::device_pointer_cast(d_senmaps[0].get() + OSEMimg.totalVoxelNum()),
                    thrust::device_pointer_cast(d_senmaps[0].get()),
                    [params] __device__(float a) { return a / params.subsetNum; });
  const auto vecSenmaps = std::vector<float *>(params.subsetNum, d_senmaps[0].get());
  std::cout << "Senmap done,do OSEM......" << std::endl;
  // OSEM
  // read listmode
  openpni::io::ListmodeFileInput listmodeFile;
  listmodeFile.open(listmode_path);
  std::cout << "prepare data" << std::endl;
  // prepare
  openpni::misc::ListmodeBuffer listmodeBuffer;
  openpni::cuda_sync_ptr<openpni::basic::Listmode_t> d_bufferForListmode;
  openpni::cuda_sync_ptr<char> d_buffer;
  std::size_t bufferSize = 0;
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(model.crystalGeometry());
  auto d_tmp_Img3D = openpni::make_cuda_sync_ptr<float>(OSEMimg.totalVoxelNum());
  auto d_out_Img3D = openpni::make_cuda_sync_ptr<float>(OSEMimg.totalVoxelNum());
  auto d_convolutionKernel = make_cuda_sync_ptr_from_hcopy(example::gaussianKernel<9>(1.5f));
  auto projectionMethod = openpni::math::ProjectionMethodUniform();
  projectionMethod.sampler.setSampleRatio(0.7f);
  std::cout << "prepare factor adaptor" << std::endl;
  using michDefaultIndexer = openpni::example::polygon::IndexerOfSubsetForMich;
  using multType = openpni::basic::_FactorAdaptorMich<openpni::basic::FactorType::Multiply, michDefaultIndexer>;
  using addType = openpni::basic::_FactorAdaptorMich<openpni::basic::FactorType::Addition, michDefaultIndexer>;
  using EMSumUpdatorType = openpni::process::EMSumUpdator_CUDA<multType, addType>;
  std::vector<multType> d_multiFctAdptor(params.subsetNum);
  std::vector<addType> d_addFctAdptor(params.subsetNum);
  std::vector<EMSumUpdatorType> EMSumUpdatorSubsets(params.subsetNum);
  auto d_Corr_Add_mich = Corr_Add_mich
                             ? make_cuda_sync_ptr_from_hcopy(std::span<const float>{Corr_Add_mich, model.michSize()})
                             : cuda_sync_ptr<float>{}; // 空的 cuda_sync_ptr

  auto d_Corr_Mul_mich = Corr_Mul_mich
                             ? make_cuda_sync_ptr_from_hcopy(std::span<const float>{Corr_Mul_mich, model.michSize()})
                             : cuda_sync_ptr<float>{};
  std::cout << "prepare ...." << std::endl;
  for (const int subsetId : std::views::iota(0, params.subsetNum)) {
    d_multiFctAdptor[subsetId].factor = d_Corr_Mul_mich.get(); // or nullPtr
    d_multiFctAdptor[subsetId].indexer = michDefaultIndexer();
    d_multiFctAdptor[subsetId].indexer.scanner = model.polygonSystem();
    d_multiFctAdptor[subsetId].indexer.detector = model.detectorInfo().geometry;
    d_multiFctAdptor[subsetId].indexer.subsetId = subsetId;
    d_addFctAdptor[subsetId].factor = d_Corr_Add_mich.get(); // or nullPtr
    d_addFctAdptor[subsetId].indexer = michDefaultIndexer();
    d_addFctAdptor[subsetId].indexer.scanner = model.polygonSystem();
    d_addFctAdptor[subsetId].indexer.detector = model.detectorInfo().geometry;
    d_addFctAdptor[subsetId].indexer.subsetId = subsetId;
    EMSumUpdatorSubsets[subsetId].additionFactorAdapter = d_addFctAdptor[subsetId];
    EMSumUpdatorSubsets[subsetId].multiplyFactorAdapter = d_multiFctAdptor[subsetId];
  }
  std::cout << "prepare listmode buffer" << std::endl;
  auto GBSize = [](unsigned long long size) -> uint64_t { return size * 1024 * 1024 * 1024; };
  listmodeBuffer.setBufferSize(GBSize(size_GB) / sizeof(openpni::basic::Listmode_t))
      .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
        if (d_bufferForListmode.elements() < __count)
          d_bufferForListmode = openpni::make_cuda_sync_ptr<openpni::basic::Listmode_t>(__count);
        cudaMemcpy(d_bufferForListmode.get(), __data, __count * sizeof(openpni::basic::Listmode_t),
                   cudaMemcpyHostToDevice);
        std::cout << "copy listmode to device done, count = " << __count << std::endl;
        const auto pairs = generatePairs(__count, params.subsetNum);
        std::cout << "prepare dataView" << std::endl;
        std::vector<openpni::basic::DataViewListmodePlain> dataSetsForRecon(params.subsetNum);
        for (const auto subsetIndex : std::views::iota(0, params.subsetNum)) {
          auto &dataView = dataSetsForRecon[subsetIndex];
          dataView.count = pairs[subsetIndex].second - pairs[subsetIndex].first;
          dataView.crystalGeometry = d_crystalGeometry;
          dataView.listmodes = d_bufferForListmode.get() + pairs[subsetIndex].first;
        }
        std::cout << "OSEM iter start." << std::endl;
        while (!openpni::process::SEM_simple_CUDA(
            dataSetsForRecon, OSEMimg, d_tmp_Img3D.data(), d_convolutionKernel.get(), vecSenmaps, 1, d_buffer.get(),
            bufferSize, projectionMethod, EMSumUpdatorSubsets, openpni::process::ImageSimpleUpdate_CUDA())) {
          bufferSize += 20 * 1024 * 1024;
          std::cout << "Resize buffer to " << bufferSize << " bytes." << std::endl;
          d_buffer = openpni::make_cuda_sync_ptr<char>(bufferSize);
        }
        thrust::transform(
            thrust::device, thrust::device_pointer_cast(d_out_Img3D.begin()),
            thrust::device_pointer_cast(d_out_Img3D.end()), thrust::device_pointer_cast(d_tmp_Img3D.begin()),
            thrust::device_pointer_cast(d_out_Img3D.begin()), [] __device__(float a, float b) { return a + b; });
        static int iter = 0;
        std::cout << "OSEM done. iter = " << iter++ << std::endl;
      })
      .append(listmodeFile, openpni::io::selectSegments(listmodeFile))
      .flush();

  d_out_Img3D.allocator().copy_from_device_to_host(out_osem, d_out_Img3D.cspan());
}

inline void OSEM_listModeTOF_CUDA(
    float *out_osem, OSEM_params params, std::string listmode_path, polygon::PolygonModel &model,
    unsigned long long size_GB, float *Corr_Add_mich = nullptr, float *Corr_Mul_mich = nullptr,
    float *sssValues = nullptr) {
  auto OSEMimg = make_ImgGeometry(params.OSEMImgVoxelNum, params.OSEMImgVoxelSize);
  // cal senmap here only generate one senmap
  std::vector<openpni::cuda_sync_ptr<float>> d_senmaps;
  createSenmap_CUDA(d_senmaps, model, OSEMimg, 1, params.binCut, Corr_Mul_mich);
  thrust::transform(thrust::device_pointer_cast(d_senmaps[0].get()),
                    thrust::device_pointer_cast(d_senmaps[0].get() + OSEMimg.totalVoxelNum()),
                    thrust::device_pointer_cast(d_senmaps[0].get()),
                    [params] __device__(float a) { return a / params.subsetNum; });
  const auto vecSenmaps = std::vector<float *>(params.subsetNum, d_senmaps[0].get());

  // OSEM
  // read listmode
  openpni::io::ListmodeFileInput listmodeFile;
  listmodeFile.open(listmode_path);
  // prepare
  openpni::misc::ListmodeBuffer listmodeBuffer;
  openpni::cuda_sync_ptr<openpni::basic::Listmode_t> d_bufferForListmode;
  openpni::cuda_sync_ptr<char> d_buffer;
  std::size_t bufferSize = 0;
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(model.crystalGeometry());
  auto d_tmp_Img3D = openpni::make_cuda_sync_ptr<float>(OSEMimg.totalVoxelNum());
  auto d_out_Img3D = openpni::make_cuda_sync_ptr<float>(OSEMimg.totalVoxelNum());
  auto d_convolutionKernel = make_cuda_sync_ptr_from_hcopy(example::gaussianKernel<9>(1.5f));
  auto projectionMethod = openpni::math::ProjectionMethodUniform();
  projectionMethod.sampler.setSampleRatio(0.7f);
  std::vector<int16_t> testTOFDeviation(d_crystalGeometry.elements(), 5000);
  auto d_crystalTOFDeviation = openpni::make_cuda_sync_ptr_from_hcopy<int16_t>(testTOFDeviation);
  static int index__ = 0;

  using michDefaultIndexer = openpni::example::polygon::IndexerOfSubsetForMich;
  using randType = openpni::basic::_FactorAdaptorMich<openpni::basic::FactorType::Multiply, michDefaultIndexer>;
  using addType = openpni::basic::_FactorAdaptorMich<openpni::basic::FactorType::Addition, michDefaultIndexer>;
  using sssType = openpni::process::scatter::_SSSTOFAdaptor<michDefaultIndexer>;
  using EMSumUpdatorType = openpni::process::EMSumUpdatorTOF_CUDA<sssType, randType, addType>;
  std::vector<randType> d_multiFctAdptor(params.subsetNum);
  std::vector<addType> d_addFctAdptor(params.subsetNum);
  std::vector<sssType> d_sssFctAdptor(params.subsetNum);
  std::vector<EMSumUpdatorType> EMSumUpdatorSubsets(params.subsetNum);
  auto d_Corr_Add_mich = Corr_Add_mich
                             ? make_cuda_sync_ptr_from_hcopy<float>(std::span{Corr_Add_mich, model.michSize()})
                             : cuda_sync_ptr<float>{}; // 空的 cuda_sync_ptr
  auto d_Corr_Mul_mich = Corr_Mul_mich
                             ? make_cuda_sync_ptr_from_hcopy<float>(std::span{Corr_Mul_mich, model.michSize()})
                             : cuda_sync_ptr<float>{};
  auto d_sssValues = make_cuda_sync_ptr_from_hcopy<float>(std::span{sssValues, model.michSize() * params.tofBinNum});
  // prepare sss downSampling model
  auto downSampledPolygonSystem = model.polygonSystem();
  auto dsDetectorGeometry = model.detectorInfo().geometry;
  const auto dsCrystalGeometry = openpni::example::calCrystalGeo(downSampledPolygonSystem, dsDetectorGeometry);
  auto d_dsCrystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(dsCrystalGeometry);
  //
  for (const int subsetId : std::views::iota(0, params.subsetNum)) {
    d_multiFctAdptor[subsetId].factor = d_Corr_Mul_mich;
    d_multiFctAdptor[subsetId].indexer = michDefaultIndexer();
    d_multiFctAdptor[subsetId].indexer.scanner = model.polygonSystem();
    d_multiFctAdptor[subsetId].indexer.detector = model.detectorInfo().geometry;
    d_multiFctAdptor[subsetId].indexer.subsetId = subsetId;
    d_addFctAdptor[subsetId].factor = d_Corr_Add_mich; // or nullPtr
    d_addFctAdptor[subsetId].indexer = michDefaultIndexer();
    d_addFctAdptor[subsetId].indexer.subsetId = subsetId;
    d_sssFctAdptor[subsetId].sssValue = d_sssValues;
    d_sssFctAdptor[subsetId].indexer = michDefaultIndexer();
    d_addFctAdptor[subsetId].indexer.scanner = model.polygonSystem();
    d_addFctAdptor[subsetId].indexer.detector = model.detectorInfo().geometry;
    d_sssFctAdptor[subsetId].indexer.subsetId = subsetId;
    d_sssFctAdptor[subsetId].tofBinNum = params.tofBinNum;
    d_sssFctAdptor[subsetId].tofBinWidth = params.tofBinWidth;
    d_sssFctAdptor[subsetId].__polygon = model.polygonSystem();
    d_sssFctAdptor[subsetId].__detectorGeometry = model.detectorInfo().geometry;
    d_sssFctAdptor[subsetId].__dsPolygon = downSampledPolygonSystem;
    d_sssFctAdptor[subsetId].__dsDetectorGeometry = dsDetectorGeometry;
    d_sssFctAdptor[subsetId].in_crystalGeometry = d_crystalGeometry.get();
    d_sssFctAdptor[subsetId].in_dsCrystalGeometry = d_dsCrystalGeometry.get();
    EMSumUpdatorSubsets[subsetId].randFactorAdapter = d_addFctAdptor[subsetId];
    EMSumUpdatorSubsets[subsetId].multiplyFactorAdapter = d_multiFctAdptor[subsetId];
    EMSumUpdatorSubsets[subsetId].sssFactorAdapter = d_sssFctAdptor[subsetId];
  }

  auto GBSize = [](unsigned long long size) -> uint64_t { return size * 1024 * 1024 * 1024; };

  listmodeBuffer.setBufferSize(GBSize(size_GB) / sizeof(openpni::basic::Listmode_t))
      .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
        if (d_bufferForListmode.elements() < __count)
          d_bufferForListmode = openpni::make_cuda_sync_ptr<openpni::basic::Listmode_t>(__count);
        if (++index__ > 1)
          return;
        cudaMemcpy(d_bufferForListmode.get(), __data, __count * sizeof(openpni::basic::Listmode_t),
                   cudaMemcpyHostToDevice);

        const auto pairs = generatePairs(__count, params.subsetNum);
        std::vector<openpni::basic::DataViewListmodeTOF<>> dataSetsForRecon(params.subsetNum);
        for (const auto subsetIndex : std::views::iota(0, params.subsetNum)) {
          auto &dataView = dataSetsForRecon[subsetIndex];
          dataView.count = pairs[subsetIndex].second - pairs[subsetIndex].first;
          dataView.crystalGeometry = d_crystalGeometry.get();
          dataView.crystalTOFDeviation = d_crystalTOFDeviation.data();
          dataView.crystalTOFMean = nullptr; // 没做TOF校正
          dataView.listmodes = d_bufferForListmode.get() + pairs[subsetIndex].first;
        }

        while (!openpni::process::SEM_simple_CUDA(
            dataSetsForRecon, OSEMimg, d_tmp_Img3D.data(), d_convolutionKernel.get(), vecSenmaps, 1, d_buffer.get(),
            bufferSize, projectionMethod, EMSumUpdatorSubsets, openpni::process::ImageSimpleUpdate_CUDA())) {
          bufferSize += 20 * 1024 * 1024;
          std::cout << "Resize buffer to " << bufferSize << " bytes." << std::endl;
          d_buffer = openpni::make_cuda_sync_ptr<char>(bufferSize);
        }
        thrust::transform(
            thrust::device, thrust::device_pointer_cast(d_out_Img3D.begin()),
            thrust::device_pointer_cast(d_out_Img3D.end()), thrust::device_pointer_cast(d_tmp_Img3D.begin()),
            thrust::device_pointer_cast(d_out_Img3D.begin()), [] __device__(float a, float b) { return a + b; });
        static int iter = 0;
        std::cout << "OSEM done. iter = " << iter << std::endl;
      })
      .append(listmodeFile, openpni::io::selectSegments(listmodeFile))
      .flush();
  std::cout << "OSEM_listModeTOF_CUDA done." << std::endl;
  d_out_Img3D.allocator().copy_from_device_to_host(out_osem, d_out_Img3D.cspan());
}

inline void OSEM_listModeTOF_CUDA_MULTIGPU(
    float *out_osem, OSEM_params params, std::string listmode_path, polygon::PolygonModel &model,
    unsigned long long size_GB, float *Corr_Add_mich = nullptr, float *Corr_Mul_mich = nullptr,
    float *sssValues = nullptr) {
  auto OSEMimg = make_ImgGeometry(params.OSEMImgVoxelNum, params.OSEMImgVoxelSize);
  // cal senmap here only generate one senmap

  std::vector<std::vector<float>> h_senmaps;
  {
    std::vector<openpni::cuda_sync_ptr<float>> d_senmaps;
    createSenmap_CUDA(d_senmaps, model, OSEMimg, 1, params.binCut, Corr_Mul_mich);
    thrust::transform(thrust::device_pointer_cast(d_senmaps[0].get()),
                      thrust::device_pointer_cast(d_senmaps[0].get() + OSEMimg.totalVoxelNum()),
                      thrust::device_pointer_cast(d_senmaps[0].get()),
                      [params] __device__(float a) { return a / params.subsetNum; });
    d_senmaps[0].allocator().copy_from_device_to_host(h_senmaps.emplace_back(OSEMimg.totalVoxelNum()).data(),
                                                      d_senmaps[0].cspan());
  }

  // OSEM
  // read listmode
  openpni::io::ListmodeFileInput listmodeFile;
  listmodeFile.open(listmode_path);
  // prepare
  openpni::misc::ListmodeBuffer listmodeBuffer;

  using michDefaultIndexer = openpni::example::polygon::IndexerOfSubsetForMich;
  using randType = openpni::basic::_FactorAdaptorMich<openpni::basic::FactorType::Multiply, michDefaultIndexer>;
  using addType = openpni::basic::_FactorAdaptorMich<openpni::basic::FactorType::Addition, michDefaultIndexer>;
  using sssType = openpni::process::scatter::_SSSTOFAdaptor<michDefaultIndexer>;
  using EMSumUpdatorType = openpni::process::EMSumUpdatorTOF_CUDA<sssType, randType, addType>;

  auto GBSize = [](unsigned long long size) -> uint64_t { return size * 1024 * 1024 * 1024; };

  struct MultiGPUData {
    std::vector<openpni::basic::Listmode_t> listmodes;
  };
  int gpuNum = 0;
  cudaGetDeviceCount(&gpuNum);
  // gpuNum=1;
  std::cout << "GPU num = " << gpuNum << std::endl;
  common::MultiCycledBuffer<MultiGPUData> cycleBuffer(gpuNum);
  std::vector<std::thread> threadsGPU;
  std::mutex h_addMutex;
  for (int i = 0; i < gpuNum; ++i) {
    threadsGPU.emplace_back([&, i]() {
      cudaSetDevice(i);
      //================================
      openpni::cuda_sync_ptr<char> d_buffer;
      // senmaps
      auto d_senMaps = openpni::make_cuda_sync_ptr_from_hcopy<float>(h_senmaps[0]);
      const auto vecSenmaps = std::vector<float *>(params.subsetNum, d_senMaps.get());
      // prepare
      auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(model.crystalGeometry());
      auto d_tmp_Img3D = openpni::make_cuda_sync_ptr<float>(OSEMimg.totalVoxelNum());
      auto d_out_Img3D = openpni::make_cuda_sync_ptr<float>(OSEMimg.totalVoxelNum());
      auto d_convolutionKernel = make_cuda_sync_ptr_from_hcopy(example::gaussianKernel<9>(1.5f));
      auto projectionMethod = openpni::math::ProjectionMethodUniform();
      projectionMethod.sampler.setSampleRatio(0.4f);
      std::vector<int16_t> testTOFDeviation(d_crystalGeometry.elements(), 5000);
      auto d_crystalTOFDeviation = openpni::make_cuda_sync_ptr_from_hcopy<int16_t>(testTOFDeviation);
      std::vector<randType> d_multiFctAdptor(params.subsetNum);
      std::vector<addType> d_addFctAdptor(params.subsetNum);
      std::vector<sssType> d_sssFctAdptor(params.subsetNum);
      std::vector<EMSumUpdatorType> EMSumUpdatorSubsets(params.subsetNum);
      auto d_Corr_Add_mich = Corr_Add_mich
                                 ? make_cuda_sync_ptr_from_hcopy<float>(std::span{Corr_Add_mich, model.michSize()})
                                 : cuda_sync_ptr<float>{}; // 空的 cuda_sync_ptr
      auto d_Corr_Mul_mich = Corr_Mul_mich
                                 ? make_cuda_sync_ptr_from_hcopy<float>(std::span{Corr_Mul_mich, model.michSize()})
                                 : cuda_sync_ptr<float>{};
      auto d_sssValues =
          make_cuda_sync_ptr_from_hcopy<float>(std::span{sssValues, model.michSize() * params.tofBinNum});
      // prepare sss downSampling model
      auto downSampledPolygonSystem = model.polygonSystem();
      auto dsDetectorGeometry = model.detectorInfo().geometry;
      const auto dsCrystalGeometry = openpni::example::calCrystalGeo(downSampledPolygonSystem, dsDetectorGeometry);
      auto d_dsCrystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(dsCrystalGeometry);
      //
      for (const int subsetId : std::views::iota(0, params.subsetNum)) {
        d_multiFctAdptor[subsetId].factor = d_Corr_Mul_mich;
        d_multiFctAdptor[subsetId].indexer = michDefaultIndexer();
        d_multiFctAdptor[subsetId].indexer.subsetId = subsetId;
        d_addFctAdptor[subsetId].factor = d_Corr_Add_mich; // or nullPtr
        d_addFctAdptor[subsetId].indexer = michDefaultIndexer();
        d_addFctAdptor[subsetId].indexer.subsetId = subsetId;
        d_sssFctAdptor[subsetId].sssValue = d_sssValues;
        d_sssFctAdptor[subsetId].indexer = michDefaultIndexer();
        d_sssFctAdptor[subsetId].indexer.subsetId = subsetId;
        d_sssFctAdptor[subsetId].tofBinNum = params.tofBinNum;
        d_sssFctAdptor[subsetId].tofBinWidth = params.tofBinWidth;
        d_sssFctAdptor[subsetId].__polygon = model.polygonSystem();
        d_sssFctAdptor[subsetId].__detectorGeometry = model.detectorInfo().geometry;
        d_sssFctAdptor[subsetId].__dsPolygon = downSampledPolygonSystem;
        d_sssFctAdptor[subsetId].__dsDetectorGeometry = dsDetectorGeometry;
        d_sssFctAdptor[subsetId].in_crystalGeometry = d_crystalGeometry.get();
        d_sssFctAdptor[subsetId].in_dsCrystalGeometry = d_dsCrystalGeometry.get();
        EMSumUpdatorSubsets[subsetId].randFactorAdapter = d_addFctAdptor[subsetId];
        EMSumUpdatorSubsets[subsetId].multiplyFactorAdapter = d_multiFctAdptor[subsetId];
        EMSumUpdatorSubsets[subsetId].sssFactorAdapter = d_sssFctAdptor[subsetId];
      }
      //
      std::cout << "GPU " << i << " ready." << std::endl;
      while (cycleBuffer.read([&](const MultiGPUData &data) {
        if (data.listmodes.empty())
          return;
        std::size_t bufferSize = 0;
        auto d_bufferForListmode = openpni::make_cuda_sync_ptr_from_hcopy<openpni::basic::Listmode_t>(data.listmodes);
        const auto pairs = generatePairs(d_bufferForListmode.elements(), params.subsetNum);
        std::vector<openpni::basic::DataViewListmodeTOF<>> dataSetsForRecon(params.subsetNum);
        for (const auto subsetIndex : std::views::iota(0, params.subsetNum)) {
          auto &dataView = dataSetsForRecon[subsetIndex];
          dataView.count = pairs[subsetIndex].second - pairs[subsetIndex].first;
          dataView.crystalGeometry = d_crystalGeometry.get();
          dataView.crystalTOFDeviation = d_crystalTOFDeviation.data();
          dataView.crystalTOFMean = nullptr; // 没做TOF校正
          dataView.listmodes = d_bufferForListmode.get() + pairs[subsetIndex].first;
        }

        while (!openpni::process::SEM_simple_CUDA(
            dataSetsForRecon, OSEMimg, d_tmp_Img3D.data(), d_convolutionKernel.get(), vecSenmaps, 1, d_buffer.get(),
            bufferSize, projectionMethod, EMSumUpdatorSubsets, openpni::process::ImageSimpleUpdate_CUDA())) {
          bufferSize += 20 * 1024 * 1024;
          std::cout << "Resize buffer to " << bufferSize << " bytes." << std::endl;
          d_buffer = openpni::make_cuda_sync_ptr<char>(bufferSize);
        }

        auto h_img = openpni::make_vector_from_cuda_sync_ptr(d_tmp_Img3D, d_tmp_Img3D.cspan());
        std::lock_guard lock(h_addMutex);
        process::for_each(d_tmp_Img3D.elements(), [&](size_t idx) { out_osem[idx] += h_img[idx]; });
        std::cout << "GPU " << i << " one iter done." << std::endl;
      }))
        ;
    });
  }

  listmodeBuffer.setBufferSize(GBSize(size_GB) / sizeof(openpni::basic::Listmode_t))
      .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
        cycleBuffer.write([&](MultiGPUData &data) {
          data.listmodes.resize(__count);
          std::memcpy(data.listmodes.data(), __data, __count * sizeof(openpni::basic::Listmode_t));
          std::cout << "copy listmode to multiGPUData done, count = " << __count << std::endl;
        });
      })
      .append(listmodeFile, openpni::io::selectSegments(listmodeFile))
      .flush();
  std::cout << "OSEM_listModeTOF_CUDA done." << std::endl;
  cycleBuffer.stop();

  for (auto &t : threadsGPU)
    t.join();
}

} // namespace openpni::example
