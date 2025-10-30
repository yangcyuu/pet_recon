#include "include/experimental/node/KGConv3D.hpp"
#include "include/experimental/node/KNNConv3D.hpp"
#include "include/experimental/tools/Parallel.hpp"
using namespace openpni::experimental;
#include "../public.hpp"
#include "include/basic/CudaPtr.hpp"
const auto fileName = "overall_wellCounter_recon_img_gpu.bin";
int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::STATIC);

  auto osemGrids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                          core::Vector<int64_t, 3>::create(320, 320, 400));

  std::unique_ptr<float[]> h_image = read_from_file<float>(fileName, osemGrids.totalSize());

  node::KNNConv3D knnConv;
  knnConv.setFeatureSizeHalf(core::Vector<int64_t, 3>::create(1, 1, 1));
  knnConv.setKNNSearchSizeHalf(core::Vector<int64_t, 3>::create(4, 4, 4));
  knnConv.setKNNNumbers(50);
  knnConv.setKNNSigmaG2(32.f);
  core::TensorDataIO<float, 3> imageDataIO;
  auto d_image = openpni::make_cuda_sync_ptr_from_hcopy(std::span<float const>(h_image.get(), osemGrids.totalSize()));
  imageDataIO.grid = osemGrids;
  imageDataIO.ptr_in = d_image.get();
  imageDataIO.ptr_out = d_image.get();
  // imageDataIO.ptr_in = h_image.get();
  // imageDataIO.ptr_out = h_image.get();
  knnConv.convD(imageDataIO);
  knnConv.deconvD(imageDataIO);
  // knnConv.convH(imageDataIO);
  // knnConv.deconvH(imageDataIO);

  d_image.allocator().copy_from_device_to_host(h_image.get(), d_image.cspan());

  write_to_file("overall_wellCounter_recon_img_gpu_knn_conv.bin", h_image.get(), osemGrids.totalSize());
}