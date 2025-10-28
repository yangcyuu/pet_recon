#include "texture.h"
#include <filesystem>
#include <fstream>
#include "cgmath.h"

void Texture2D::save(const std::string &filename) const {
  torch::Tensor tensor = this->tensor().detach().clone().permute({1, 2, 0}).contiguous(); // to [H, W, C]
  if (tensor.dtype() != torch::kFloat32) {
    tensor = tensor.to(torch::kFloat32);
  }
  if (!tensor.device().is_cpu()) {
    tensor = tensor.to(torch::kCPU);
  }
  cv::Mat img(static_cast<int>(tensor.size(0)), static_cast<int>(tensor.size(1)),
              CV_32FC(static_cast<int>(tensor.size(2))), tensor.data_ptr());
  std::filesystem::path filepath = std::filesystem::absolute(filename);
  if (!std::filesystem::exists(filepath.parent_path())) {
    std::filesystem::create_directories(filepath.parent_path());
  }
  if (filepath.extension() != ".exr" && filepath.extension() != ".hdr") {
    for (int y = 0; y < img.rows; ++y) {
      for (int x = 0; x < img.cols; ++x) {
        for (int c = 0; c < img.channels(); ++c) {
          switch (img.channels()) {
            case 1: {
              img.at<float>(y, x) = linear_to_gamma(img.at<float>(y, x));
              break;
            }
            case 2: {
              img.at<cv::Vec2f>(y, x)[c] = linear_to_gamma(img.at<cv::Vec2f>(y, x)[c]);
              break;
            }
            case 3: {
              img.at<cv::Vec3f>(y, x)[c] = linear_to_gamma(img.at<cv::Vec3f>(y, x)[c]);
              break;
            }
            case 4: {
              img.at<cv::Vec4f>(y, x)[c] = linear_to_gamma(img.at<cv::Vec4f>(y, x)[c]);
              break;
            }
            default: {
              ERROR_AND_EXIT("Texture2D::save: unsupported number of channels {}", img.channels());
            }
          }
        }
      }
    }
    cv::Mat img_u8;
    img.convertTo(img_u8, CV_8U);
    if (!cv::imwrite(filename, img_u8)) {
      ERROR_AND_EXIT("Texture2D::save: failed to save image to {}", filename);
    }
    return;
  }
  if (!cv::imwrite(filename, img)) {
    ERROR_AND_EXIT("Texture2D::save: failed to save image to {}", filename);
  }
}

void Texture3D::save(const std::string &filename) const {
  torch::Tensor tensor = this->tensor().detach().clone().permute({1, 2, 3, 0}).contiguous(); // to [D, H, W, C]
  if (tensor.dtype() != torch::kFloat32) {
    tensor = tensor.to(torch::kFloat32);
  }
  if (!tensor.device().is_cpu()) {
    tensor = tensor.to(torch::kCPU);
  }
  std::filesystem::path dir = std::filesystem::absolute(filename);
  if (!std::filesystem::exists(dir.parent_path())) {
    std::filesystem::create_directories(dir.parent_path());
  }
  for (int d = 0; d < tensor.size(0); ++d) {
    cv::Mat img(static_cast<int>(tensor.size(1)), static_cast<int>(tensor.size(2)),
                CV_32FC(static_cast<int>(tensor.size(3))),
                tensor.data_ptr<float>() + d * tensor.size(1) * tensor.size(2) * tensor.size(3));
    auto slice_dir = dir;
    slice_dir.replace_filename(std::format("{}_{:06}{}", dir.stem().string(), d, dir.extension().string()));
    if (dir.extension() != ".exr" && dir.extension() != ".hdr") {
      for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
          for (int c = 0; c < img.channels(); ++c) {
            switch (img.channels()) {
              case 1: {
                img.at<float>(y, x) = linear_to_gamma(img.at<float>(y, x));
                break;
              }
              case 2: {
                img.at<cv::Vec2f>(y, x)[c] = linear_to_gamma(img.at<cv::Vec2f>(y, x)[c]);
                break;
              }
              case 3: {
                img.at<cv::Vec3f>(y, x)[c] = linear_to_gamma(img.at<cv::Vec3f>(y, x)[c]);
                break;
              }
              case 4: {
                img.at<cv::Vec4f>(y, x)[c] = linear_to_gamma(img.at<cv::Vec4f>(y, x)[c]);
                break;
              }
              default: {
                ERROR_AND_EXIT("Texture3D::save: unsupported number of channels {}", img.channels());
              }
            }
          }
        }
      }
      cv::Mat img_u8;
      img.convertTo(img_u8, CV_8U);
      if (!cv::imwrite(slice_dir.string(), img_u8)) {
        ERROR_AND_EXIT("Texture3D::save: failed to save image to {}", slice_dir.string());
      }
      continue;
    }
    if (!cv::imwrite(slice_dir.string(), img)) {
      ERROR_AND_EXIT("Texture3D::save: failed to save image to {}", slice_dir.string());
    }
  }
}

void Texture3D::save_rawdata(const std::string_view filename) const {
  torch::Tensor tensor = this->tensor().detach().clone().permute({1, 2, 3, 0}).contiguous(); // to [D, H, W, C]
  if (tensor.dtype() != torch::kFloat32) {
    tensor = tensor.to(torch::kFloat32);
  }
  if (!tensor.device().is_cpu()) {
    tensor = tensor.to(torch::kCPU);
  }
  if (!std::filesystem::exists(std::filesystem::path(filename).parent_path())) {
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
  }
  std::ofstream ofs(filename.data(), std::ios::binary);
  if (!ofs) {
    ERROR_AND_EXIT("Texture3D::save_rawdata: failed to open file {}", filename);
  }
  ofs.write(static_cast<const char *>(tensor.data_ptr()), tensor.numel() * sizeof(float));
  ofs.flush();
  ofs.close();
}
