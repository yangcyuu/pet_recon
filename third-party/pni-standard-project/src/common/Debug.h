#pragma once
#include <fstream>
#include <termcolor/termcolor.hpp>

#include "include/basic/CudaPtr.hpp"
namespace openpni::debug {
void d_print_max_value(float const *d_data, std::size_t size);
void d_print_none_zero_average_value(float const *d_data, std::size_t size);
void d_print_value_at_index(uint32_t const *d_data, std::size_t begin, std::size_t end);
} // namespace openpni::debug

namespace openpni::debug {
template <typename T>
void h_write_array_to_disk(
    std::vector<T> const &h_data, const std::string &filename) {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    std::cerr << termcolor::red << termcolor::bold << "Error opening file for writing: " << filename << termcolor::reset
              << std::endl;
    return;
  }
  outfile.write(reinterpret_cast<const char *>(h_data.data()), h_data.size() * sizeof(T));
}
template <typename T_Ptr>
void h_write_array_to_disk(
    T_Ptr const &h_data, std::size_t size, const std::string &filename) {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    std::cerr << termcolor::red << termcolor::bold << "Error opening file for writing: " << filename << termcolor::reset
              << std::endl;
    return;
  }
  outfile.write(reinterpret_cast<const char *>(&h_data[0]), size * sizeof(decltype(T_Ptr()[0])));
}
template <typename T>
void d_write_array_to_disk(
    cuda_sync_ptr<T> const &d_data, const std::string &filename) {
  auto h_data = openpni::make_vector_from_cuda_sync_ptr(d_data, d_data.cspan());
  h_write_array_to_disk(h_data, d_data.cspan().size(), filename);
}
inline void say_error_message(
    const std::string &message) {
  std::cerr << termcolor::red << message << termcolor::reset;
}
inline void say_critical_message(
    const std::string &message) {
  std::cerr << termcolor::red << termcolor::bold << message << termcolor::reset;
}

} // namespace openpni::debug