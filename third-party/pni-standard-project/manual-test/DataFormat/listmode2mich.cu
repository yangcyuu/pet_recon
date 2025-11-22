#include <format>
#include <pni/IO.hpp>
#include <pni/Polygon.hpp>
#include <pni/misc/ListmodeBuffer.hpp>
const char *listmode_filename = "test_listmode.pni";
const char *model_json_filename = "../raw-data-to-coin/bdm2.json";
constexpr uint64_t operator""_GB(
    unsigned long long size) {
  return size * 1024 * 1024 * 1024;
}
std::string read_char_file(
    std::string_view filename) {
  std::ifstream ifs{std::string(filename)};
  if (!ifs.is_open()) {
    throw std::runtime_error("Could not open file: " + std::string(filename));
  }
  return std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
}

std::shared_ptr<openpni::example::polygon::PolygonModel> build_model() {
  auto model = openpni::example::polygon::quickBuildFromJson(read_char_file(model_json_filename));
  return model
      .or_else([](auto &&err) {
        throw std::runtime_error(
            std::format("Error building model from JSON: {}", openpni::example::polygon::getErrorMessage(err)));
      })
      .value();
}
int main() {
  auto model = build_model();
  std::cout << std::format("Model has {} crystals\n", model->crystalNum());

  openpni::io::ListmodeFileInput listmodeFileInput;
  listmodeFileInput.open(listmode_filename);

  std::vector<float> mich(model->michSize(), 0);
  std::vector<unsigned> crystalCounts(model->crystalNum(), 0);
  openpni::cuda_sync_ptr<openpni::basic::Listmode_t> d_bufferForListmode;
  openpni::misc::ListmodeBufferAsync()
      .setBufferSize(16_GB / sizeof(openpni::basic::Listmode_t))
      .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
        openpni::example::polygon::listmode_to_mich(mich, *model, std::span(__data, __count));
        for (std::size_t i = 0; i < __count; ++i) {
          ++crystalCounts[openpni::example::polygon::getRectangleIDFromUniformID(
              model->polygonSystem(), model->detectorInfo().geometry, __data[i].globalCrystalIndex1)];
          ++crystalCounts[openpni::example::polygon::getRectangleIDFromUniformID(
              model->polygonSystem(), model->detectorInfo().geometry, __data[i].globalCrystalIndex2)];
        }
      })
      .append(listmodeFileInput, openpni::io::selectSegments(listmodeFileInput))
      .flush();
  std::cout << "Done processing listmode file." << std::endl;
  std::ofstream michFile("mich.bin", std::ios::binary);
  michFile.write(reinterpret_cast<const char *>(mich.data()), mich.size() * sizeof(float));
  std::ofstream crystalCountsFile("crystal_counts.bin", std::ios::binary);
  crystalCountsFile.write(reinterpret_cast<const char *>(crystalCounts.data()),
                          crystalCounts.size() * sizeof(unsigned));
  return 0;
}
