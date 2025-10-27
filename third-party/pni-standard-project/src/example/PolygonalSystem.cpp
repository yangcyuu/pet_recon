#include "include/example/PolygonalSystem.hpp"

#include <iostream>
#include <ranges>
#include <regex>

#include "../autogen/autogen_xml.hpp"
#include "../integration/IntegratedModelBuilder.hpp"
#include "include/detector/BDM2.hpp"
#include "src/detector/BDM2/BDM2_impl.hpp"

bool sparseEquals(
    const std::string &a, const std::string &b) {
  std::regex whitespaceRegex("^\\s+|\\s+$"); // 去除字符串两端的空白字符
  auto trim = [&whitespaceRegex](const std::string &str) { return std::regex_replace(str, whitespaceRegex, ""); };

  auto toLower = [](const std::string &str) // 将字符串转换为小写
  {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
  };

  return toLower(trim(a)) == toLower(trim(b));
}
bool sparseContains(
    const std::set<std::string> &set, const std::string &value) {
  return std::ranges::any_of(set, [&value](const std::string &item) { return sparseEquals(item, value); });
}

template<typename T>
  requires std::is_base_of_v<openpni::device::DetectorBase, T>
  static auto _buildPolygonModel(
     const openpni::autogen::json::PolygonalSystemDefine&define) noexcept(false) {
  auto toPolygon = [](const openpni::autogen::json::PolygonalSystemDefine &define) {
    openpni::example::PolygonalSystem polygon;
    polygon.edges = define.Edges;
    polygon.detectorPerEdge = define.DetectorPerEdge;
    polygon.detectorLen = define.DetectorLen;
    polygon.radius = define.Radius;
    polygon.angleOf1stPerp = define.AngleOf1stPerp_degree;
    polygon.detectorRings = define.DetectorRings;
    polygon.ringDistance = define.RingDistance;
    return polygon;
  };

  auto modelBuilder = openpni::example::polygon::PolygonModelBuilder<T>(toPolygon(define));
  for (const auto index : std::views::iota(0, modelBuilder.totalDetectorNum())) {
    modelBuilder.setDetectorIP(
        index, define.DetectorInfo[index].IP.src_addr, static_cast<uint16_t>(define.DetectorInfo[index].IP.src_port),
        define.DetectorInfo[index].IP.dst_addr, static_cast<uint16_t>(define.DetectorInfo[index].IP.dst_port));
    if (!define.DetectorInfo[index].CalibrationFile.empty())
      modelBuilder.loadCalibration(index, define.DetectorInfo[index].CalibrationFile);
  }
  return modelBuilder.build();
}

namespace openpni::example::polygon {
expected<std::shared_ptr<PolygonModel>, QuickBuildFromJsonError> quickBuildFromJson(
    const std::string &unparsedJson) noexcept {
#define RETURN_ERROR(err)                                                                                              \
  {                                                                                                                    \
    return tl::unexpected(QuickBuildFromJsonError::err);                                                               \
  }
  autogen::json::json j;
  try {
    j = autogen::json::json::parse(unparsedJson);
  } catch (const std::exception &e) {
    RETURN_ERROR(JsonParseError)
  }

  const auto jsonFile = autogen::json::struct_cast<autogen::json::PolygonalSystemDefine>(j);

  const auto ringNum = jsonFile.DetectorRings;
  const auto detectorInRingNum = jsonFile.DetectorPerEdge * jsonFile.Edges;
  if (ringNum * detectorInRingNum == 0 || jsonFile.DetectorInfo.empty())
    RETURN_ERROR(EmptyJson)
  if (ringNum * detectorInRingNum != jsonFile.DetectorInfo.size())
    RETURN_ERROR(InfoNumMismatch)
  if (!sparseContains(device::setSupportedScanners, jsonFile.DetectorType))
    RETURN_ERROR(UnKnownDetector)

  try {
    using namespace openpni::device;
    if (sparseEquals(jsonFile.DetectorType, names::BDM2)) {
      return _buildPolygonModel<bdm2::BDM2Runtime>(jsonFile);
    } else {
      RETURN_ERROR(UnKnownDetector)
    }
  } catch (const std::exception &e) {
    RETURN_ERROR(CaliFileInvalid)
  }

#undef RETURN_ERROR
}

const char *getErrorMessage(
    QuickBuildFromJsonError error) noexcept {
  switch (error) {
  case QuickBuildFromJsonError::NoError:
    return "No error";
  case QuickBuildFromJsonError::JsonParseError:
    return "JSON parse error";
  case QuickBuildFromJsonError::EmptyJson:
    return "Empty JSON";
  case QuickBuildFromJsonError::InfoNumMismatch:
    return "Calibration file number mismatch the expected number (note: supposed "
           "detector number = edges * detectorPerEdge * detectorRings)";
  case QuickBuildFromJsonError::UnKnownDetector:
    return "Unknown detector type, supposed to be one of: "
           "BDM2, "
           "BDM1286";
  case QuickBuildFromJsonError::CaliFileInvalid:
    return "Invalid calibration file (file not exist, or file format error)";
  default:
    return "Unknown error";
  }
}
} // namespace openpni::example::polygon
