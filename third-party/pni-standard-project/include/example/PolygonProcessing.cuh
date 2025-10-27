#pragma once
#include "../IO.hpp"
#include "../basic/CudaPtr.hpp"
#include "../basic/PetDataType.h"
#include "../process/ListmodeProcessing.hpp"
#include "PolygonalSystem.hpp"
namespace openpni::example::polygon {
process::LocalSinglesOfEachChannel RtoS_DData(PolygonModel &model, process::RawDataView d_rawDataView);
process::LocalSinglesOfEachChannel RtoS_HData(PolygonModel &model, process::RawDataView h_rawDataView);
process::CoincidenceResult StoC_DData(PolygonModel &model, const process::LocalSinglesOfEachChannel &d_globalSingles,
                                      process::CoincidenceProtocol protocol);
std::vector<unsigned> rearrange_countmap(const std::vector<unsigned> &countmap, PolygonModel &model);
void listmode_to_mich(std::vector<float> &mich, PolygonModel &model, std::span<const basic::Listmode_t> listmodes);
void fill_senmap_cuda(PolygonModel &model, io::F32Image &senmap, int subsetId, int subsetNum, float hfwhm);
} // namespace openpni::example::polygon
