#pragma once
#include <algorithm>
#include <numeric>

#include "../basic/CudaPtr.hpp"
#include "../basic/Model.hpp"
#include "../basic/PetDataType.h"
#include "../detector/Detectors.hpp"
#include "../process/Acquisition.hpp"
namespace openpni::process {

using LocalSinglesOfEachChannel = std::vector<cuda_sync_ptr<basic::LocalSingle_t>>;
LocalSinglesOfEachChannel rtos(process::RawDataView d_rawdata, device::DetectorBase *const *h_detectors);

struct CoincidenceResult {
  cuda_sync_ptr<basic::Listmode_t> d_promptListmode;
  cuda_sync_ptr<basic::Listmode_t> d_delayListmode;
  cuda_sync_ptr<unsigned> d_countMap;
  uint64_t promptCount{0};
  uint64_t delayCount{0};
};

struct CoincidenceProtocol {
  int timeWindow_ps{2'000};
  int delayTime_ps{2'000'000};
  float energyLower_eV{350'000};
  float energyUpper_eV{650'000};
};

// Single 2 Coin with count map addition
CoincidenceResult stoc(const LocalSinglesOfEachChannel &d_localSingles, device::DetectorBase *const *h_detectors,
                       CoincidenceProtocol protocol);
} // namespace openpni::process
