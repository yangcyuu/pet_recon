#pragma once
#include "include/experimental/node/Coincidence.hpp"
namespace openpni::experimental::node::impl {
void d_copy(interface::LocalSingle *d_dest, interface::LocalSingle const *d_src, size_t numElements);
void d_countCrystal(unsigned *d_crystalCountMap, interface::LocalSingle const *d_singles, uint64_t singleCount,
                    unsigned const *d_crystalNumInclusiveSum, unsigned channelCount);
void d_countCrystal(unsigned *d_crystalCountMap, LocalListmode const *d_listmode, uint64_t listmodeCount,
                    unsigned const *d_crystalNumInclusiveSum, unsigned channelCount);
std::size_t d_energyFilter(interface::LocalSingle *d_singles, uint64_t singleCount, float energyLowerBound,
                           float energyUpperBound);
void d_doubleSinglesAndShiftTime(interface::LocalSingle *d_singles, uint64_t singleCount, uint64_t timeShift_pico);
void d_sortSinglesByTime(interface::LocalSingle *d_singles, uint64_t singleCount);
void d_countCoincidences(interface::LocalSingle const *d_singles, uint64_t singleCount, uint8_t *d_promptCount,
                         uint8_t *d_delayCount, int timeWindow_pico);
void d_inclusiveSumUint8(uint8_t const *d_input, uint64_t elementCount, uint32_t *d_output);
void d_extractCoincidenceListmode(interface::LocalSingle const *d_singles, uint64_t singleCount,
                                  uint32_t const *d_promptInclusiveSum, uint32_t const *d_delayInclusiveSum,
                                  LocalListmode *d_promptListmode, LocalListmode *d_delayListmode, uint64_t maxCount);
} // namespace openpni::experimental::node::impl
