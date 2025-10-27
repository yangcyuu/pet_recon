#pragma once
#include "../core/BasicMath.hpp"
namespace openpni::experimental::algorithms {
template <typename T, typename Compare>
inline __PNI_CUDA_MACRO__ void bubble_sort(
    T *data, int size, Compare comp) {
  for (int i = 0; i < size - 1; i++) {
    for (int j = 0; j < size - 1 - i; j++) {
      if (comp(data[j], data[j + 1])) {
        T temp = data[j];
        data[j] = data[j + 1];
        data[j + 1] = temp;
      }
    }
  }
}
template <typename T, typename Compare>
__PNI_CUDA_MACRO__ inline void bubble_sort_partial(
    T *data, int size, int part, Compare comp) {
  part = core::FMath<int>::min(part, size);
  for (int i = 0; i < part; i++) {
    for (int j = 0; j < size - 1 - i; j++) {
      if (comp(data[j], data[j + 1])) {
        T temp = data[j];
        data[j] = data[j + 1];
        data[j + 1] = temp;
      }
    }
  }
}
template <typename T>
__PNI_CUDA_MACRO__ inline T *binary_search(
    // into a sorted range, supposed: [N+1] > [N] > [N-1] > ... > [0]
    // Return the last position that value <= *pos
    T *__begin, T *__end, T __value) {
  if (*__end > __value)
    return __end;
  if (*__begin <= __value)
    return __begin;
  T *pos = __begin;
  T *left = __begin;
  T *right = __end;
  while (left + 1 < right) {
    pos = left + (right - left) / 2;
    if (*pos > __value)
      right = pos;
    else
      left = pos + 1;
  }
  return right;
}

template <typename T, typename Compare>
__PNI_CUDA_MACRO__ inline void insert(
    // into a sorted range, supposed: [N+1] > [N] > [N-1] > ... > [0]
    T *__begin, T *__end, T __value) {
  T *pos = __begin;
  while (pos != __end && comp(*pos, __value))
    pos++;
  if (pos != __end) {
    T temp = *pos;
    *pos = __value;
    pos++;
    while (pos != __end) {
      T temp2 = *pos;
      *pos = temp;
      temp = temp2;
      pos++;
    }
  }
}

template <typename T, typename Compare>
__PNI_CUDA_MACRO__ inline void min_heap_insert_replace(
    // Insert a new value into min-heap and remove the maximum element
    // Maintains the same array size
    T *__begin, T *__end, T __value, Compare comp) {
  int size = __end - __begin;
  if (size == 0)
    return;

  // If new value is larger than current max (at end of sorted heap), skip
  if (comp(__begin[size - 1], __value))
    return;

  // Replace the last element (max) with new value
  __begin[size - 1] = __value;

  // Heapify up from the last position
  int pos = size - 1;
  while (pos > 0) {
    int parent = (pos - 1) / 2;
    if (comp(__begin[pos], __begin[parent])) {
      // Swap with parent
      T temp = __begin[pos];
      __begin[pos] = __begin[parent];
      __begin[parent] = temp;
      pos = parent;
    } else {
      break;
    }
  }
}

template <typename T>
__PNI_CUDA_MACRO__ inline void min_heap_insert_replace(
    T *__begin, T *__end, T __value) {
  min_heap_insert_replace(__begin, __end, __value, [] __PNI_CUDA_MACRO__(const T &a, const T &b) { return a < b; });
}

template <typename T, typename Compare>
__PNI_CUDA_MACRO__ inline void make_min_heap(
    T *__begin, T *__end, Compare comp) {
  int size = __end - __begin;
  // Build heap from bottom up
  for (int i = size / 2 - 1; i >= 0; i--) {
    int pos = i;
    while (true) {
      int left = 2 * pos + 1;
      int right = 2 * pos + 2;
      int smallest = pos;

      if (left < size && comp(__begin[left], __begin[smallest]))
        smallest = left;
      if (right < size && comp(__begin[right], __begin[smallest]))
        smallest = right;

      if (smallest != pos) {
        T temp = __begin[pos];
        __begin[pos] = __begin[smallest];
        __begin[smallest] = temp;
        pos = smallest;
      } else {
        break;
      }
    }
  }
}
} // namespace openpni::experimental::algorithms