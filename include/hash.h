#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
inline uint64_t murmur_hash64a(const std::byte *key, size_t len,
                               uint64_t seed) noexcept {
  constexpr uint64_t m = 0xc6a4a7935bd1e995ull;
  constexpr int r = 47;

  uint64_t h = seed ^ (len * m);

  const std::byte *end = key + (len & ~0x7);

  while (key != end) {
    uint64_t k = *reinterpret_cast<const uint64_t *>(key);
    key += 8;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  switch (len & 7) {
  case 7: {
    h ^= static_cast<uint64_t>(key[6]) << 48;
  }
  case 6: {
    h ^= static_cast<uint64_t>(key[5]) << 40;
  }
  case 5: {
    h ^= static_cast<uint64_t>(key[4]) << 32;
  }
  case 4: {
    h ^= static_cast<uint64_t>(key[3]) << 24;
  }
  case 3: {
    h ^= static_cast<uint64_t>(key[2]) << 16;
  }
  case 2: {
    h ^= static_cast<uint64_t>(key[1]) << 8;
  }
  case 1: {
    h ^= static_cast<uint64_t>(key[0]);
    h *= m;
    break;
  }
  default: {
    break;
  }
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

inline uint64_t hash_memory(const void *data, size_t size,
                            uint64_t seed = 0) noexcept {
  return murmur_hash64a(static_cast<const std::byte *>(data), size, seed);
}

template <typename... Args> uint64_t hash(const Args &...args) noexcept {
  constexpr size_t size = (sizeof(args) + ...);
  constexpr size_t n = (size + 7) / 8;
  uint64_t buffer[n];
  size_t offset = 0;
  ((memcpy(buffer + offset, &args, sizeof(args)), offset += sizeof(args)),
   ...);
  return murmur_hash64a(reinterpret_cast<const std::byte *>(buffer), size, 0);
}
