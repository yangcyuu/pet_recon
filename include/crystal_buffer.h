#pragma once

#include "define.h"
#include "utils.h"

template<typename T>
class CrystalBuffer {
  constexpr static size_t VECTOR_SIZE = sizeof(Vector3<T>);

public:
  CrystalBuffer() = default;

  void crystals(std::span<const RectangleGeom<T>> geoms) {
    allocate_buffer(geoms.size() / 2);
    for (size_t i = 0; i < geoms.size() / 2; ++i) {
      p0()[i] = geoms[2 * i].O;
      u0()[i] = geoms[2 * i].U;
      v0()[i] = geoms[2 * i].V;
      p1()[i] = geoms[2 * i + 1].O;
      u1()[i] = geoms[2 * i + 1].U;
      v1()[i] = geoms[2 * i + 1].V;
    }
  }

  std::span<Vector3<T>> p0() { return std::span<Vector3<T>>(reinterpret_cast<Vector3<T> *>(_buffer.get()), _size); }

  std::span<const Vector3<T>> p0() const {
    return std::span<const Vector3<T>>(reinterpret_cast<const Vector3<T> *>(_buffer.get()), _size);
  }

  std::span<Vector3<T>> u0() {
    return std::span<Vector3<T>>(reinterpret_cast<Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size), _size);
  }

  std::span<const Vector3<T>> u0() const {
    return std::span<const Vector3<T>>(reinterpret_cast<const Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size),
                                       _size);
  }

  std::span<Vector3<T>> v0() {
    return std::span<Vector3<T>>(reinterpret_cast<Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size * 2), _size);
  }

  std::span<const Vector3<T>> v0() const {
    return std::span<const Vector3<T>>(reinterpret_cast<const Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size * 2),
                                       _size);
  }

  std::span<Vector3<T>> p1() {
    return std::span<Vector3<T>>(reinterpret_cast<Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size * 3), _size);
  }

  std::span<const Vector3<T>> p1() const {
    return std::span<const Vector3<T>>(reinterpret_cast<const Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size * 3),
                                       _size);
  }

  std::span<Vector3<T>> u1() {
    return std::span<Vector3<T>>(reinterpret_cast<Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size * 4), _size);
  }

  std::span<const Vector3<T>> u1() const {
    return std::span<const Vector3<T>>(reinterpret_cast<const Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size * 4),
                                       _size);
  }

  std::span<Vector3<T>> v1() {
    return std::span<Vector3<T>>(reinterpret_cast<Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size * 5), _size);
  }

  std::span<const Vector3<T>> v1() const {
    return std::span<const Vector3<T>>(reinterpret_cast<const Vector3<T> *>(_buffer.get() + VECTOR_SIZE * _size * 5),
                                       _size);
  }

private:
  std::unique_ptr<std::byte[]> _buffer;
  size_t _size = 0;

  void allocate_buffer(size_t new_size) {
    if (new_size > _size) {
      _buffer = std::make_unique<std::byte[]>(2 * new_size * VECTOR_SIZE * 3);
      _size = new_size;
    }
  }
};
