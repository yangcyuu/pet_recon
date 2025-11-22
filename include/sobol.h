#pragma once

#include <torch/torch.h>

struct SobolEngine {
  static constexpr int MAXBIT = 30;
  static constexpr int MAXDIM = 21201;

  SobolEngine(int64_t dimension, bool scramble = false, std::optional<uint64_t> seed = std::nullopt) :
      _dimension(dimension), _scramble(scramble), _seed(seed) {
    torch::Device device = torch::kCPU;
    _sobolstate = torch::zeros({_dimension, MAXBIT}, torch::dtype(torch::kLong).device(device));
    torch::_sobol_engine_initialize_state_(_sobolstate, _dimension);

    if (!_scramble) {
      _shift = torch::zeros({_dimension}, torch::dtype(torch::kLong).device(device));
    } else {
      this->scramble();
    }

    _aquisition = _shift.clone(torch::MemoryFormat::Contiguous);
    _first = (_aquisition / (1LL << MAXBIT)).reshape({1, -1});
  }

  torch::Tensor draw(int64_t n = 1, torch::ScalarType dtype = torch::kFloat32) {
    TORCH_CHECK(n >= 0, "Number of samples must be non-negative");
    // [n, dimension]
    torch::Tensor result;

    if (_num_generated == 0) {
      if (n == 1) {
        result = _first.to(dtype);
      } else {
        std::tie(result, _aquisition) =
            torch::_sobol_engine_draw(_aquisition, n - 1, _sobolstate, _dimension, _num_generated, dtype);
        result = torch::cat({_first.to(dtype), result}, -2);
      }
    } else {
      std::tie(result, _aquisition) =
          torch::_sobol_engine_draw(_aquisition, n, _sobolstate, _dimension, _num_generated - 1, dtype);
    }
    _num_generated = (static_cast<uint64_t>(_num_generated) + n) % std::numeric_limits<int64_t>::max();
    return result;
  }

private:
  int64_t _dimension;
  int64_t _num_generated = 0;
  bool _scramble;
  std::optional<uint64_t> _seed;
  torch::Tensor _sobolstate;
  torch::Tensor _aquisition;
  torch::Tensor _shift;
  torch::Tensor _first;

  void scramble() {
    std::optional<torch::Generator> gen;
    if (_seed.has_value()) {
      gen = torch::make_generator<torch::CPUGeneratorImpl>(_seed.value());
    }

    torch::Device device = torch::kCPU;

    torch::Tensor shift = torch::randint(2, {_dimension, MAXBIT}, gen, torch::dtype(torch::kLong).device(device));

    _shift = torch::mv(shift, torch::pow(2, torch::arange(0, MAXBIT, torch::dtype(torch::kLong).device(device))));

    torch::Tensor ltm =
        torch::randint(2, {_dimension, MAXBIT, MAXBIT}, gen, torch::dtype(torch::kLong).device(device)).tril();

    torch::_sobol_engine_scramble_(_sobolstate, ltm, _dimension);
  }
};
