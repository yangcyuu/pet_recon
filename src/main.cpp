#include "renderer.h"


int main() {
  // torch::autograd::DetectAnomalyGuard detect_anomaly_guard;
  auto renderer = Renderer({
      .mich_file = "graphic_test/coin_all.image3d",
      .mich_delay_file = "graphic_test/delay_all.image3d",
      .mich_attn_file = "graphic_test/attnCoff.bin",
      .mich_norm_file = "graphic_test/normCoff.bin",
      .offset = 6,
      .delay_offset = 6,
      .attn_offset = 0,
      .norm_offset = 0,
      .crystal_sigma = 1.0f / 3.0f,
      .samples_per_crystal = 10,
      .samples_per_lor = 4,
      .iter_per_slice = 20,
      .batch_size = 104 * 78,
      .use_sobol = true,
      .linear_sampling = true,
      .linear_step = 1.0f,
      .tof_sigma = 0.0f,
      .voxel_size = {0.5f, 0.5f, 0.5f},
      .image_size = {320, 320, 400},
      .define = E180(),
  });

  renderer.render("mich_recon.image3d");
}
