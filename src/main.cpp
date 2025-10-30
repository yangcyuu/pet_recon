#include "renderer.h"


int main() {
  auto renderer = Renderer({
      .mich_file = "coin_all.image3d",
      .offset = 6,
      .samples_per_crystal = 1,
      .samples_per_lor = 1,
      .iter_per_slice = 10,
      // .batch_size = 256,
      .use_sobol = true,
      .tof_sigma = 0.0f,
      .voxel_size = {0.5f, 0.5f, 0.5f},
      .image_size = {320, 320, 400},
  });

  renderer.render("mich_recon.image3d");
}
