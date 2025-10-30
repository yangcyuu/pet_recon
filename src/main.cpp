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
      .voxel_size = {1.0f, 1.0f, 1.0f},
      .image_size = {160, 160, 240},
  });

  renderer.render("mich_recon.image3d");
}
