#include "renderer.h"


int main() {
  auto renderer = Renderer({
      .mich_file = "D80-mouse-mich.dat",
      .offset = 0,
      .samples_per_crystal = 10,
      .samples_per_lor = 2,
      .iter_per_slice = 10,
      // .batch_size = 256,
      .use_sobol = true,
      //.linear_sampling = true,
      .linear_step = 2.0f,
      .tof_sigma = 0.0f,
      .voxel_size = {1.0f, 1.0f, 1.0f},
      .image_size = {80, 80, 140},
      .define = D80(),
  });

  renderer.render("d80_mich_recon.image3d");
}
