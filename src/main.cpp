#include "renderer.h"


int main() {
  auto renderer = Renderer({
      .mich_file = "D80-mouse-mich.dat",
      .offset = 0,
      .samples_per_crystal = 1,
      .samples_per_lor = 40,
      .iter_per_slice = 10,
      // .batch_size = 256,
      .use_sobol = true,
      .tof_sigma = 0.0f,
      .voxel_size = {1.0f, 1.0f, 1.0f},
      .image_size = {120, 120, 160},
      .define = D80(),
  });

  renderer.render("d80_mich_recon.image3d");
}
