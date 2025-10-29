#include "raw_data.h"
#include "renderer.h"
#include "utils.h"


int main() {

  auto renderer = Renderer({
      .mich_file = "coin_all.image3d",
      .offset = 6,
      .samples_per_crystal = 1,
      .samples_per_lor = 1,
      .iter_per_slice = 1,
      .use_sobol = true,
      .tof_sigma = 0.0f,
      .voxel_size = {1.0f, 1.0f, 1.0f},
      .image_size = {240, 240, 240},
  });

  renderer.render("mich_recon.image3d");
}
