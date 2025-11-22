#include "renderer.h"

int main() {
  RendererParams params;
  params.define = E180();
  params.image_grid = Grids<3>::create_by_center_spacing_size({0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {320, 320, 400});
  params.norm_factor_path = "graphic_test/norm_factors.dat";
  params.attn_map_path = "20251103_wellcounterdata/CTRecon4888/ct_attn_img.dat";
  params.attn_map_offset = 6;
  params.mich_path = "20251103_wellcounterdata/Slice4798/coin_all.image3d";
  params.mich_offset = 6;
  params.attn_path = "20251103_wellcounterdata/CTRecon4888/ct_attn_mich.dat";
  params.attn_offset = 6;
  params.delay_path = "20251103_wellcounterdata/Slice4798/delay_all.image3d";
  params.delay_offset = 6;
  params.norm_path = "graphic_test/normCoff.bin";
  params.norm_offset = 0;
  params.psf_sigma = 0.1f;
  params.sub_lor_num = 10;
  params.sample_num = 8;
  params.max_iterations = 10;
  params.batch_size = 104 * 104 * 5;
  params.use_sobol = true;
  params.compute_scatter = true;

  Renderer renderer(params);
  renderer.render(true);
  renderer.save("reconstructed_image.image3d");
}
