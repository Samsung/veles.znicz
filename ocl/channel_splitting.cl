#include "defines.cl"
#include "highlight.cl"

// input: [n_pics][pic_h][pic_w][n_chans] array
// output: [n_pics][n_chans][pic_h][pic_W]
// runs in configuration [pic_h]

__kernel void split_channels(__global dtype* input, __global dtype* output,
                             int n_pics, int n_chans, int pic_h, int pic_w) {

  size_t i = get_global_id(0);

  for(int pic = 0; pic < n_pics; ++pic){
    int pic_index = pic * n_chans * pic_h * pic_w;
    for(int j = 0; j < pic_w; ++j) {
      for(int chan = 0; chan < n_chans; ++chan) {
        int nhwc_index = pic_index + i * pic_w * n_chans + j * n_chans + chan;
        int nchw_index = pic_index + chan * pic_h * pic_w + i * pic_w + j;
        output[nchw_index] = input[nhwc_index];
      }
    }
  }
}

__kernel void merge_channels(__global dtype* input, __global dtype* output,
                             int n_pics, int n_chans, int pic_h, int pic_w) {

  size_t i = get_global_id(0);

    for(int pic = 0; pic < n_pics; ++pic){
    int pic_index = pic * n_chans * pic_h * pic_w;
    for(int j = 0; j < pic_w; ++j) {
      for(int chan = 0; chan < n_chans; ++chan) {
        int nhwc_index = pic_index + i * pic_w * n_chans + j * n_chans + chan;
        int nchw_index = pic_index + chan * pic_h * pic_w + i * pic_w + j;
        output[nhwc_index] = input[nchw_index];
      }
    }
  }
}
