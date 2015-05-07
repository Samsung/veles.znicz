/**
 * Should be defined:
 *   SX: input image width
 *   SY: input image height
 *   N_CHANNELS: number of input channels
 *   KX: kernel width
 *   KY: kernel height
 *   SLIDE_X: kernel sliding by x-axis
 *   SLIDE_Y: kernel sliding by y-axis
 *   PAD_TOP: padding size at the top of each image
 *   PAD_BOTTOM: padding size at the bottom of each image
 *   PAD_LEFT: padding size at the left of each image
 *   PAD_RIGHT: padding size at the right of each image
 */

#define KX_APP (1 + ((SX - KX + PAD_LEFT + PAD_RIGHT) / SLIDE_X))
#define KY_APP (1 + ((SY - KY + PAD_TOP + PAD_BOTTOM) / SLIDE_Y))
#define KERNEL_SIZE (KX * KY * N_CHANNELS)
#define IMG_SIZE (SX * SY * N_CHANNELS)


/**
 * Grid:
 *  1D with total size >= (KX_APP * KY_APP * number_of_images_to_unpack) * (KX * KY * N_CHANNELS)
 */
extern "C"
__global__ void Unpack1D(const dtype *data, dtype *unpack_data, const int limit) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // we are processing small number of images at a time, so size_t is not required

  int ty = idx / KERNEL_SIZE;
  int tx = idx % KERNEL_SIZE;

  int img_idx = ty / KX_APP / KY_APP;
  int kernel_j = SLIDE_X *  (ty % KX_APP);
  int kernel_i = SLIDE_Y * ((ty / KX_APP) % KY_APP);

  int ch_idx = tx % N_CHANNELS;
  int x = kernel_j + (tx / N_CHANNELS) % KX;
  int y = kernel_i + tx / N_CHANNELS / KX;

  if (idx < limit) {
    unpack_data[idx] = (x >= PAD_LEFT && x < SX + PAD_LEFT && y >= PAD_TOP && y < SY + PAD_TOP) ?
        data[IMG_SIZE * img_idx + ((y - PAD_TOP) * SX + x - PAD_LEFT) * N_CHANNELS + ch_idx] : 0;
  }
}


// apply_bias_with_activation
#ifndef ACCUMULATE_GRADIENT
#include "all2all/forward.cu"
#endif
