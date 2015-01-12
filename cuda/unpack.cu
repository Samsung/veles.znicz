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
 *   BATCH: minibatch size
 */

#define KX_APP (1 + ((SX - KX + PAD_LEFT + PAD_RIGHT) / SLIDE_X))
#define KY_APP (1 + ((SY - KY + PAD_TOP + PAD_BOTTOM) / SLIDE_Y))
#define KERNEL_SIZE (KX * KY * N_CHANNELS)
#define IMG_SIZE (SX * SY * N_CHANNELS)
#define UNPACK_IMG_SIZE (KX_APP * KY_APP * KX * KY * N_CHANNELS)

/**
 * GridDim:
 *     x = KX_APP (number of kernel application along OX axis)
 *     y = KY_APP * BATCH (number of kernel application along OY axis)
 *     
 * BlockDim:
 *     x = KX * N_CHANNELS
 *     y = KY
 */
__global__ void DirectUnpack(const dtype *data, dtype *unpack_data) {
  int img_idx = blockIdx.y / KY_APP;
  int ch_idx = threadIdx.x % N_CHANNELS;
  int x = blockIdx.x * SLIDE_X + threadIdx.x / N_CHANNELS;
  int kernel_i = blockIdx.y % KY_APP;
  int y = kernel_i * SLIDE_Y + threadIdx.y;
  
  if (x >= PAD_LEFT && x < SX + PAD_LEFT &&
      y >= PAD_TOP && y < SY + PAD_TOP) {
    unpack_data[UNPACK_IMG_SIZE * img_idx +
                (kernel_i * KX_APP + blockIdx.x) * KERNEL_SIZE +
                threadIdx.y * KX * N_CHANNELS + threadIdx.x] =
        data[IMG_SIZE * img_idx +
             (y - PAD_TOP) * SX * N_CHANNELS + (x - PAD_LEFT) * N_CHANNELS + ch_idx];
  } else {
    unpack_data[UNPACK_IMG_SIZE * img_idx +
                (kernel_i * KX_APP + blockIdx.x) * KERNEL_SIZE +
                threadIdx.y * KX * N_CHANNELS + threadIdx.x] = 0;
  }
}


__global__ void ReverseUnpack(const dtype *data, dtype *unpack_data) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;

  int img_idx = ty / KX_APP / KY_APP;
  int kernel_j = SLIDE_X *  (ty % KX_APP);
  int kernel_i = SLIDE_Y * ((ty / KX_APP) % KY_APP);

  int ch_idx = tx % N_CHANNELS;
  int x = kernel_j + (tx / N_CHANNELS) % KX;
  int y = kernel_i + tx / N_CHANNELS / KX;
  
  if (x >= PAD_LEFT && x < SX + PAD_LEFT &&
      y >= PAD_TOP && y < SY + PAD_TOP) {
    unpack_data[ty * KERNEL_SIZE + tx] =
        data[IMG_SIZE * img_idx +
             ((y - PAD_TOP) * SX + x - PAD_LEFT) * N_CHANNELS + ch_idx];
  } else {
    unpack_data[ty * KERNEL_SIZE + tx] = 0;
  }
}
