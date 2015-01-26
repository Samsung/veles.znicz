#if (SLIDE_X < KX) || (SLIDE_Y < KY)
#define USE_ATOMICS 1
#endif
#include "pooling_common.cuh"


/// @brief Backpropagates max pooling.
/// @param err_y error on current level.
/// @param err_h backpropagated error for previous layer.
/// @param h_offs indexes of err_h max values.
/// @details err_h should be filled with zeros before calling this function.
///          Should be defined externally:
///          SX - input image width,
///          SY - input image height,
///          N_CHANNELS - number of input channels,
///          KX - pooling kernel width,
///          KY - pooling kernel height,
///          SLIDE_X - kernel sliding by x-axis,
///          SLIDE_Y - kernel sliding by y-axis.
///          Kernel should be run as:
///          global_size = [number of elements in err_y],
///          local_size = None.
extern "C"
__global__ void gd_max_pooling(const dtype    *err_y,
                               dtype          *err_h,
                               const int      *h_offs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= OUTPUT_SIZE) {
    return;
  }
  #if (SLIDE_X >= KX) && (SLIDE_Y >= KY)
    err_h[h_offs[idx]] = err_y[idx];
  #else
    atomicAdd(&err_h[h_offs[idx]], err_y[idx]);
  #endif
}


/// @brief Backpropagates avg pooling.
/// @param err_y error on current level.
/// @param err_h backpropagated error for previous layer.
/// @details err_h should be filled with zeros before calling this function if
///          SLIDE_X < KX or SLIDE_Y < KY.
///          Should be defined externally:
///          SX - input image width,
///          SY - input image height,
///          N_CHANNELS - number of input channels,
///          KX - pooling kernel width,
///          KY - pooling kernel height,
///          SLIDE_X - kernel sliding by x-axis,
///          SLIDE_Y - kernel sliding by y-axis.
///          Kernel should be run as:
///          global_size = [number of elements in err_y],
///          local_size = None.
extern "C"
__global__ void gd_avg_pooling(const dtype    *err_y,
                               dtype          *err_h) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= OUTPUT_SIZE) {
    return;
  }
  int target_y = idx / (OUT_SX * N_CHANNELS);
  int target_x = idx % (OUT_SX * N_CHANNELS);

  int start_x = TARGET_PIXEL_X * SLIDE_X * N_CHANNELS + TARGET_CHANNEL,
      start_y = target_y % OUT_SY * SLIDE_Y;

  #if (OUT_SY - 1) * SLIDE_Y + KY == SY
  #define NY KY
  #else
  #define NY MIN(KY, SY - (target_y % OUT_SY) * SLIDE_Y)
  #endif

  #if (OUT_SX - 1) * SLIDE_X + KX == SX
  #define NX KX
  #else
  #define NX MIN(KX, SX - TARGET_PIXEL_X * SLIDE_X)
  #endif

  dtype avg = err_y[idx] / (NY * NX);

  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;
  #if (OUT_SY - 1) * SLIDE_Y + KY == SY
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
  #else
  for (int i = 0, y = start_y; (i < KY) && (y < SY); i++, y++, offs += SX * N_CHANNELS) {
  #endif
    #if (OUT_SX - 1) * SLIDE_X + KX == SX
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
    #else
    for (int j = 0, x = start_x; (j < KX) && (x < SX * N_CHANNELS); j++, x += N_CHANNELS) {
    #endif
      #if (SLIDE_X >= KX) && (SLIDE_Y >= KY)
        err_h[offs + x] = avg;
      #else
        atomicAdd(&err_h[offs + x], avg);
      #endif
    }
  }

  #undef NX
  #undef NY
}


#undef TARGET_CHANNEL
#undef TARGET_PIXEL_X
#undef OUT_SY
#undef OUT_SX
