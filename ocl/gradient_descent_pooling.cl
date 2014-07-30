#if (SLIDE_X < KX) || (SLIDE_Y < KY)
#define USE_ATOMICS 1
#endif
#include "defines.cl"
#include "highlight.cl"


// Pool only full-size kernels
#define LAST_APPLY_X (SX - KX)
#define LAST_APPLY_Y (SY - KY)
#define OUT_SX (LAST_APPLY_X / SLIDE_X + 1)
#define OUT_SY (LAST_APPLY_Y / SLIDE_Y + 1)

#define TARGET_PIXEL_X (target_x / N_CHANNELS)
#define TARGET_CHANNEL (target_x % N_CHANNELS)


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
__kernel
void gd_max_pooling(__global const dtype    /* IN */    *err_y,
                    __global dtype         /* OUT */    *err_h,
                    __global const int      /* IN */    *h_offs) {
  int idx = get_global_id(0);
  #if (SLIDE_X >= KX) && (SLIDE_Y >= KY)
  err_h[h_offs[idx]] = err_y[idx];
  #else
  ATOM_ADD(&err_h[h_offs[idx]], err_y[idx]);
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
///          global_size = [out_width, out_height],
///          local_size = None.
__kernel
void gd_avg_pooling(__global const dtype    /* IN */    *err_y,
                    __global dtype         /* OUT */    *err_h) {
  int target_x = get_global_id(0),
      target_y = get_global_id(1);
  int start_x = TARGET_PIXEL_X * SLIDE_X * N_CHANNELS + TARGET_CHANNEL,
      start_y = target_y % OUT_SY * SLIDE_Y;

  int idx = target_y * OUT_SX * N_CHANNELS + target_x;
  dtype avg = err_y[idx] / (KX * KY);

  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;

  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
      #if (SLIDE_X >= KX) && (SLIDE_Y >= KY)
      err_h[offs + x] = avg;
      #else
      ATOM_ADD(&err_h[offs + x], avg);
      #endif
    }
  }
}


#undef TARGET_CHANNEL
#undef TARGET_PIXEL_X
#undef OUT_SY
#undef OUT_SX


KERNEL_CLEAR(err_input_clear, dtype)
