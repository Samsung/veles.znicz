#include "defines.cl"


#define MINIMUM(a, b) ((a) < (b) ? (a) : (b))


#if SX % SLIDE_X == 0
#define OUT_SX (SX / SLIDE_X)
#else
#define OUT_SX (SX / SLIDE_X + 1)
#endif
#if SY % SLIDE_Y == 0
#define OUT_SY (SY / SLIDE_Y)
#else
#define OUT_SY (SY / SLIDE_Y + 1)
#endif

#define TARGET_PIXEL_X (target_x / N_CHANNELS)
#define TARGET_CHANNEL (target_x % N_CHANNELS)


/// @brief Does max pooling over convolutional layer output.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param h batch of input multichannel interleaved images.
/// @param y batch of output multichannel interleaved images.
/// @param h_offs indexes of y value in corresponding to it h array.
/// @details Should be defined externally:
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
void do_max_pooling(__global c_dtype /*IN*/ *h, __global c_dtype /*OUT*/ *y,
                    __global int /*OUT*/ *h_offs) {

  dtype max_absvle = -1;
  c_dtype max_vle = c_from_re(0);
  int max_offs = 0;
  int target_x = get_global_id(0),
      target_y = get_global_id(1);

  int start_x = TARGET_PIXEL_X * SLIDE_X * N_CHANNELS + TARGET_CHANNEL,
      start_y = target_y % OUT_SY * SLIDE_Y;
  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;

  #if (OUT_SY - 1) * SLIDE_Y + KY == SY
  // No partial windows at the bottom
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
  #else
  // There are partial windows at the bottom
  for (int i = 0, y = start_y; (i < KY) && (y < SY); i++, y++, offs += SX * N_CHANNELS) {
  #endif
    #if (OUT_SX - 1) * SLIDE_X + KX == SX
    // No partial windows at the right
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
    #else
    // There are partial windows at the right
    for (int j = 0, x = start_x; (j < KX) && (x < SX * N_CHANNELS); j++, x += N_CHANNELS) {
    #endif
      // TODO: continue here.
      c_dtype vle = h[offs + x];
      dtype absvle = c_norm(vle);
      if (absvle > max_absvle) {
        max_absvle = absvle;
        max_vle = vle;
        max_offs = offs + x;
      }
    }
  }

  int idx = target_y * OUT_SX * N_CHANNELS + target_x;
  y[idx] = max_vle;
  h_offs[idx] = max_offs;
}


/// @brief Does avg pooling over convolutional layer output.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param h batch of input multichannel interleaved images.
/// @param y batch of output multichannel interleaved images.
/// @details Should be defined externally:
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
void do_avg_pooling(__global c_dtype /*IN*/ *h, __global c_dtype /*OUT*/ *y) {

  c_dtype smm = c_from_re(0);
  int target_x = get_global_id(0),
      target_y = get_global_id(1);

  int start_x = TARGET_PIXEL_X * SLIDE_X * N_CHANNELS + TARGET_CHANNEL,
      start_y = target_y % OUT_SY * SLIDE_Y;
  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;

  #if (OUT_SY - 1) * SLIDE_Y + KY == SY
  // No partial windows at the bottom
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
  #else
  // There are partial windows at the bottom
  for (int i = 0, y = start_y; (i < KY) && (y < SY); i++, y++, offs += SX * N_CHANNELS) {
  #endif
    #if (OUT_SX - 1) * SLIDE_X + KX == SX
    // No partial windows at the right
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
    #else
    // There are partial windows at the right
    for (int j = 0, x = start_x; (j < KX) && (x < SX * N_CHANNELS); j++, x += N_CHANNELS) {
    #endif
      smm += h[offs + x];
    }
  }

  #if (OUT_SY - 1) * SLIDE_Y + KY == SY
  #define NY KY
  #else
  #define NY (target_y % OUT_SY < OUT_SY - 1 ? KY : MINIMUM(KY, SY - (OUT_SY - 1) * SLIDE_Y))
  #endif

  #if (OUT_SX - 1) * SLIDE_X + KX == SX
  #define NX KX
  #else
  #define NX (TARGET_PIXEL_X < OUT_SX - 1 ? KX: MINIMUM(KX, SX - (OUT_SX - 1) * SLIDE_X))
  #endif

  int idx = target_y * OUT_SX * N_CHANNELS + target_x;
  y[idx] = smm / (NX * NY);

  #undef NX
  #undef NY
}


#undef TARGET_CHANNEL
#undef TARGET_PIXEL_X
#undef OUT_SY
#undef OUT_SX


#undef MINIMUM
