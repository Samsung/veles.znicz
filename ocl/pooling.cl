#include "defines.cl"

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
///          KY - pooling kernel height.
__kernel
void do_max_pooling(__global c_dtype /*IN*/ *h, __global c_dtype /*OUT*/ *y,
                    __global int /*OUT*/ *h_offs) {
  #if SX % KX == 0
  #define OUT_SX (SX / KX)
  #else
  #define OUT_SX (SX / KX + 1)
  #endif
  #if SY % KY == 0
  #define OUT_SY (SY / KY)
  #else
  #define OUT_SY (SY / KY + 1)
  #endif
  dtype max_absvle = -1;
  c_dtype max_vle = c_from_re(0);
  int max_offs = 0;
  int target_x = get_global_id(0),
      target_y = get_global_id(1);
  #define TARGET_PIXEL (target_x / N_CHANNELS)
  #define TARGET_CHANNEL (target_x % N_CHANNELS)
  int start_x = TARGET_PIXEL * N_CHANNELS * KX + TARGET_CHANNEL,
      start_y = target_y % OUT_SY * KY;
  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;

  #if SY % KY == 0
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
  #else
  for (int i = 0, y = start_y; (i < KY) && (y < SY); i++, y++, offs += SX * N_CHANNELS) {
  #endif
    #if SX % KX == 0
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
    #else
    for (int j = 0, x = start_x; (j < KX) && (x < SX * N_CHANNELS); j++, x += N_CHANNELS) {
    #endif
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
///          KY - pooling kernel height.
__kernel
void do_avg_pooling(__global c_dtype /*IN*/ *h, __global c_dtype /*OUT*/ *y) {
  #if SX % KX == 0
  #define OUT_SX (SX / KX)
  #else
  #define OUT_SX (SX / KX + 1)
  #endif
  #if SY % KY == 0
  #define OUT_SY (SY / KY)
  #else
  #define OUT_SY (SY / KY + 1)
  #endif
  c_dtype smm = c_from_re(0);
  int target_x = get_global_id(0),
      target_y = get_global_id(1);
  #define TARGET_PIXEL (target_x / N_CHANNELS)
  #define TARGET_CHANNEL (target_x % N_CHANNELS)
  int start_x = TARGET_PIXEL * N_CHANNELS * KX + TARGET_CHANNEL,
      t_y = target_y % OUT_SY,
      start_y = t_y * KY;
  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;

  #if SY % KY == 0
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
  #else
  for (int i = 0, y = start_y; (i < KY) && (y < SY); i++, y++, offs += SX * N_CHANNELS) {
  #endif
    #if SX % KX == 0
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
    #else
    for (int j = 0, x = start_x; (j < KX) && (x < SX * N_CHANNELS); j++, x += N_CHANNELS) {
    #endif
      smm += h[offs + x];
    }
  }

  #if SY % KY == 0
  #define NY KY
  #else
  #define NY ((t_y == (SY / KY)) ? SY % KY : KY)
  #endif

  #if SX % KX == 0
  #define NX KX
  #else
  #define NX ((TARGET_PIXEL == (SX / KX)) ? SX % KX : KX)
  #endif

  int idx = target_y * OUT_SX * N_CHANNELS + target_x;
  y[idx] = smm / (NX * NY);
}
