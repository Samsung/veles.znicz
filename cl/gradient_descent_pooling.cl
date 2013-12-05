/// @brief Backpropagates max pooling.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param err_y error on current level.
/// @param err_h backpropagated error for previous layer.
/// @param h_offs indexes of err_h max values.
/// @details err_h should be filled with zeros before calling this function.
__kernel
void gd_max_pooling(__global c_dtype /*IN*/ *err_y, __global c_dtype /*OUT*/ *err_h,
                    __global int /*IN*/ *h_offs) {
  int idx = get_global_id(0);
  err_h[h_offs[idx]] = err_y[idx];
}


/// @brief Backpropagates avg pooling.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param err_y error on current level.
/// @param err_h backpropagated error for previous layer.
/// @details Should be defined externally:
///          SX - input image width,
///          SY - input image height,
///          N_CHANNELS - number of input channels,
///          KX - pooling kernel width,
///          KY - pooling kernel height.
__kernel
void gd_avg_pooling(__global c_dtype /*IN*/ *err_y, __global c_dtype /*OUT*/ *err_h) {
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
  int target_x = get_global_id(0) % (OUT_SX * N_CHANNELS),
      target_y = get_global_id(0) / (OUT_SX * N_CHANNELS);
  #define TARGET_PIXEL (target_x / N_CHANNELS)
  #define TARGET_CHANNEL (target_x % N_CHANNELS)
  int start_x = TARGET_PIXEL * N_CHANNELS * KX + TARGET_CHANNEL,
      t_y = target_y % OUT_SY,
      start_y = t_y * KY;

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
  c_dtype avg = err_y[idx] / (NY * NX);

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
      err_h[offs + x] = avg;
    }
  }
}
