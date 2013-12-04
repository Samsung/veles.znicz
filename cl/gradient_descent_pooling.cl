/// @brief Backpropagates max_pooling.
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
