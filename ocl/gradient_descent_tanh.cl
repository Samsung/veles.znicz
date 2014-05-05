#include "defines.cl"
#include "highlight.cl"

/// @brief Updates backpropagated error by activation derivative.
/// @details err_y *= y * y * (-0.388484177) + 1.14381894
__kernel
void err_y_update(__global c_dtype /*OUT*/ *err_y, __global c_dtype /*IN*/ *y) {
  int offs = get_global_id(0);
  c_dtype x = y[offs];
  err_y[offs] = c_mul(err_y[offs], c_mul(x, x) * (dtype)(-0.388484177) + c_from_re((dtype)1.14381894));
}
