#include "defines.cl"
#include "highlight.cl"

/// @brief Updates backpropagated error by activation derivative.
/// @details err_y *= y * y * (-0.388484177) + 1.14381894
__kernel
void err_y_update(__global dtype         /* OUT */    *err_y,
                  __global const dtype    /* IN */    *y) {
  int offs = get_global_id(0);
  dtype x = y[offs];
  err_y[offs] = err_y[offs] * (x * x * (dtype)(-0.388484177) + (dtype)1.14381894);
}
