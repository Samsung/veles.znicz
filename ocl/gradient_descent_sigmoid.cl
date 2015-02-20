#include "defines.cl"
#include "highlight.cl"

/// @brief Updates backpropagated error by activation derivative.
/// @details err_y *= y * (1 - y)
__kernel
void err_y_update(__global dtype         /* OUT */    *err_y,
                  __global const dtype    /* IN */    *y) {
  int offs = get_global_id(0);
  dtype x = y[offs];
  err_y[offs] *= x * ((dtype)1.0 - x);
}
