#include "defines.cl"
#include "highlight.cl"

/// @brief Updates backpropagated error by activation derivative.
/// @details err_y *= 1.0 - exp(-y)
__kernel
void err_y_update(__global dtype         /* OUT */    *err_y,
                  __global const dtype    /* IN */    *y) {
  int offs = get_global_id(0);
  err_y[offs] *= (dtype)1.0 - exp(-y[offs]);
}
