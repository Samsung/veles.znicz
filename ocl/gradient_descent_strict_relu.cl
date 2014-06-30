#include "defines.cl"
#include "highlight.cl"
/// Strict ReLU back propagation
/// @brief Updates backpropagated error by activation derivative.
/// @details err_y *= (y > 0) ? 1 : 0
__kernel
void err_y_update(__global dtype         /* OUT */    *err_y,
                  __global const dtype    /* IN */    *y) {
  int offs = get_global_id(0);
  if(y[offs] <= 0) {
    err_y[offs] = 0;
  }
}
