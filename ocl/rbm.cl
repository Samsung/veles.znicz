#include "defines.cl"

/// @brief Applies rand() over output (makes sense for real numbers only).
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param y output of the layer.
/// @param rand array with random values distributed in [y.min(), y.max()].
/// @param y_low "0".
/// @param y_high "1".
/// @details Y should be defined.
__kernel
void apply_rand(__global dtype *y, __global dtype *rand,
                const dtype y_low, const dtype y_high) {
   int offs = get_global_id(1) * Y + get_global_id(0);
   y[offs] = (y[offs] < rand[offs]) ? y_low : y_high;
}
