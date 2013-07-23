/*
 * RBM
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
 */

/// @brief Applies rand() over output.
/// @param y output of the layer.
/// @param rand array with random values distributed in [y.min(), y.max()].
/// @param y_low "0".
/// @param y_high "1".
/// @details Y should be defined.
__kernel //__attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void apply_rand(__global dtype *y, __global dtype *rand,
                const dtype y_low, const dtype y_high) {
   int offs = get_global_id(1) * Y + get_global_id(0);
   y[offs] = (y[offs] < rand[offs]) ? y_low : y_high;
}

/// @brief Multiplies backpropagated error by activation derivative.
/// @details err_y *= y * y * (-0.508262) + 1.143819
__kernel //__attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_y_update(__global dtype *err_y, __global dtype *y, __global dtype *rand,
                  const dtype window_size)
{
 int offs = get_global_id(1) * Y + get_global_id(0);
 dtype x = y[offs];
 err_y[offs] *= (fabs(x - rand[offs]) < window_size) ? x * x * (-0.508262f) + 1.143819f : 0;
}
