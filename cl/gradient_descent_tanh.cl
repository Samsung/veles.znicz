/*
 * Updates backpropagated error by activation derivative.
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
 */


/// @brief err_y *= y * y * (-0.388484177) + 1.14381894
__kernel //__attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_y_update(__global c_dtype /*OUT*/ *err_y, __global c_dtype /*IN*/ *y) {
  int offs = get_global_id(1) * Y + get_global_id(0);
  c_dtype x = y[offs];
  err_y[offs] = c_mul(err_y[offs], c_mul(x, x) * (dtype)(-0.388484177) + c_from_re((dtype)1.14381894));
}
