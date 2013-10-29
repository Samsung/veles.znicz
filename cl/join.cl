/// @brief Joins two inputs into one.
/// @param output Output.
/// @param a First input.
/// @param b Second input.
/// @param output_offs Where to put output.
/// @param a_size Size of a.
/// @details etype is an element type and should be defined externally.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
__kernel
void join2(__global etype *output, __global etype *a, __global etype *b,
           const int output_offs, const int a_size) {
  int offs = get_global_id(0);
  output[output_offs + offs] = offs < a_size ? a[offs] : b[offs - a_size];
}
