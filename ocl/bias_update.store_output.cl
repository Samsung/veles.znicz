if (!tx) {
  sum += AS[0];
  dtype weight = bias[bx];
  dtype gd = -lr * (sum + gradient_step_l12(weight, factor_l12, l1_vs_l2));
  #define weights bias
  #define idx bx
  #include "gradient_descent.store_output.cl"
  #undef idx
  #undef weights
}
