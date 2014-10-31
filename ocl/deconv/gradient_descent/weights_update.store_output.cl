dtype weight = weights[idx];
dtype gd = -lr * (sum + gradient_step_l12(weight, factor_l12, l1_vs_l2)
#if USE_ORTHO > 0
  #if WEIGHTS_TRANSPOSED > 0
           + gradient_step_ortho(weight, factor_ortho, get_global_id(1), Y, col_sums)
  #else
           + gradient_step_ortho(weight, factor_ortho, get_global_id(0), Y, col_sums)
  #endif
#endif
                 );
#if STORE_GRADIENT > 0
  gd += gradient[idx] * gradient_moment;
  gradient[idx] = gd;
#endif
#if APPLY_GRADIENT > 0
  weights[idx] = weight + gd;
#endif
