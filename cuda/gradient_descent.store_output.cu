#if ACCUMULATE_GRADIENT > 0
dtype acc = acc_beta ? acc_beta * accumulated_gradient[idx] : 0;
acc += acc_alpha * gd;
accumulated_gradient[idx] = acc;

gd *= gd_beta;
gd += gd_alpha * acc;
#endif

#if USE_MOMENT > 0
gd += gradient_with_moment[idx] * moment;
gradient_with_moment[idx] = gd;
#endif

#if APPLY_GRADIENT > 0
weights[idx] = weight + gd;
#endif
