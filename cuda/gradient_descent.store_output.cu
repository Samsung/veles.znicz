#if ACCUMULATE_GRADIENT == OP_STORE
accumulated_gradient[idx] = gd;
#elif ACCUMULATE_GRADIENT == OP_ADD
accumulated_gradient[idx] += gd;
#elif ACCUMULATE_GRADIENT == OP_FLUSH
gd += accumulated_gradient[idx];
accumulated_gradient[idx] = 0;
#endif

#ifndef USE_MOMENT
#error "USE_MOMENT should be defined"
#endif
#if USE_MOMENT > 0
gd += gradient_with_moment[idx] * moment;
gradient_with_moment[idx] = gd;
#endif

#if APPLY_GRADIENT > 0
weights[idx] = weight + gd;
#endif
