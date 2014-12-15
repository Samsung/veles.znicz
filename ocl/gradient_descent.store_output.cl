#if (STORE_GRADIENT == OP_DEFAULT) || (STORE_GRADIENT == OP_STORE)
gradient[idx] = gd;
#elif STORE_GRADIENT == OP_ACCUMULATE
gradient[idx] += gd;
#elif STORE_GRADIENT == OP_FLUSH
gd += gradient[idx];
gradient[idx] = 0;
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
