#include "defines.cl"
#include "highlight.cl"

#ifndef INCLUDE_BIAS
#error "INCLUDE_BIAS should be defined"
#endif
#if INCLUDE_BIAS != 0
#error "INCLUDE_BIAS should be 0"
#endif

#ifndef WEIGHTS_TRANSPOSED
#error "WEIGHTS_TRANSPOSED should be defined"
#endif

#ifndef STORE_GRADIENT
#error "STORE_GRADIENT should be defined"
#endif

#ifndef APPLY_GRADIENT
#error "APPLY_GRADIENT should be defined"
#endif

#include "conv.cl"

// Just swap weights transposed flag
#if WEIGHTS_TRANSPOSED == 0
#undef WEIGHTS_TRANSPOSED
#define WEIGHTS_TRANSPOSED 1
#else
#undef WEIGHTS_TRANSPOSED
#define WEIGHTS_TRANSPOSED 0
#endif
#include "gradient_descent_conv.cl"
