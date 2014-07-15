#ifndef _GRADIENT_DESCENT_COMMON_
#define _GRADIENT_DESCENT_COMMON_

#include "defines.cl"
#include "highlight.cl"


#ifndef INCLUDE_BIAS
#error "INCLUDE_BIAS should be defined"
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


#define gradient_step(weight, gradient, lr, lr_x_l, l1_vs_l2) (gradient * lr + lr_x_l * (((dtype)1.0 - l1_vs_l2) * weight + (dtype)0.5 * l1_vs_l2 * sign(weight)))


#endif  // _GRADIENT_DESCENT_COMMON_
