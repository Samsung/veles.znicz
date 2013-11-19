/*
 * General definitions.
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
 */

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifdef COMPLEX

inline dtype c_re(c_dtype a) {
  return a.x;
}

inline c_dtype c_from_re(dtype re) {
  return (c_dtype)(re, 0);
}

inline c_dtype c_mul(c_dtype a, c_dtype b) {
  return (c_dtype)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline c_dtype c_div(c_dtype a, c_dtype b) {
  dtype d = b.x * b.x + b.y * b.y;
  return (c_dtype)((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

inline c_dtype c_exp(c_dtype a) {
  dtype d = exp(a.x);
  return (c_dtype)(cos(a.y) * d, sin(a.y) * d);
}

inline c_dtype c_tanh(c_dtype a) {
  dtype s = sign(a.x);
  c_dtype z = (c_dtype)(a.x * s, a.y);
  c_dtype ze = c_exp(z * (dtype)-2.0);
  z = c_div((c_dtype)(1, 0) - ze, (c_dtype)(1, 0) + ze);
  z.x *= s;
  return z;
}

inline dtype c_norm2(c_dtype a) {
  return a.x * a.x + a.y * a.y;
}

inline dtype c_norm(c_dtype a) {
  return sqrt(a.x * a.x + a.y * a.y);
}

#else

#define c_re(a) (a)
#define c_from_re(re) (re)
#define c_mul(a, b) ((a) * (b))
#define c_div(a, b) ((a) / (b))
#define c_exp(a) exp(a)
#define c_tanh(a) tanh(a)
#define c_norm2(a) ((a) * (a))
#define c_norm(a) fabs(a)

#endif
