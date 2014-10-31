#define in_offs idx
if (IN_REAL_OFFS_VALID) {
  #if USE_HITS > 0
    int i = IN_REAL_OFFS;
    ATOM_ADD(&output[i], sum);
    atomic_inc(&hits[i]);
  #else
    sum /= (KX / SLIDE_X) * (KY / SLIDE_Y);
    ATOM_ADD(&output[IN_REAL_OFFS], sum);
  #endif
}
#undef in_offs
