#include "defines.cl"

{% if normalization == "pointwise" %}
  __constant const dtype mul[] = {{ "{" + coeffs[0] | join(", ") + "}" }};
  __constant const dtype add[] = {{ "{" + coeffs[1] | join(", ") + "}" }};
{% elif normalization == "mean_disp" %}
  __constant const dtype mean[] = {{ "{" + coeffs[0] | join(", ") + "}" }};
  __constant const dtype disp[] = {{ "{" + coeffs[1] | join(", ") + "}" }};
{% endif %}


inline dtype denormalize(dtype src) {
  {% if normalization == "pointwise" %}
    return (src - add[index]) / mul[index];
  {% elif normalization == "mean_disp" %}
    return src * disp[index] + mean[index];
  {% elif normalization == "none" %}
    return src;
  {% endif %}
  // compilation error
}