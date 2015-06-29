#include "defines.cl"

{% if normalization == "pointwise" %}
  __constant const dtype mul[] = {{ "{" + coeffs[0] | join(", ") + "}" }};
  __constant const dtype sub[] = {{ "{" + coeffs[1] | join(", ") + "}" }};
{% elif normalization == "mean_disp" %}
  __constant const dtype mean[] = {{ "{" + coeffs[0] | join(", ") + "}" }};
  __constant const dtype disp[] = {{ "{" + coeffs[1] | join(", ") + "}" }};
{% elif normalization == "range_linear" %}
  __constant const dtype sub = {{ (coeffs[2] * coeffs[1] - coeffs[3] * coeffs[0]) / (coeffs[2] - coeffs[3]) }};
  __constant const dtype mul = {{ (coeffs[2] - coeffs[3]) / (coeffs[0] - coeffs[1]) }};
{% endif %}


inline dtype denormalize(dtype src, int index) {
  {% if normalization == "pointwise" %}
    return (src - sub[index]) / mul[index];
  {% elif normalization == "mean_disp" %}
    return src * disp[index] + mean[index];
  {% elif normalization == "none" %}
    return src;
  {% elif normalization == "range_linear" %}
      return (src - sub) * mul;
  {% else %}
    #error Unsupported normalization type
  {% endif %}
}
