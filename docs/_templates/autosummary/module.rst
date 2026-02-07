{{ fullname }}
{{ "=" * fullname|length }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% if packages %}
Subpackages
-----------

.. autosummary::
   :toctree: .
   :nosignatures:
   :undoc-members:
{% for p in packages %}
   {{ fullname }}.{{ p }}
{% endfor %}
{% endif %}

{% if modules %}
Submodules
----------

.. autosummary::
   :toctree: .
   :nosignatures:
   :undoc-members:
{% for m in modules %}
   {{ fullname }}.{{ m }}
{% endfor %}
{% endif %}
