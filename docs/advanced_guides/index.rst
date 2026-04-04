.. _advanced_guides:

===============
Advanced Guides
===============

This section is for users who want implementation-level detail and extension
patterns. The pages here are intentionally technical and assume you are already
comfortable with the standard user-facing guides.

Use this section when you need to:

- inspect or reason about internal data models,
- extend compiler behavior,
- author custom lowering paths,
- profile/diagnose symbolic compilation behavior.

The goal is to help advanced users move from API usage to architecture-level
reasoning: understanding not only *what* works, but *why* it works and where to
intervene safely when extending internals.


Quick links
===========

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Structs and data containers
      :link: structs
      :link-type: doc

      Immutable JAX-native data containers, field kinds, serialisation
      contracts, and advanced container lifecycle semantics.

   .. grid-item-card:: Symbolic compiler internals
      :link: symbolic_compiler/index
      :link-type: doc

      Technical documentation of the declarative symbolic operator IR,
      compiler pipeline, cache/signature model, and extension points.


.. toctree::
   :hidden:
   :maxdepth: 2

   structs
   symbolic_compiler/index
