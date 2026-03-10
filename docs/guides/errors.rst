.. _errors:

==============
Errors
==============

This page lists some of the errors and warnings that you might encounter when using
neuraLQX. For each error below, we detail

- what the error means,
- when it occurs,
- and how to fix it (often with example snippets when possible).

The goal is to help you quickly identify and resolve issues in your simulations. If you encounter
something that is not documented here, reach out to us through our `GitHub page <https://www.github.com/waleed-sh/neuralqx>`_.

.. raw:: html

   <hr style="margin: 20px 0;">

DeniedExperimentalModuleImportError
------------------------------------------

**Description**

This error arises when you try to import the ``experimental`` module of neuraLQX.

The features in this module are currently under development, and their stability is not yet guaranteed.
Moreover, they might change without prior notice. Hence, this module is protected and cannot be
imported unless explicitly enabled.

**Cause**

Trying to run something like:

.. code-block:: python

   import neuralqx.experimental as nqxx

will raise:

.. code-block:: text

   DeniedExperimentalModuleImportError: experimental module import is disabled

**Fix**

To use experimental features, you must set the environment variable ``NQX_EXPERIMENTAL=1``
**before importing** neuralqx.

Example:

.. code-block:: python

   import os
   os.environ['NQX_EXPERIMENTAL'] = '1'

   import neuralqx as nqx

   # now this works
   import neuralqx.experimental as nqxx

.. raw:: html

   <hr style="margin: 30px 0;">


DeniedExperimentalFeatureError
-----------------------------------

**Description**

Sometimes, some experimental features may be more safe than others, and in this case they are
placed in the public API. However, due to their testing not being fully completed, we still mark
them as experimental using an ``@experimental`` decorator.
Even though they are not in the ``experimental`` module, they are treated as
such.

**Cause**

Trying to run something that is marked as experimental within neuraLQX:

.. code-block:: python

   @experimental
   def some_dummy_function(): ...

   res = some_dummy_function()

will raise an ``DeniedExperimentalFeatureError`` error.

**Fix**

To use experimental features, you have to set the environment variable ``NQX_EXPERIMENTAL`` to ``1``
**before** importing neuralqx. For example, in your script, you would do this


Example:

.. code-block:: python

    import os
    os.environ['NQX_EXPERIMENTAL'] = '1'

    import neuralqx as nqx

    # now, the experimentally marked some_dummy_function() should work
.. raw:: html

   <hr style="margin: 30px 0;">


IncorrectMonitoringValueError
-----------------------------------

**Description**

The ``StateExportCallback`` in neuraLQX allows you to export a frozen snapshot of the state if the
chosen monitored loss is equal to or below a certain threshold you have specified. If the monitored
you have specified is not from the allowed list of metrics available (`Mean`, `Sigma`, `R_hat`,
`TauCorr`), you will see this error.

**Cause**

Trying to use the ``StateExportCallback`` as below

.. code-block:: python

    from neuralqx.callbacks import StateExportCallback

    my_callback = StateExportCallback(monitor = 'Acceptance', target = 0.3, solver_object)

will raise an ``IncorrectMonitoringValueError`` error because ``'Acceptance'`` is not from the list
of allowed values to monitor.

**Fix**

To fix this, only use one of those four values for the value of
the `monitor` parameter when instantiating the `StateExportCallback`.

.. raw:: html

   <hr style="margin: 30px 0;">


AcceptanceUnavailableError
-----------------------------------

**Description**


The `AcceptanceCallback` of neuraLQX appends to your training log the acceptance rate of your
sampler. This error arises because the `AcceptanceCallback` is not compatible with sampler of type
`ExactSampler`, as they do not have acceptance criteria (they use exact inference methods).

**Cause**

Trying to use this callback with a sampler of type ``ExactSampler`` like this

.. code-block:: python

    from neuralqx.callbacks import AcceptanceCallback

    my_callback = AcceptanceCallback(my_exact_sampler)

**Fix**

will raise an ``AcceptanceUnavailableError`` error.

To use this callback, please choose another, Metropolis-type, sampler.


Example:

.. code-block:: python

    from neuralqx.callbacks import AcceptanceCallback

    # now this should be fine
    my_callback = AcceptanceCallback(my_metropolis_local_sampler)

.. raw:: html

   <hr style="margin: 30px 0;">


IncorrectEdgeFormatError
-----------------------------------

**Description**

You will encounter this error if you attempt to create a graph with an unrecognised format for the
list of edges.

**Cause**

The accepted formats in neuraLQX are as follows:

- For planar graphs
    - The set of all edges must be a python list
    - An edge must be specified by a python list
    - An edge is directed and specified as ``[a, b]``, where  ``a`` is the vertex it is emanating from
      and ``b`` the vertex it is incident at. The vertices are labeled by **positive** integers.


- For non-planar graphs
    - The set of all edges must be a python list
    - An edge must be specified by a python list
    - An edge is directed and specified as ``[(x, y, z), (a, b, c)]``, where now ``(x, y, z)`` denote
      the coordinates of the vertex it emanates from and ``(a, b, c)`` the coordinates of the vertex
      it is incident at. All coordinate values must be **positive** integers.

Attempting to run the following code will raise an error:

.. code-block:: python

    import neuralqx as nqx

    # example of bad non-planar edges, the -1 is not allowed
    edges = [((0, 0, 0), (0, 0, 1)), ((0, 0, 0), (0, 0, -1))]

    # this will raise the error
    graph = nqx.graph.Graph(edges)

**Fix**

Provide a correctly formatted oriented edge list as described above

Example:

.. code-block:: python

    import neuralqx as nqx

    # example of good non-planar edges
    edges = [((0, 0, 0), (0, 0, 1)), ((0, 0, 0), (0, 1, 0))]

    # this will work
    graph = nqx.graph.Graph(edges)


.. raw:: html

   <hr style="margin: 30px 0;">


UnspecifiedGaugeFixingError
-----------------------------------

**Description**

This error will arise if you try to create a gauge invariant Hilbert space but do not specify the
gauge fixing array necessary for filtering out and choosing the states which to remain in the gauge
invariant subspace or allow neuraLQX to automatically do the gauge fixing for you.

**Cause**

Trying to create a gauge invariant Hilbert space as follows

.. code-block:: python

    # note how no gauge fixing is provided and no automatic gauge fixing is instructed
    H = nqx.hilbert.u1.HilbertU1(my_graph, gauge_invariant = True, cutoff = 10)

will raise an ``UnspecifiedGaugeFixingError`` error.

**Fix**

You have to options here, you can either specify a gauge fixing array by hand, or allow neuraLQX
to do this for you (we recommend the latter for complicated graphs!). This can be done as follows


Example (by-hand):

.. code-block:: python

    # specify an array of gauge fixing which explicitly states which edges fix which edges
    H = nqx.hilbert.u1.HilbertU1(my_graph, gauge_invariant = True, cutoff = 10, gauge_fixing = my_gauge_fixing_array)

Example (automatic):

.. code-block:: python

    # let neuraLQX do the gauge fixing for you
    H = nqx.hilbert.u1.HilbertU1(my_graph, gauge_invariant = True, cutoff = 10, auto_constraint = True)

.. raw:: html

   <hr style="margin: 30px 0;">


CyclicGaugeFixingError
-----------------------------------

**Description**

When using gauge invariant Hilbert spaces, one way to go is to provide a gauge fixing array explicitly.
This gauge fixing will be used when filtering out states at initialisation of your Hilbert space but
als when reimposing the gauge fixing on a given basis state (for whatever reason). Reimposing the
gauge fixing must be applied in topological order (e.g. a slave edge should only be fixed after all
its dependents have been updated). This is called topological ordering. Topological ordering can
only be applied if your gauge fixing array does not contain a cycle (in other words, it must form a
directed acyclic graph).


**Cause**

You will encounter this error if your gauge fixing array has a cycle (e.g. edge A depends on
B, B depends on C and C depends on A).

**Fix**

To fix this, you have to rewrite your gauge fixing array such
that it contains no cycle. If this is not possible, you have to choose a different orientation for
your graph and therefore end up with a different gauge fixing.

Alternatively (and much easier), you can keep your graph orientation and set ``auto_constraint = True``
when creating your gauge invariant Hilbert space. This way, neuraLQX will attempt to impose a gauge
fixing which does not contain any cycles.

.. raw:: html

   <hr style="margin: 30px 0;">

IncorrectGaugeFixingArrayError
-----------------------------------

**Description**

neuraLQX accepts gauge fixing arrays which have a very specific format. The total gauge fixing
array must be a python list. Inside it, it should contain at least one gauge condition. Any gauge
condition should be a list of two lists: the first is **always** of size 1, and the next is of
arbitrary size. The elements of those lists should be of string type, and they represent the
label of the edges being fixed.

- Allowed gauge condition: ``[['(3, 2)'], ['(3, 4)', '(0, 3)']]``

- NOT allowed gauge conditions:
    - More than one edge on the left: ``[['(3, 2)', '(3, 4)'], ['(0, 3)']]``
    - Incorrect data type (not string): ``[[(3, 2)], [(3, 4), (0, 3)]]``
    - Incorrect data structure (tuple instead of lists): ``[('(3, 2)'), ('(3, 4)', '(0, 3)')]`` or ``(['(3, 2)'], ['(3, 4)', '(0, 3)'])``

Further, any edge string ``'(a, b)`` in any gauge condition can be preceeded by a minus sign (so
``'-(a, b)'`` is allowed). For example, ``[['(3, 2)'], ['-(3, 4)', '(0, 3)']]`` means that the quantum
number of the edge ``(3, 2)`` is equal to the quantum number of the edge ``(0, 3)`` (mod) plus the
**minus** of the quantum number of the edge ``(3, 4)``.

Lastly, if you specify a token ``N`` anywhere in the right hand side of the gauge condition, then the
quantum numbers of all following edges will be subtracted instead of added.

**Cause**

If your gauge fixing array deviates from the format listed above, you will see this error, for example

.. code-block:: python

    gauge_fixing = [
        [['(3, 2)', '(2, 1)'], ['-(3, 4)', '(0, 3)']]
    ]

will raise a ``IncorrectGaugeFixingArrayError`` error, as there are two edges specified in the
left hand side list.

**Fix**

To fix this, adhere to the format defined above. In the example, this translates to

.. code-block:: python

    # just an example, depends on graph orientation
    gauge_fixing = [
        [['(3, 2)'], ['-(3, 4)', '(0, 3)', '-(2, 1)']]
    ]


.. raw:: html

   <hr style="margin: 30px 0;">

InvalidFreeEdgeSelectionError
-----------------------------------

**Description**

In some Metropolis samplers (e.g. ``MetropolisLocal``) a function called ``flip_state`` is used to
generate the new proposed state. For unconstrained Hilbert spaces, this is provided from NetKet. For
constrained Hilbert spaces, this is custom made. In this custom implementation, you can specify how
many edges to "flip".

If you have requested to flip a number of edges ``M > N``, where ``N`` is the number of **free** edges in
your graph, you will see this error.

**Cause**

Example of a bad-flip in a gauge invariant Hilbert space assumed to have only 3 free edges

.. code-block:: python

    # attempting to flip 5 states in the configuration σ
    σp = H.flip_state(σ, jax.random.PRNGKey(0), number_of_edges = 5)

This will raise an ``InvalidFreeEdgeSelectionError`` error.

**Fix**

To fix this, always specify the number of edges to be flipped at a time to be at most as
many as the free edges. In the example above, ``number_of_edges`` should be at most 3.

.. raw:: html

   <hr style="margin: 30px 0;">

InvalidEdgeSelectionError
-----------------------------------

**Description**

This error is similar to the ``InvalidFreeEdgeSelectionError`` above, except now you have asked to
flip more edges than there are edges in the graph in total.

**Cause**

Example of a bad-flip in a gauge invariant Hilbert space on a graph which has in total 10 edges

.. code-block:: python

    # attempting to flip 5 states in the configuration σ
    σp = H.flip_state(σ, jax.random.PRNGKey(0), number_of_edges = 15)

This will raise an ``InvalidEdgeSelectionError`` error.

**Fix**

To fix this, always specify the number of edges to be flipped at a time to be at most as
many free edges (let alone edges in the entire graph...).
In the example above, ``number_of_edges`` should be at most equal to the number of free edges in
your graph (and never above the number of total edges in your graph).

.. raw:: html

   <hr style="margin: 30px 0;">


CrossProductInHigherDimensionsError
------------------------------------------

**Description**

Some operators in LQG (e.g. volume) have a cross product term in them. This cross product can only
be computed for specific gauge groups (e.g. in the weak coupling model, this means a :math:`U(1)^3` gauge
group. The operators which rely on this cross product will not be available if you choose the gauge
dimensions to be, for example, 2 or 4.

**Cause**

Trying to run the following code with an ``lqx`` object for the 4d weak coupling limit model

.. code-block:: python

    # note the gauge_dimensions being 4
    gauge_group = nqx.gauge_groups.u1.U1GaugeGroup(H, gauge_dimensions = 4, is_4d = True)

    lqx = nqx.lqx.EuclideanWCL(H, gaugeGroup, spacetime_dims = 4)

    # this will raise an error
    vol = lqx.model.volume_operator_at_vertex(0)

will raise an ``CrossProductInHigherDimensionsError`` error.

**Fix**

To fix this, use the correct required gauge dimensions for the desired operator in your model, or
you have to implement your desired operator from scratch for the new gauge dimensions.

Example:

.. code-block:: python

    # note the gauge_dimensions being 3
    gauge_group = nqx.gauge_groups.u1.U1GaugeGroup(H, gauge_dimensions = 3, is_4d = True)

    lqx = nqx.lqx.EuclideanWCL(H, gaugeGroup, spacetime_dims = 4)

    # this will work
    vol = lqx.model.volume_operator_at_vertex(0)

.. raw:: html

   <hr style="margin: 30px 0;">


HilbertSpaceGaugeGroupDimensionsMismatchError
-------------------------------------------------

**Description**

This error will be encountered if you try to have your ``lqx`` object have a Hilbert space with gauge
dimensions which differ from the gauge dimensions of the gauge group you have passed to it.

**Cause**

Trying to run something as the code below

.. code-block:: python

    # note the gauge_dimensions being 2
    H = nqx.hilbert.u1.HilbertU1(graph_4d_new, cutoff, gauge_dimensions = 2, is_gauge_invariant = True, auto_constraint = True)

    # note the gauge_dimensions being 3
    gauge_group = nqx.gauge_groups.u1.U1GaugeGroup(H, gauge_dimensions = 3, is_4d = True)

    # this will raise an error
    lqx = nqx.lqx.EuclideanWCL(H, gaugeGroup, spacetime_dims = 4)

will raise an ``HilbertSpaceGaugeGroupDimensionsMismatchError`` error.

**Fix**

To fix this, please make sure that your ``Hilbert`` object and your, for example, ``U1Gauge`` object have
the same gauge dimensions. In the example above, this means

.. code-block:: python

    # note the gauge_dimensions being 3
    H = nqx.hilbert.u1.HilbertU1(graph_4d_new, cutoff, gauge_dimensions = 3, is_gauge_invariant = True, auto_constraint = True)

    # note the gauge_dimensions being 3
    gauge_group = nqx.gauge_groups.u1.U1GaugeGroup(H, gauge_dimensions = 3, is_4d = True)

    # this will run
    lqx = nqx.lqx.EuclideanWCL(H, gaugeGroup, spacetime_dims = 4)

.. raw:: html

   <hr style="margin: 30px 0;">

MissingPenaltyFactorError
-----------------------------------

**Description**

neuraLQX allows to find eigenstates for constraints. It also allows you to try to regularise the
minimisation process a little bit. For this purpose, we implemented a ``PenaltyCost`` lazy wrapper
which wraps around any local operator. The total cost will now go from ``Cost = <C>``, where ``C`` is
your constraint, to ``Cost = <C> + factor * <P>``, where ``P`` is an operator which you have wrapped
and ``factor`` is a float which decides how much should the cost function be affected by this penalty.

If you have created a local operator which does not have a ``factor`` attribute, you will face this
error. To fix this, you can just do ``yourOperator.factor = 0.01`` or set the factor properly when
creating the wrapper.

**Cause**

Trying to run something like the below

.. code-block:: python

    # ...

    vol = lqx.model.volume_operator_at_vertex(0)

    P = PenaltyCost(vol)

    lqx.constraint += P

    # ...

    # this will crash
    solver.run(100)

will raise an ``MissingPenaltyFactorError`` error.

**Fix**

To fix this, set a correct factor when creating the wrapped operator.

.. code-block:: python

    # ...

    vol = lqx.model.volume_operator_at_vertex(0)

    P = PenaltyCost(vol, factor = 0.1)

    lqx.constraint += P

    # ...

    # this will work
    solver.run(100)

.. raw:: html

   <hr style="margin: 30px 0;">


InvalidChargeError
-----------------------------------

**Description**

You will encounter this error if you try to use charge (vector) coloring operators and have
specified a charge which is not in the list of allowed charges of your space. To fix this, ensure
that you try to measure the color of a charge which is in your Hilbert space.

**Cause**

Trying to run something like the following in a Hilbert space of cutoff 2

.. code-block:: python

    from neuralqx.operators import charge_coloring

    op = charge_coloring(H, site = 0, charge = 4)

will raise an ``InvalidChargeError`` error.

**Fix**

Set the ``charge`` parameter to something within the span of allowed charges in your model. In the
example above, this means

.. code-block:: python

    from neuralqx.operators import charge_coloring

    # this should work
    op = charge_coloring(H, site = 0, charge = 2)

.. raw:: html

   <hr style="margin: 30px 0;">

ExpectationValueError
-----------------------------------

**Description**

nueraLQX allows you to measure the expectation value of a single operator like this

.. code-block:: python

    # assume you have a Solver object called solver and a local operator called op
    solver.variational_state.expect(op)

You can also compute the expectation value of a sum of operators like this


.. code-block:: python

    # for two operators op1 and op2
    solver.variational_state.expect(op1 + op2)


which will mathematically do :math:`\langle op_1 + op_2\rangle`. However, this will computationally form a new
operator which is ``op1 + op2``, and then compute its expectation value. If you wish to compute
something like this :math:`\langle op_1\rangle + \langle op_2\rangle`, for whatever reason, then you can use

.. code-block::python
    solver.variational_state.expect([op1, op2])

Naturally, if you give the expectation value function just an empty list, you will see this error.


**Cause**

Trying to run something like this

.. code-block:: python

    solver.variational_state.expect([])

will raise an ``ExpectationValueError`` error.

**Fix**

Put some operators to compute the expectation value of!

Example:

.. code-block:: python

    solver.variational_state.expect([op1, op2, op3])

.. raw:: html

   <hr style="margin: 30px 0;">

InvalidOperatorsSequenceError
-----------------------------------

**Description**

neuralQX expectation values are computed for a single local operator
object or a list of them (see ``ExpectationValueError`` above). If you try to pass the sequence of
operators in another data structure (e.g. tuples, numpy arrays, etc.) you will face this error.

**Cause**

Trying to run something the following

.. code-block:: python

    solver.variational_state.expect((op1, op2))

will raise an ``InvalidOperatorsSequenceError`` error.

**Fix**

To fix this, always make sure that your operators are grouped together in a python ``list``.


Example:

.. code-block:: python

    # this should now work
    solver.variational_state.expect([op1, op2])

.. raw:: html

   <hr style="margin: 30px 0;">


InvalidOperatorsInSequenceError
-----------------------------------

**Description**

neuraLQX enables you to compute the expectation value of a sequence of operators grouped together and
passed to the expectation value function as a list. If you do so, then all operators in that list
**must** be wrapped by one of the following wrappers:

- the NetKet ``Squared`` wrapper,
- the neuraLQX ``PenaltyCost`` wrapper
- the neuraLQX ``InverseVolumeCost`` wrapper

Mixing between the both of them in the same list is okay. This is because these operators may have
their gradients computed using different functions than standard operators.

**Cause**

Trying to run the following

.. code-block:: python

    from netket.operators import Squared

    solver.variational_state.expect([Squared(local_operator_1), local_operator_2])

will raise an ``InvalidOperatorsInSequenceError`` error.

**Fix**

To fix this, use only the allowed types when using the ``.expect()`` function with a list as an
argument


Example:

.. code-block:: python

    from netket.operators import Squared

    solver.variational_state.expect([Squared(local_operator_1), Squared(local_operator_2)])


.. note::
    Of course, this might not be what you always want... We are currently working on making this
    feature generally applicable to all types of operators.

.. raw:: html

   <hr style="margin: 30px 0;">

InvalidMelsFunctionError
-----------------------------------

**Description**

When using the ``FunctionalLocalOperator``, you have the ability to decide which function to use to
apply to the matrix elements once computed. This function is specified via the ``mels_func``
parameter. While you can choose any function you want, it has to be written in terms of a ``lambda``
expression. The reason for this is that we need to Numba just-in-time compile it first as it will
be called in a Numba just-in-time routine.

**Cause**

Let's say you want to use numpy's square root as the ``mels_func``. Instantiating a ``FunctionalLocalOperator``
as follows

.. code-block:: python

    FunctionalLocalOperator(
        # ...
        mels_func = numpy.sqrt
    )


will raise an ``InvalidMelsFunctionError`` error.

**Fix**

Instead of instantiating the operator as shown above, do the following

.. code-block:: python

    FunctionalLocalOperator(
        # ...
        mels_func = lambda x: numpy.sqrt(x)
    )

.. warning::
    ``FunctionalLocalOperator`` will apply this function at the matrix elements after they have been
    computed. **This is conceptually correct only for diagonal operators!** Be careful of using this
    with operators not diagonal in the computational basis.

.. raw:: html

   <hr style="margin: 30px 0;">


IncompatibleJaxOperatorError
-----------------------------------

**Description**

In the supported NetKet version of neuraLQX, you can create ``LocalOperator`` objects which are by
default Numba friendly (for future versions this will not be true).
However, every ``LocalOperator`` object has a ``.to_jax_operator()`` function which returns a
``LocalOperatorJax`` object: an identical implementation but it is Jax friendly.
In our case, this has been disabled and marked as an experimental feature for the
``MarkedLocalOperator`` and ``FunctionalLocalOperator`` classes as they handles the matrix elements
computations in a very specific manner and the NetKet Jax logic is yet to be modified. You will receive
this error when trying to use the ``.to_jax_operator()`` function in the mentioned classes.

**Cause**

Doing the following

.. code-block:: python

    my_marked_local_operator_jax = my_marked_local_operator.to_jax_operator()


will raise an ``IncompatibleJaxOperatorError`` error.

**Fix**

To fix this, you can set the ``NQX_EXPERIMENTAL`` configuration option to ``'1'`` before importing
neuraLQX

.. code-block:: python

    import os
    os.environ['NQX_EXPERIMENTAL'] = '1'

    import neuralqx as nqx

    # ...

    my_marked_local_operator_jax = my_marked_local_operator.to_jax_operator()

and then the ``.to_jax_operator()`` call will work.

.. raw:: html

   <hr style="margin: 30px 0;">


OutOfRangeIndexError
-----------------------------------

**Description**

When trying to use the ``.index_to_edge()`` method in the ``AbstractGraph`` or any subclass of it, you
will receive this error if you are requesting an index which does not exist (e.g. you are trying to
ask for an edge which does not exist, since the indexing is a bijective map between the edges and
integers).

**Cause**

Doing the following

.. code-block:: python

    edge = graph.index_to_edge(50)


in a graph of only 10 edges will raise an ``OutOfRangeIndexError`` error, as there is not 50 edges
which have indices to represent them.

**Fix**

To fix this, only request indices which represent edges which are actually there in the graph. In
the above example, this means doing the following

.. code-block:: python

    edge = graph.index_to_edge(10)

assuming once again, that the graph has 10 edges.

.. note::
    You can view the mapping between the edges and the indices using ``.mapping`` on any
    ``AbstractGraph`` object.

.. raw:: html

   <hr style="margin: 30px 0;">


InvalidIndexError
-----------------------------------

**Description**

This is like the ``OutOfRangeIndexError`` error above, except the other way around. Namely, if you
try to use the ``.edge_to_index(your_edge)`` for an edge that does not exist in the graph, you will
see this error.

**Cause**

The following

.. code-block:: python

    import neuralqx as nqx

    graph = nqx.graph.Graph([[0, 1], [0, 2], [0, 3], [1, 2], [3, 2]])

    # edge [5, 2] is not in the graph
    edge_idx = graph.edge_to_index([5, 2])

will raise an ``InvalidIndexError`` error.

**Fix**

To fix this, only request indices of edges which are *actually* in the graph

.. code-block:: python

    import neuralqx as nqx

    graph = nqx.graph.Graph([[0, 1], [0, 2], [0, 3], [1, 2], [3, 2]])

    # the following will work
    edge_idx = graph.edge_to_index([3, 2])


.. raw:: html

   <hr style="margin: 30px 0;">


NonExistentNonPlanarEdgesError
-----------------------------------

**Description**

Under the ``AbstractGraph`` implementation, you can request the edges of the graph by using ``.edges``.
Similarly, you can request them in the non-planar representation, where vertices are labeled by ``(x, y, z)``
instead of integers, by using ``.nonplanar_edges``. However, this only works if you have created a non-planar
graph to begin with. If you did not and then attempt to request the non-planar edges, you will receive
this error.

**Cause**

The following

.. code-block:: python

    import neuralqx as nqx

    graph = nqx.graph.Graph([[0, 1], [0, 2], [0, 3], [1, 2], [3, 2]])

    my_edges = graph.nonplanar_edges

will raise an ``NonExistentNonPlanarEdgesError`` error, as the edges specified to create the graph
are planar.

**Fix**

To fix this, create the graph as non-planar by feeding it the vertices in the non-planar representation

.. code-block:: python

    import neuralqx as nqx

    # same graph, but non-planar
    graph = nqx.graph.Graph(
        [
            ((0, 0, 0), (0, 0, 1)),
            ((0, 0, 0), (0, 1, 0)),
            ((0, 0, 0), (1, 0, 0)),
            ((0, 0, 1), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0))
        ]
    )

    # this should now work
    my_edges = graph.nonplanar_edges


.. raw:: html

   <hr style="margin: 30px 0;">


OrientationValenceMismatchError
-----------------------------------

**Description**

When creating an instance of the ``SingleVertexGraph``, you have the option to specify the orientation
of each of the edges attached to that single vertex. If you specify a list of orientations which do not
match the number of requested edges attached to the vertex, you will see this error.

**Cause**

The following

.. code-block:: python

    import neuralqx as nqx

    graph = nqx.graph.SingleVertexGraph(valence = 2, orientation = [1, -1, -1])

will raise an ``OrientationValenceMismatchError`` error, as we have requested a single-vertex graph
with valence 2, but supplied orientation for 3 edges.

**Fix**

To fix this, match the number of edges and orientation

.. code-block:: python

    import neuralqx as nqx

    graph = nqx.graph.SingleVertexGraph(valence = 2, orientation = [1, -1])

This should now work, as we have specified the orientation for the only available 2 edges at the vertex.

.. raw:: html

   <hr style="margin: 30px 0;">

NonExistentNonPlanarVerticesError
-----------------------------------

**Description**

Under the ``AbstractGraph`` implementation, or any subclass of it, you can request the mapping of non-planar
vertices to an indexing where they are labeled by integers using ``.nonplanar_vertex_mapping``. However,
this only works if your graph is non-planar and you will receive this error if your graph is not. Further,
you can also map non-planar vertices to their integer labeling using ``.nonplanar_vertex_to_index()``.
However, this also only works in the case of non-planar graphs and the same error will be received if not.

**Cause**

The following

.. code-block:: python

    import neuralqx as nqx

    graph = nqx.graph.Graph([[0, 1], [0, 2], [0, 3], [1, 2], [3, 2]])

    # this will crash
    my_edges = graph.nonplanar_vertex_to_index([0, 3])

    # this will also crash
    nonplanar_vertex_mapping = graph.nonplanar_vertex_mapping

will raise an ``NonExistentNonPlanarVerticesError`` error. This is because our graph is planar, but
we are requesting properties from it as if it were non-planar.

**Fix**

To fix this, use non-planar graphs.

.. code-block:: python

    import neuralqx as nqx

    # same graph, but non-planar
    graph = nqx.graph.Graph(
        [
            ((0, 0, 0), (0, 0, 1)),
            ((0, 0, 0), (0, 1, 0)),
            ((0, 0, 0), (1, 0, 0)),
            ((0, 0, 1), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0))
        ]
    )

    # this will work now
    my_edges = graph.nonplanar_vertex_to_index([0, 3])

    # this will also work now
    nonplanar_vertex_mapping = graph.nonplanar_vertex_mapping


.. raw:: html

   <hr style="margin: 30px 0;">


DistributedRuntimeUnavailableWarning
-----------------------------------

**Description**

This warning indicates that distributed execution was requested but neuraLQX could not initialise
the distributed runtime, so execution falls back to serial mode.

**Cause**

The following

.. code-block:: python

    import os
    os.environ['NQX_JAX_DISTRIBUTED'] = '1'

    import neuralqx as nqx

can trigger this warning when the distributed launch/runtime environment is not set up correctly.

**Fix**

Use a proper distributed launch setup and ensure JAX distributed initialisation works on your system.
If you intend to run locally, disable distributed mode and run in serial.

.. raw:: html

   <hr style="margin: 30px 0;">


DistributedStateImportInSerialModeError
-----------------------------------

**Description**

If you export a state from a distributed run, loading that state in a serial run is currently not allowed.

Simply put, distributed-exported states cannot be loaded in serial mode (for now).

.. note::
    We are aware this can be restrictive when moving checkpoints between HPC and local environments.
    **This is currently something we are
    actively fixing and should be cleared in a later release.**

.. raw:: html

   <hr style="margin: 30px 0;">


DistributedStateImportMismatchError
-----------------------------------

**Description**

Currently, if you run a simulation in distributed mode and export a state, you can only load it with
the **same** distributed layout (for example, matching total processes and processes per host).

Simply put, distributed-exported states can only be loaded in simulations with identical runtime setup as the
one they were exported from.

.. note::
    This is currently something we are actively fixing and should be cleared in a later release.

.. raw:: html

   <hr style="margin: 30px 0;">



NonHermitianInverseCostError
-----------------------------------

**Description**

neuraLQX provides you with the ``InverseVolumeCost`` lazy wrapper for any NetKet `AbstractOperator` (or
any implementation which subclasses it). However, this lazy wrapper only works with **Hermitian operators.**
This is because it uses a covariance based method to compute its gradient and such a method can only work
for Hermitian operators.

**Cause**

The following code for an operator ``O`` which is not Hermitian

.. code-block:: python

    # ..
    from neuralqx.operators import InverseVolumeCost

    my_wrapped_operator = InverseVolumeCost(O, alpha = 0.2, factor = 1.2)

will raise a ``NonHermitianInverseCostError`` error.


**Fix**

To resolve this, only wrap operators which are Hermitian (e.g. ``O.is_hermitian = True``) with this
wrapper.

.. warning::
    In principle, it is easy to force neuraLQX to accept non-Hermitian operators for this wrapper.
    However, your gradients will be incorrect, so it does not make sense to try to bypass this.

.. raw:: html

   <hr style="margin: 30px 0;">


InvalidSurfaceError
-----------------------------------

**Description**

When using the area operator in the 4d weak coupling model, the surface to compute the area for must
be provided as a python ``List`` (or ``Tuple``) of edges. To resolve this error, please make sure you pass a
``List`` of edges (e.g. not an array of edges).


**Cause**

The following code

.. code-block:: python

    A = lqx.model.area_operator(np.array([3, 2], [2, 4]))

will raise a ``InvalidSurfaceError`` error.


**Fix**

To resolve this, pass the edges in a list or a tuple

.. code-block:: python

    A = lqx.model.area_operator(([3, 2], [2, 4]))

.. raw:: html

   <hr style="margin: 30px 0;">


AreaDifferenceEdgesError
-----------------------------------

**Description**

To use the area difference operator in the 4d weak coupling model, you need to provide 2 **(and only 2)**
edges. If you provide more (or less), you will see this error.


**Cause**

The following code

.. code-block:: python

    Adiff = lqx.model.area_difference_operator(([3, 2], [2, 4], [1, 5]))

will raise a ``AreaDifferenceEdgesError`` error.


**Fix**

To resolve this, provide only two edges

.. code-block:: python

    Adiff = lqx.model.area_difference_operator(([3, 2], [2, 4]))

.. raw:: html

   <hr style="margin: 30px 0;">

AreaDifferenceSurfacesError
-----------------------------------

**Description**

To use the area difference operator between **two surfaces** in the 4d weak coupling model, you need
to provide 2 **(and only 2)** surfaces. If you provide more (or less), you will see this error.


**Cause**

The following code

.. code-block:: python

    Adiff = lqx.model.area_difference_surfaces_operator([[[3, 2], [1, 2]], [[0, 2], [4, 5]], [0, 1], [1, 5]])

will raise a ``AreaDifferenceSurfacesError`` error.


**Fix**

To resolve this, provide only two surfaces

.. code-block:: python

    Adiff = lqx.model.area_difference_surfaces_operator([[[3, 2], [1, 2]], [[0, 2], [4, 5]]])

.. raw:: html

   <hr style="margin: 30px 0;">


IncompatibleNonGIOperatorError
-----------------------------------

**Description**

Sometimes, some operators are not compatible with gauge invariant Hilbert spaces. If you see this error,
it means that you have chosen such an operator. To use it, you must use the full Hilbert space or
implement your own custom operator.



AutoConstraintGaugeFixingConflictError
---------------------------------------

**Description**

When creating gauge invariant Hilbert spaces with Abelian degrees of freedom, neuraLQX has the option for you to allow it
to figure out the gauge fixing conditions. If you specify that using ``auto_constraint=True`` and simultaneously provide
an explicit gauge fixing list, you will see this error. Choose one or the other, but not both.



IncompatibleNonPlanarGraphModel
---------------------------------------

**Description**

Some models in neuraLQX are only compatible with planar graphs. If you supplied a non-planar graph to such a model, you
will see this error. To resolve this, use ``non_planar=False`` when instantiating your graph object if it is a pre-defined
neuraLQX graph class.



IncompatiblePlanarGraphModel
---------------------------------------

**Description**

Some models in neuraLQX are only compatible with non-planar graphs. If you supplied a planar graph to such a model, you
will see this error. To resolve this, use ``non_planar=True`` when instantiating your graph object if it is a pre-defined
neuraLQX graph class.



IncompatibleHilbertSpaceError
---------------------------------------

**Description**

Some models in neuraLQX are particular about the choice of Hilbert spaces. If you supply Hilbert spaces of incorrect type
(i.e. too many gauge dimensions in Hilbert spaces with Abelian degrees of freedom, you will see this error).




IncompatibleModdedOperatorWarning
---------------------------------------

**Description**

The Gauß constraint in Abelian models is only available with modded arithmetic in the computational backend. If you
have requested the gauge group with a ``LocalOperator`` backend and simultaneously requested the ``modded=True``, you
will see this error. To resolve this, either use ``modded=False`` or switch to the computational backend.



RandomEmbeddingForPlanarGraphWarning
---------------------------------------

**Description**

Random embedding is only compatible with non-planar graphs. If you have requested a planar graph with ``random_embedding=True``,
you will see this warning. This means that the request for random embedding has been ignored. Use non-planar graphs for
random embeddings if your physical model supports it.


LiveMonitoringUnavailableWarning
---------------------------------------

**Description**
By default, neuraLQX disables live monitoring in distributed computations, whether it is on CPUs or GPUs. If you see
this warning, then you have requested live monitoring while running in distributed mode.


ComputationalModelConcretizationWarning
------------------------------------------

**Description**
Some pre-implemented models in neuraLQX are strictly only computational (or support only computational backends for the
main constraints). If you see this warning, it means you have requested a local operator backend in a model which does
not support it.


SuboptimalOperatorForGPUWarning
------------------------------------------

**Description**
neuraLQX provides both ``ComputationalOperator`` and ``ComputationalJaxOperator`` as computational backends for operators.
When computing on GPUs, the ``ComputationalOperator`` type is suboptimal in performance and we recommend using the JAX
variant instead.

If you are using one of neuraLQX's implemented operators, you can find the JAX variant in the ``neuralqx.operators.computational``
module

.. code-block:: python

   # ComputationalOperator
   from neuralqx.operators.computational.misc.numba import U1Holonomy

   # ComputationalJaxOperator
   from neuralqx.operators.computational.misc.jax import U1HolonomyJax

   # alternatively
   from neuralqx.operators.computational.misc import U1HolonomyOperator
   op = U1HolonomyOperator(..., jax=True)

If this is a custom operator, consider implementing it as a subclass of ``ComputationalJaxOperator`` for better
performance on GPUs.
