:orphan:

.. _multistate_vmc:

===========================================
Multi-state variational Monte Carlo
===========================================

The MT-MH multi-state VMC solver is no longer experimental. It has been promoted to the
public API and is documented in the stable Solver guide under :ref:`multistate_solver`.

Use the public imports:

.. code-block:: python

   from neuralqx.solver import MultiSolver
   from neuralqx.driver import MultiStateVMC
   from neuralqx.vqs import MultiMCState

The old experimental MT-MH import paths are intentionally unavailable and should be
treated as removed.

The experimental multi-state page is kept only as a migration pointer so older links remain
understandable. New code and documentation should use :class:`neuralqx.solver.MultiSolver`,
:class:`neuralqx.driver.MultiStateVMC`, and :class:`neuralqx.vqs.MultiMCState`.
