.. _cite:

=======================
Citing neuraLQX
=======================

neuraLQX would not exist without NetKet. If you got results with neuraLQX, then NetKet is part of the
scientific and engineering chain that produced those results: sampling, VMC drivers, SR/QGT machinery,
operator interfaces, logging, distributed computing integration patterns, and the surrounding ecosystem.

**You have to always cite NetKet whenever you cite neuraLQX!**

This page gives ready-to-use citation text and BibTeX snippets for common usage scenarios.


.. important::

   If you publish work that used neuraLQX, **you have to cite neuraLQX and you absolutely have to cite NetKet at minimum**.
   If you used MPI-distributed runs with older neuraLQX versions, also cite **mpi4jax**.
   You should additionally cite **JAX** (and **Flax**) as it is the main workhorse behind everything.


-------------------------
What should I cite?
-------------------------

Use one of the bundles below as a default. Pick the one that matches how you ran neuraLQX.

**Bundle A: neuraLQX + NetKet (a must)**
  Cite neuraLQX and both NetKet papers. This is the standard citation for essentially all neuraLQX-based work.

**Bundle B: neuraLQX + NetKet + MPI**
  If you ran across multiple CPUs/nodes using MPI-backed workflows, add mpi4jax.

**Bundle C: neuraLQX + NetKet + JAX/Flax**
  *Ideally*, you cite NetKet, JAX as well as Flax.

**Bundle D: Full stack**
  If you used MPI, *ideally*, you should cite NetKet, JAX, Flax and mpi4jax.


------------------------------------------
Suggested citation sentences
------------------------------------------

Below are suggested sentences you can paste into your manuscript. Use one *as-is* or adjust to your style.
All variants **include NetKet** by construction.

LaTeX examples
^^^^^^^^^^^^^^

Minimal (recommended default)
  .. code-block:: latex

     Simulations for this work were performed using neuraLQX~\cite{neuralqx1:2026}. neuraLQX is a high-performance
     simulations toolkit for loop quantum gravity built on top of NetKet~\cite{netket3:2022,netket2:2019}.

If you used MPI / multi-node runs
  .. code-block:: latex

     Simulations for this work were performed using neuraLQX~\cite{neuralqx1:2026}. neuraLQX is a high-performance
     simulations toolkit for loop quantum gravity built on top of NetKet~\cite{netket3:2022,netket2:2019}.
     Distributed execution used MPI-enabled JAX communication via mpi4jax~\cite{mpi4jax:2021}.

The ideal non-MPI case:
  .. code-block:: latex

     Simulations for this work were performed using neuraLQX~\cite{neuralqx1:2026}. neuraLQX is a high-performance
     simulations toolkit for loop quantum gravity built on top of NetKet~\cite{netket3:2022,netket2:2019}.
     The implementation relies on JAX~\cite{jax2018github} and Flax~\cite{flax2020github}.

Full-stack acknowledgement:
  .. code-block:: latex

     Simulations for this work were performed using neuraLQX~\cite{neuralqx1:2026}. neuraLQX is a high-performance
     simulations toolkit for loop quantum gravity built on top of NetKet~\cite{netket3:2022,netket2:2019}.
     The implementation relies on JAX~\cite{jax2018github} and Flax~\cite{flax2020github}. MPI-distributed execution
     used mpi4jax~\cite{mpi4jax:2021}.

"Built with / derived from" (when neuraLQX inspired internal code)
  .. code-block:: latex

     The software used in this work was developed with neuraLQX~\cite{neuralqx1:2026}
     as an implementation reference and was built on concepts and components from
     NetKet~\cite{netket3:2022,netket2:2019}.


-------------------------
BibTeX
-------------------------

Bundle A: neuraLQX + NetKet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Click to reveal BibTeX for **neuraLQX + NetKet**

   .. code-block:: bibtex

      @software{neuralqx1:2026,
        title = {{neuraLQX}: a high-performance simulations toolkit for loop quantum gravity},
        author = {Sherif, Waleed},
        url = {http://github.com/waleed-sh/neuraLQX},
        version = {1.1.2},
        year = {2026},
      }

      @article{netket2:2019,
          title={NetKet: A machine learning toolkit for many-body quantum systems},
          author={Carleo, Giuseppe and Choo, Kenny and Hofmann, Damian and Smith, James ET and Westerhout, Tom and Alet, Fabien and Davis, Emily J and Efthymiou, Stavros and Glasser, Ivan and Lin, Sheng-Hsuan and Mauri, Marta and Mazzola, Guglielmo and Pereira, Christian B and Vicentini, Filippo},
          journal={SoftwareX},
          volume={10},
          pages={100311},
          year={2019},
          publisher={Elsevier},
          doi={10.1016/j.softx.2019.100311}
      }

      @article{netket3:2022,
          title={NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems},
          author={Vicentini, Filippo and Hofmann, Damian and Szabó, Attila and Wu, Dian and Roth, Christopher and Giuliani, Clemens and Pescia, Gabriel and Nys, Jannes and Vargas-Calderón, Vladimir and Astrakhantsev, Nikita and Carleo, Giuseppe},
          journal={SciPost Phys. Codebases},
          pages={7},
          year={2022},
          doi={10.21468/SciPostPhysCodeb.7}
      }


Bundle B: neuraLQX + NetKet + MPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Click to reveal BibTeX for **neuraLQX + NetKet + MPI**

   .. code-block:: bibtex

      @software{neuralqx1:2026,
        title = {{neuraLQX}: a high-performance simulations toolkit for loop quantum gravity},
        author = {Sherif, Waleed},
        url = {http://github.com/waleed-sh/neuraLQX},
        version = {1.1.2},
        year = {2026},
      }

      @article{netket2:2019,
          title={NetKet: A machine learning toolkit for many-body quantum systems},
          author={Carleo, Giuseppe and Choo, Kenny and Hofmann, Damian and Smith, James ET and Westerhout, Tom and Alet, Fabien and Davis, Emily J and Efthymiou, Stavros and Glasser, Ivan and Lin, Sheng-Hsuan and Mauri, Marta and Mazzola, Guglielmo and Pereira, Christian B and Vicentini, Filippo},
          journal={SoftwareX},
          volume={10},
          pages={100311},
          year={2019},
          publisher={Elsevier},
          doi={10.1016/j.softx.2019.100311}
      }

      @article{netket3:2022,
          title={NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems},
          author={Vicentini, Filippo and Hofmann, Damian and Szabó, Attila and Wu, Dian and Roth, Christopher and Giuliani, Clemens and Pescia, Gabriel and Nys, Jannes and Vargas-Calderón, Vladimir and Astrakhantsev, Nikita and Carleo, Giuseppe},
          journal={SciPost Phys. Codebases},
          pages={7},
          year={2022},
          doi={10.21468/SciPostPhysCodeb.7}
      }

      @article{mpi4jax:2021,
          title={mpi4jax: Zero-copy MPI communication of JAX arrays},
          author={Häfner, Dion and Vicentini, Filippo},
          journal={Journal of Open Source Software},
          volume={6},
          number={65},
          pages={3419},
          year={2021},
          doi={10.21105/joss.03419}
      }


Bundle C: neuraLQX + NetKet + JAX/Flax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Click to reveal BibTeX for **neuraLQX + NetKet + JAX + Flax**

   .. code-block:: bibtex

      @software{neuralqx1:2026,
        title = {{neuraLQX}: a high-performance simulations toolkit for loop quantum gravity},
        author = {Sherif, Waleed},
        url = {http://github.com/waleed-sh/neuraLQX},
        version = {1.1.2},
        year = {2026},
      }

      @article{netket2:2019,
          title={NetKet: A machine learning toolkit for many-body quantum systems},
          author={Carleo, Giuseppe and Choo, Kenny and Hofmann, Damian and Smith, James ET and Westerhout, Tom and Alet, Fabien and Davis, Emily J and Efthymiou, Stavros and Glasser, Ivan and Lin, Sheng-Hsuan and Mauri, Marta and Mazzola, Guglielmo and Pereira, Christian B and Vicentini, Filippo},
          journal={SoftwareX},
          volume={10},
          pages={100311},
          year={2019},
          publisher={Elsevier},
          doi={10.1016/j.softx.2019.100311}
      }

      @article{netket3:2022,
          title={NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems},
          author={Vicentini, Filippo and Hofmann, Damian and Szabó, Attila and Wu, Dian and Roth, Christopher and Giuliani, Clemens and Pescia, Gabriel and Nys, Jannes and Vargas-Calderón, Vladimir and Astrakhantsev, Nikita and Carleo, Giuseppe},
          journal={SciPost Phys. Codebases},
          pages={7},
          year={2022},
          doi={10.21468/SciPostPhysCodeb.7}
      }

      @misc{jax2018github,
        title = {{{JAX}}: Composable Transformations of {{Python}}+{{NumPy}} Programs},
        author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and Wanderman-Milne, Skye and Zhang, Qiao},
        year = {2018}
      }

      @misc{flax2020github,
        title = {Flax: A Neural Network Library and Ecosystem for JAX},
        author = {Heek, Jonathan and Levskaya, Anselm and Oliver, Avital and Ritter, Marvin and Rondepierre, Bertrand and Steiner, Andreas and van Zee, Marc},
        year = {2024}
      }


Bundle D: Full stack (everything on this page)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Click to reveal BibTeX for **Full stack (neuraLQX + NetKet + mpi4jax + JAX + Flax)**

   .. code-block:: bibtex

      @software{neuralqx1:2026,
        title = {{neuraLQX}: a high-performance simulations toolkit for loop quantum gravity},
        author = {Sherif, Waleed},
        url = {http://github.com/waleed-sh/neuraLQX},
        version = {1.1.2},
        year = {2026},
      }

      @article{netket2:2019,
          title={NetKet: A machine learning toolkit for many-body quantum systems},
          author={Carleo, Giuseppe and Choo, Kenny and Hofmann, Damian and Smith, James ET and Westerhout, Tom and Alet, Fabien and Davis, Emily J and Efthymiou, Stavros and Glasser, Ivan and Lin, Sheng-Hsuan and Mauri, Marta and Mazzola, Guglielmo and Pereira, Christian B and Vicentini, Filippo},
          journal={SoftwareX},
          volume={10},
          pages={100311},
          year={2019},
          publisher={Elsevier},
          doi={10.1016/j.softx.2019.100311}
      }

      @article{netket3:2022,
          title={NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems},
          author={Vicentini, Filippo and Hofmann, Damian and Szabó, Attila and Wu, Dian and Roth, Christopher and Giuliani, Clemens and Pescia, Gabriel and Nys, Jannes and Vargas-Calderón, Vladimir and Astrakhantsev, Nikita and Carleo, Giuseppe},
          journal={SciPost Phys. Codebases},
          pages={7},
          year={2022},
          doi={10.21468/SciPostPhysCodeb.7}
      }

      @article{mpi4jax:2021,
          title={mpi4jax: Zero-copy MPI communication of JAX arrays},
          author={Häfner, Dion and Vicentini, Filippo},
          journal={Journal of Open Source Software},
          volume={6},
          number={65},
          pages={3419},
          year={2021},
          doi={10.21105/joss.03419}
      }

      @misc{jax2018github,
        title = {{{JAX}}: Composable Transformations of {{Python}}+{{NumPy}} Programs},
        author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and Wanderman-Milne, Skye and Zhang, Qiao},
        year = {2018}
      }

      @misc{flax2020github,
        title = {Flax: A Neural Network Library and Ecosystem for JAX},
        author = {Heek, Jonathan and Levskaya, Anselm and Oliver, Avital and Ritter, Marvin and Rondepierre, Bertrand and Steiner, Andreas and van Zee, Marc},
        year = {2024}
      }

.. hint::

    You can also see the BibTeX above in neuraLQX by doing the following

    .. code-block:: python

        import neuralqx as nqx
        nqx.cite.all  # show BibTeX for neuraLQX, NetKet, Jax, Flax, mpi4jax
        nqx.cite.neuralqx.netket  # show BibTeX for neuraLQX and NetKet only

        # you can also compose them
        nqx.cite.neuralqx.netket.mpi
        # or
        nqx.cite.neuralqx.jax.netket

-------------------------
FAQ
-------------------------

**Do I really need both NetKet citations?**
  Yes.

**What if I used neuraLQX for prototyping but rewrote the final code?**
  Cite neuraLQX as an implementation reference (see the "Built with / derived from" sentence above)
  **and still cite NetKet!**

**What if my journal has a strict citation limit?**
  Use Bundle A at minimum, you have to cite neuraLQX, and you also must cite NetKet. If MPI was essential for feasibility, prefer Bundle B
  over adding JAX/Flax.

