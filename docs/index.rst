:layout: landing
:description: neuraLQX – High-performance variational simulations for Loop Quantum Gravity

.. raw:: html

   <style>
     .btn-primary {
       background: #14806e;
       color: white;
       transition: background 0.2s ease;
     }

     .btn-primary:hover {
       background: #1aa08a;
     }

     .btn-secondary {
       color: inherit;
       border: 1px solid rgba(0,0,0,0.25);
       transition: color 0.2s ease, border-color 0.2s ease;
     }

     .btn-secondary:hover {
       color: #14806e;
       border-color: #14806e;
     }
   </style>


.. raw:: html

   <div style="max-width:900px; margin:0 auto;">

     <img src="_static/_logo.png"
          alt="neuraLQX logo"
          style="width:420px; margin-bottom:18px;margin-top:60px;" />

     <div style="font-size:18px; opacity:0.85; margin-bottom:26px;">
       High-performance simulation toolkit, built on NetKet &amp; JAX,
       tailored for canonical loop quantum gravity.
     </div>


.. raw:: html

   <div style="display:flex; gap:16px; margin-bottom:82px;">
     <a href="start.html" style="text-decoration:none; border-bottom:none;">
       <span class="btn-primary"
             style="
               padding:14px 24px;
               border-radius:999px;
               font-weight:900;
               font-size:16px;
               display:inline-block;
             ">
         Get Started
       </span>
     </a>

     <a href="documentation/api/index.html" style="text-decoration:none; border-bottom:none;">
       <span class="btn-secondary"
             style="
               padding:14px 24px;
               border-radius:999px;
               font-size:16px;
               display:inline-block;
             ">
         API Docs
       </span>
     </a>
   </div>

     <div style="font-size:15px; line-height:1.6; opacity:0.5;">
       neuraLQX is an open-source Python package for variational canonical loop
       quantum gravity. It lets you work directly with graphs, Hilbert spaces,
       gauge groups, and constraints, while leveraging state of the art Monte Carlo methods,
       automatic differentiation, and scalable optimisation.
     </div>

   </div>


.. toctree::
   :caption: Getting Started
   :maxdepth: 3
   :hidden:

   philosophy
   roadmap
   installation
   config_options
   citing
   quickstart.ipynb


.. toctree::
   :caption: Contents
   :maxdepth: 3
   :hidden:

   documentation/api/index
   Tutorials/index
   guides/index
   getting_started/parallelisation/index


.. toctree::
   :caption: Developers
   :maxdepth: 3
   :hidden:

   contribute
   change_log
   api_versioning
   experimental/index