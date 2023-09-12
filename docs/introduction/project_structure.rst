=================
Project Structure
=================

.. highlight:: none

::

  src/gtl     library code
  scripts/    executable scripts for specific tasks
  data/       input, processed, and generated datasets
  results/   
  docs/
  Makefile   


* I prefer a "Unix" approach to data science - specific tasks (such as data
  preparation, visualisation, experiments, etc. ) are provided in small scripts
  in the ``scripts`` directory. These can then be chained together using
  ``make``.

* The code in ``src/gtl`` is importable as a library within the project as ``gtl``.

* For reproducibility, the contents of ``data`` and ``results`` are (mostly) git-ignored. Data can
  be downloaded or generated using the ``Makefile``.

----




* These practices are based on `The Good Research Code Handbook
  <https://goodresearch.dev/>`_ and `Cookiecutter data science
  <https://drivendata.github.io/cookiecutter-data-science/>`_.
