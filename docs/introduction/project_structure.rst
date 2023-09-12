=================
Project Structure
=================

.. highlight:: none

::

  src/gtl              generic library code
  scripts/             _executable_ scripts for generating specific experiments, datasets, etc.
  data/                input, processed, and generated datasets
  results/          
  docs/
  Makefile          


* I prefer a "unix" approach to data science - analysis is provided in small
  scripts in the ``scripts`` directory, which can be ran in sequence using the
  ``Makefile``.

* The code in ``src/gtl`` is importable as a library within the project

* For reproducability, ``data`` and ``results`` is (mostly) git-ignored. Data can
  be downloaded or generated using the ``Makefile``


----


**For details on loading** ``src/gtl`` **, running code, and the development environment, see** :doc:`installation` **.**


These practices are based on `Good Research <https://goodresearch.dev/>`_ and
`Cookiecutter data science
<https://drivendata.github.io/cookiecutter-data-science/>`_.
