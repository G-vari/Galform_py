## `Galform_py`: a semi-analytic galaxy formation model written in Python 
`Galform_py` is an offshoot of the long established Durham semi-analytic galaxy formation model, `Galform`.
Where as `Galform` is written in Fortran, `Galform_py` is written from scratch in Python.
The (significant) drawback compared to the original `Galform` Fortran code is that `Galform_py` is slow, rendering it unsuitable for use on cosmologically representative samples of haloes.
The upside is that `Galform_py` is written in Python which, combined with a more modular structure, makes the code much easier to devlop.
As such, the objective of `galform_py` is to provide a test bench for quickly trying new scientific ideas.

To get started, do BLAH.

### Prerequisites

You need a python installation with the core scientific packages: `numpy`, `matplotlib` and `scipy`.
Try

     easy_install numpy matplotlib scipy

or

     pip install numpy matplotlib scipy

to install these dependences if you miss them.

## Authors
Peter Mitchell [ICC, Durham / CRAL, Lyon]