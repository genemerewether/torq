# Scipy prerequisites
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran

# cvxopt prerequisites
sudo apt-get install libsuitesparse-dev

# Numpy and Scipy installation dependency/order

If `setup_requires=['numpy']` in `setup.py` is not enough to get Numpy to finish
installing before scipy is installed, then see [http://stackoverflow.com/questions/21605927/why-doesnt-setup-requires-work-properly-for-numpy/21621493#21621493].

This appears to be fixed as of 3/30/2017 but if not, then either install with
pip or make setup.py look like this:

```
#!/usr/bin/env python

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

#
# This cludge is necessary for horrible reasons: see comment below and
# http://stackoverflow.com/q/19919905/447288
#
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    #
    # Amazingly, `pip install scipy` fails if `numpy` is not already installed.
    # Since we cannot control the order that dependencies are installed via
    # `install_requires`, use `setup_requires` to ensure that `numpy` is available
    # before `scipy` is installed.
    #
    # Unfortunately, this is *still* not sufficient: `numpy` has a guard to
    # check when it is in its setup process that we must circumvent with
    # the `cmdclass`.
    #
    setup_requires=['numpy'],
    cmdclass={'build_ext':build_ext},
    install_requires=[
        'numpy',
        'scipy',
    ],
    ...
)
```

# MATLAB vs Octave vs Numpy

## From MathWorks documentation on `mldivide`, `\`

> `x = A\B` solves the system of linear equations `A*x = B`. The matrices A and B must have the same number of rows. MATLAB® displays a warning message if A is badly scaled or nearly singular, but performs the calculation regardless.
> If A is a scalar, then `A\B` is equivalent to `A.\B`.
> If A is a square n-by-n matrix and B is a matrix with n rows, then `x = A\B` is a solution to the equation `A*x = B`, if it exists.
> If A is a rectangular m-by-n matrix with m ~= n, and B is a matrix with m rows, then `A\B` returns a least-squares solution to the system of equations `A*x= B`.

## Octave documentation on Matlab method

From [http://wiki.octave.org/FAQ#Solvers_for_singular.2C_under-_and_over-determined_matrices]:

> Matlab's solvers as used by the operators `mldivide`. `\` and `mrdivide`, `/`, use a different approach than Octave's in the case of singular, under-, or over-determined matrices. In the case of a singular matrix, Matlab returns the result given by the LU decomposition, even though the underlying solver has flagged the result as erroneous. Octave has made the choice of falling back to a minimum norm solution of matrices that have been flagged as singular which arguably is a better result for these cases.

> In the case of under- or over-determined matrices, Octave continues to use a minimum norm solution, whereas Matlab uses an approach that is equivalent to:
```
function x = mldivide (A, b)
  m = rows(A);
  [Q, R, E] = qr(A);
  x = [A \ b, E(:, 1:m) * (R(:, 1:m) \ (Q' * b))]
end
```
> While this approach is certainly faster and uses less memory than Octave's minimum norm approach, this approach seems to be inferior in other ways.

The documentation references growth of the null space component relative to minimum-norm solution and lack of invariance w.r.t column permutations as disadvantages of the Matlab approach.

## Best Numpy equivalents:

- Matlab: `a\b` <-> Numpy: `linalg.solve(a, b)`
- Matlab: `a/b` <-> Numpy: ``


# `*` operator in Python for replication; copying objects

Watch out for any nested use of the replication operator.
```
(Pdb) test = [[True]*2]*2

(Pdb) [id(a) for a in test]
[139928639902064, 139928639902064]
```

This next bit is a little misleading; Python creates a new object for an entry in a *flat* list when that entry is modified. (see below)

```
(Pdb) [[id(b) for b in a] for a in test]
[[9544944, 9544944], [9544944, 9544944]]
```

Even `deepcopy` does not save us.

```
(Pdb) test1 = copy.deepcopy(test)

(Pdb) [id(a) for a in test1]
[139928639902064, 139928639902064]

(Pdb) [[id(b) for b in a] for a in test1]
[[9544944, 9544944], [9544944, 9544944]]

(Pdb) test1[0][0] = False

(Pdb) test1
[[False, True], [False, True]]

(Pdb) [[id(b) for b in a] for a in test1]
[[9544496, 9544944], [9544496, 9544944]]

```

But by stepping down one level, we only need regular `copy`.
```
(Pdb) test2 = [copy.copy(a) for a in test]

(Pdb) [id(a) for a in test2]
[139928639902136, 139928639902064]

(Pdb) [[id(b) for b in a] for a in test2]
[[9544944, 9544944], [9544944, 9544944]]

(Pdb) test2[0][0]=False

(Pdb) test2
[[False, True], [True, True]]

(Pdb) [[id(b) for b in a] for a in test2]
[[9544496, 9544944], [9544944, 9544944]]
```
