# abeliantensors
[![][travis-img]][travis-url] [![][codecov-img]][codecov-url]

abeliantensors is a Python 3 package that implements U(1) and Zn symmetry preserving
tensors, as described by Singh et al. in
[arXiv: 0907.2994](https://arxiv.org/abs/0907.2994) and
[arXiv: 1008.4774](https://arxiv.org/abs/1008.4774). abeliantensors has been designed
for use in tensor network algorithms, and works well with the
[ncon function](https://github.com/mhauru/ncon).

## Installation

If you just want to use the library:
```
pip install --user git+https://github.com/mhauru/abeliantensors
```

If you also want to modify and develop the library
```
git clone https://github.com/mhauru/abeliantensors
cd abeliantensors
pip install --user -e .[tests]
```
after which you can run the test suite by just calling `pytest`.

## Usage

abeliantensors exports classes `TensorU1`, `TensorZ2`, and `TensorZ3`. Other
cyclic groups Zn can be implemented with one-liners, see the file
`symmetrytensors.py` for examples. abeliantensors also exports a class called
`Tensor`, that is just a wrapper around regular numpy ndarrays, but that
implements the exact same interface as the symmetric tensor classes. This
allows for easy switching between utilizing and not utilizing the symmetry
preserving tensors by simply changing the class that is imported.

Each symmetric tensor has, in addition to its tensor elements, the following
pieces of what we call form data:
* `shape` describes the dimensions of the tensors, just like with numpy arrays.
  The difference is that for each index, the dimension isn't just a number, but
  a list of numbers, that sets how the vector space is partitioned by the
  irreducible representations (irreps) of the symmetry. So for instance
  `shape=[[2,3], [5,4]]` could be the shape of a Z2 symmetric matrix of
  dimensions 5 x 9, where the first 2 rows and 5 columns are associated with
  one of the two irreps of Z2, and the remaining 3 rows and 4 columns with the
  other.
* `qhape` is like `shape`, but lists the irrep charges instead of the
  dimensions. Irrep charges are often also called quantum numbers. In the above
  example `qhape=[[0,1], [0,1]]` would mark the first part of both the row and
  column space to belong to the trivial irrep of charge 0, and the second part
  to the irrep with charge 1. For Zn the possible charges are 0, 1, ..., n, for
  U(1) they are all positive and negative integers.
* `dirs` is a list of 1s and -1s, that gives a direction to each index: either
  1 for outgoing or -1 for ingoing.
* `charge` is an integer, the irrep charge associated to tensor. In most cases
  you want `charge=0`, which is also the default when creating new tensors.

Note that each element of the tensor is associated with one irrep charge for
each of the indices. The symmetry property is then that an element can only be
non-zero if the charges from each index, multiplied by the direction of that
index, add up to the charge of the tensor. Addition of charges for Zn tensors
is modulo n.  So for instance for a `charge=0` `TensorZ2` object this means
that the charges on each leg must add up to an even number for an element to be
non-zero. The whole point of this library is to store and use such symmetric
tensors in an efficient way, where we don't waste memory or computation time on
the elements we know are zero by symmetry, and can't accidentally let them be
non-zero.

Here's a simple nonsense example of how abeliantensors can be used:
```
import numpy as np
from abeliantensors import TensorZ2

# Create a symmetric tensor from an ndarray. All elements that should be zero
# by symmetry are simply discarded, whether they are zero or not.
sigmaz = np.array([[1, 0], [0, -1]])
sigmaz = TensorZ2.from_ndarray(
    sigmaz, shape=[[1, 1], [1, 1]], qhape=[[0, 1], [0, 1]], dirs=[1, -1]
)

# Create a random symmetric tensor.
a = TensorZ2.random(
    shape=[[3, 2], [2, 4], [4, 4], [1, 1]],
    qhape=[[0, 1]] * 4,
    dirs=[-1, 1, 1, -1],
)

# Do a singular value decomposition of a tensor, thinking of it as a matrix
# with some of the indices combined to a single matrix index, like one does
# with numpy.reshape. Here we combine indices 0 and 2 to form the left matrix
# index, and 1 and 3 to form the right one. The indices are reshaped back to
# the original form after the SVD, so U and V are in this case order-3 tensors.
U, S, V = a.svd([0, 2], [1, 3])

# You can also do a truncated SVD, in this case to truncating to dimension 4.
U, S, V = a.svd([0, 2], [1, 3], chis=4)

# We can contract tensors together easily using the ncon package.
# Note that conjugation flips the direction of each index, as well as the
# charge of the tensor, which in this case though is 0.
from ncon import ncon
aadg = ncon((a, a.conjugate()), ([1, 2, -1, -2], [1, 2, -11, -12]))

# Finally, knowing that aadg is Hermitian, do an eigenvalue
# decomposition of it, this time truncating not to a specific dimension, but
# to a maximum relative truncation error of 1e-5.
E, U = aadg.eig([0, 1], [2, 3], hermitian=True, eps=1e-5)
```

There are many other user-facing methods and features, but for more details,
see the docstrings in the package.

## Design and structure

The implementation is built on top of numpy, and the block-wise sparse
structure of the symmetry preserving tensors is implemented with Python
dictionaries. Here's a quick summary of what each file does.

`tensorcommon.py`: A parent class of all the other classes, `TensorCommon`,
that implements some higher-level features using the lower-level methods.

`abeliantensor.py`: All the fun is in here. Implements the class
`AbelianTensor`, that is the parent of all the symmetric tensor classes. This
includes implementations of various common tensor operations, such as
contractions and decompositions, preserving and making use of the block-wise
sparse structure these tensors have.

`tensor.py`: `Tensor`, the wrapper class for numpy arrays. It is designed so that
any call to a method of the `AbelianTensor` class is also a valid call to a
similarly named method of the `Tensor` class. All the symmetry-related
information is simply discarded and some underlying numpy function is called.
Even if one doesn't use symmetry preserving tensors, the `Tensor` class provides
some neat convenience functions, such as an easy-to-read one-liner for the
transpose-reshape-decompose-reshape-transpose procedure for singular value and
eigenvalue decompositions of tensors.

`symmetrytensors.py`: A small file that simply creates subclasses of
`AbelianTensor` for specific symmetry groups. If you need something other than
Z2, Z3 and U(1), check this file to see how you could add what you need.

`tests`: Plenty of tests for the various classes. The tests require the [ncon
package](https://github.com/mhauru/ncon), which pip automatically installs for
you. Most of the tests are based on generating a random instance of one of the
"fancy" tensor classes in this package, and confirming that the following
diagram commutes:
```
Fancy tensor ─── map to numpy ndarray ───> ndarray
    │                                         │
    │                                         │
Do the thing                             Do the thing
    │                                         │
    │                                         │
    V                                         V
Fancy tensor ─── map to numpy ndarray ───> ndarray
```


[travis-img]: https://travis-ci.org/mhauru/abeliantensors.svg?branch=master
[travis-url]: https://travis-ci.org/mhauru/abeliantensors
[codecov-img]: https://codecov.io/gh/mhauru/abeliantensors/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/mhauru/abeliantensors
