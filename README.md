# Tensors

Tensors is a Python 3 package that implements Abelian symmetry preserving
tensors, as described in
https://arxiv.org/abs/0907.2994
https://arxiv.org/abs/1008.4774

The implementation is built on top of numpy, and the block-wise sparse
structure of the symmetry preserving tensors is implemented with Python
dictionaries.

At the core of the package is the class AbelianTensor, that implements tensors
with Abelian symmetries. It is subclassed to create the classes TensorZ2,
TensorZ3 and TensorU1, for the symmetry groups $Z_2$, $Z_3$ and $U(1)$.
Implementing other symmetries should vary from quite easy to trivial. In
addition, there's a class called Tensor, that is simply a wrapper around
regular numpy arrays, that provides an interface for numpy arrays that is
exactly the same as the one used by symmetric tensors. This allows for easy
switching between utilizing and not utilizing the symmetry preserving tensors
by simply changing the class that is imported in the beginning.

Tensors works well with the NCon function, implemented here: `TODO`

So far Tensors has been used solely for my own work in tensor networks. Thus,
documentation may be lacking or out of date (it does exist though!), and
features that I have not needed may be missing. I hope that in the future I'll
have time polish the package to an easy-to-pick-up tool, but in the mean time,
let me know if you would have a use for it and I'll be glad to help you get
going.

### The files
`tensorcommon.py`: A parent class of all the other classes, TensorCommon, that
implements some higher-level features using the lower-level functions.

`tensor.py`: Tensor, the wrapper class for numpy arrays. It is designed so that
any call to a function of the AbelianTensor class is also a valid call to a
similarly named function of the Tensor class, and all the symmetry-related
information is simply discarded. Even if one doesn't use symmetry preserving
tensors, the Tensor class provides some neat convenience functions, which
provide easy one-line interfaces for instance for the
transpose-reshape-decompose-reshape-transpose chain for singular value and
eigenvalue decompositions of tensors.

`abeliantensor.py`: All the fun is in here. Implementations of various common
tensor operations, such as contractions and decompositions, for Abelian
symmetry preserving tensors.

`symmetrytensors.py`: A small file that simply creates subclasses of
AbelianTensor for specific symmetry groups. If you need something other than
$Z_2$, $Z_3$ and $U_1$, check this file to see how you could add what you need.

`tests`: Plenty of tests for the various classes, most of which are based on
generating a random tensor as an instance of one of the "fancy" tensor classes
in this package, and confirming that the following diagram commutes:
```
Fancy tensor --- map to numpy ndarray ---> ndarray
    |                                         |
    |                                         |
Do stuff                                  Do stuff
    |                                         |
    |                                         |
    V                                         V
Fancy tensor --- map to numpy ndarray ---> ndarray
```

