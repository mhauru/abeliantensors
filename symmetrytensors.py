import numpy as np
import collections
import itertools
import sys
from functools import reduce
from tensors.abeliantensor import AbelianTensor

class TensorZQ(AbelianTensor):

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Initializers and factory methods

    def __init__(self, shape, *args, qhape=None, qodulus=None, **kwargs):
        if qhape is None:
            qhape = type(self).shape_to_qhape(shape)
        qodulus = type(self).qodulus
        return super(TensorZQ, self).__init__(shape, *args, qhape=qhape,
                                              qodulus=qodulus, **kwargs)

    @classmethod
    def eye(cls, dim, qim=None, qodulus=None, dtype=np.float_):
        if qim is None:
            qim = cls.dim_to_qim(dim)
        qodulus = cls.qodulus
        return super(TensorZQ, cls).eye(dim, qim=qim, qodulus=qodulus,
                     dtype=np.float_)

    @classmethod
    def initialize_with(cls, numpy_func, shape, *args,
                        qhape=None, qodulus=None, **kwargs):
        if qhape is None:
            qhape = cls.shape_to_qhape(shape)
        qodulus = cls.qodulus
        return super(TensorZQ, cls).initialize_with(numpy_func, shape, *args,
                                                    qhape=qhape,
                                                    qodulus=qodulus, **kwargs)



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # To and from normal numpy arrays

    @classmethod
    def from_ndarray(cls, a, *args, shape=None, qhape=None, qodulus=None,
                     **kwargs):
        if qhape is None:
            qhape = cls.shape_to_qhape(shape)
        qodulus = cls.qodulus
        return super(TensorZQ, cls).from_ndarray(a, *args, shape=shape,
                                                 qhape=qhape, qodulus=qodulus,
                                                 **kwargs)



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    @classmethod
    def dim_to_qim(cls, dim):
        return [0] if len(dim)==1 else list(range(cls.qodulus))

    @classmethod
    def shape_to_qhape(cls, shape):
        return [cls.dim_to_qim(dim) for dim in shape]


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def split_indices(self, indices, dims, qims=None, dirs=None):
        # Buildind qims.
        if qims is None:
            if isinstance(indices, collections.Iterable):
                qims = [type(self).shape_to_qhape(dim) for dim in dims]
            else:
                qims = type(self).shape_to_qhape(dims)
        return super(TensorZQ, self).split_indices(indices, dims, qims=qims,
                                                   dirs=dirs)


#=============================================================================#


class TensorZ2(TensorZQ):
    qodulus = 2

class TensorZ3(TensorZQ):
    qodulus = 3

class TensorU1(AbelianTensor):
    qodulus = None

