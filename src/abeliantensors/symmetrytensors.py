import numpy as np
from .abeliantensor import AbelianTensor
from collections.abc import Iterable


class TensorZN(AbelianTensor):
    """A symmetric tensor class for the cyclic group of order N.

    See `AbelianTensor` for the details: A `TensorZN` is just an
    `AbelianTensor` for which addition of charges is done modulo N.
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Initializers and factory methods

    # We implement some of the initialization methods so that if the quantum
    # numbers aren't given in qhape, they are automatically generated.

    def __init__(self, shape, *args, qhape=None, qodulus=None, **kwargs):
        if qhape is None:
            qhape = type(self)._shape_to_qhape(shape)
        qodulus = type(self).qodulus
        return super(TensorZN, self).__init__(
            shape, *args, qhape=qhape, qodulus=qodulus, **kwargs
        )

    @classmethod
    def eye(cls, dim, qim=None, qodulus=None, dtype=np.float_):
        """Return the identity matrix of the given dimension `dim`."""
        if qim is None:
            qim = cls._dim_to_qim(dim)
        qodulus = cls.qodulus
        return super(TensorZN, cls).eye(
            dim, qim=qim, qodulus=qodulus, dtype=np.float_
        )

    @classmethod
    def initialize_with(
        cls, numpy_func, shape, *args, qhape=None, qodulus=None, **kwargs
    ):
        """Return a tensor of the given `shape`, initialized with
        `numpy_func`.
        """
        if qhape is None:
            qhape = cls._shape_to_qhape(shape)
        qodulus = cls.qodulus
        return super(TensorZN, cls).initialize_with(
            numpy_func, shape, *args, qhape=qhape, qodulus=qodulus, **kwargs
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # To and from normal numpy arrays

    @classmethod
    def from_ndarray(
        cls, a, *args, shape=None, qhape=None, qodulus=None, **kwargs
    ):
        """Build a `TensorZN` out of a given NumPy array, using the provided
        form data.

        If `qhape` is not provided, it is automatically generated based on
        `shape` to be ``[0, ..., N]`` for each index. See
        `AbelianTensor.from_ndarray` for more documentation.
        """
        if qhape is None:
            qhape = cls._shape_to_qhape(shape)
        qodulus = cls.qodulus
        return super(TensorZN, cls).from_ndarray(
            a, *args, shape=shape, qhape=qhape, qodulus=qodulus, **kwargs
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    @classmethod
    def _dim_to_qim(cls, dim):
        """Given the dimensions of sectors along an index, generate the the
        corresponding default quantum numbers.
        """
        return [0] if len(dim) == 1 else list(range(cls.qodulus))

    @classmethod
    def _shape_to_qhape(cls, shape):
        """Given the `shape` of a tensor, generate the the corresponding default
        quantum numbers.
        """
        return [cls._dim_to_qim(dim) for dim in shape]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def split_indices(self, indices, dims, qims=None, dirs=None):
        """Split indices in the spirit of reshape.

        If `qhape` is not provided, it is automatically generated based on
        `shape` to be ``[0, ..., N]`` for each index. See `AbelianTensor.split`
        for more documentation.
        """
        # Buildind qims.
        if qims is None:
            if isinstance(indices, Iterable):
                qims = [type(self)._shape_to_qhape(dim) for dim in dims]
            else:
                qims = type(self)._shape_to_qhape(dims)
        return super(TensorZN, self).split_indices(
            indices, dims, qims=qims, dirs=dirs
        )


class TensorZ2(TensorZN):
    """A class for Z2 symmetric tensors.

    See the parent class `AbelianTensor` for details.
    """

    qodulus = 2


class TensorZ3(TensorZN):
    """A class for Z3 symmetric tensors.

    See the parent class `AbelianTensor` for details.
    """

    qodulus = 3


class TensorU1(AbelianTensor):
    """A class for U(1) symmetric tensors.

    See the parent class `AbelianTensor` for details.
    """

    qodulus = None
