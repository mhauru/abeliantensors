import numpy as np
import operator as opr
import functools as fct
import scipy.sparse.linalg as spsla
import warnings
from .tensorcommon import TensorCommon
from collections.abc import Iterable


class Tensor(TensorCommon, np.ndarray):
    """A wrapper class for NumPy arrays.

    This class implements no new functionality beyond NumPy arrays, but simply
    provides them with the same interface that is used by the symmetry
    preserving tensor classes. `Tensors` always have ``qhape == None``, ``dirs
    == None`` and ``charge == 0``.

    Note that `Tensor` is a subclass of both `TensorCommon` and
    `numpy.ndarray`, so many NumPy functions work directly on `Tensors`. It's
    preferable to use methods of the `Tensor` class instead though, because it
    allows to easily switching to a symmetric tensor class without modifying
    the code.
    """

    # Practically all methods of Tensor take keyword argument like qhape, dirs,
    # and charge, and do nothing with them. This is to match the interface of
    # AbelianTensor, where these keyword arguments are needed.

    def __new__(
        cls,
        shape,
        *args,
        qhape=None,
        charge=None,
        invar=None,
        dirs=None,
        **kwargs
    ):
        shape = cls.flatten_shape(shape)
        res = np.ndarray(shape, *args, **kwargs).view(cls)
        return res

    @classmethod
    def initialize_with(
        cls,
        numpy_func,
        shape,
        *args,
        qhape=None,
        charge=None,
        invar=None,
        dirs=None,
        **kwargs
    ):
        """Use the given `numpy_func` to initialize a tensor of `shape`."""
        shape = cls.flatten_shape(shape)
        res = numpy_func(shape, *args, **kwargs).view(cls)
        return res

    @classmethod
    def eye(cls, dim, qim=None, qodulus=None, *args, **kwargs):
        """Return the identity matrix of the given dimension dim."""
        dim = cls.flatten_dim(dim)
        res = np.eye(dim, *args, **kwargs).view(cls)
        return res

    def diag(self, **kwargs):
        """Return the diagonal of a given matrix, or a diagonal matrix with the
        given values on the diagonal.
        """
        res = np.diag(self).view(Tensor)
        return res

    # Every tensor object has the attributes qhape, dirs and charge just to
    # match the interface of AbelianTensor.
    qhape = None
    dirs = None
    charge = 0

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # To and from numpy arrays

    def to_ndarray(self, **kwargs):
        """Return the corresponding NumPy array, as a copy."""
        return np.asarray(self.copy())

    @classmethod
    def from_ndarray(cls, a, **kwargs):
        """Given an NumPy array, return the corresponding `Tensor` instance."""
        if isinstance(a, np.ndarray):
            res = a.copy().view(cls)
        else:
            res = np.array(a).view(cls)
        return res

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Methods for slicing, setting and getting elements

    def fill(self, value):
        """Fill the tensor with a scalar value."""
        return np.ndarray.fill(self, value)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Operator methods

    def conjugate(self):
        """Return the complex conjugate."""
        return np.ndarray.conj(self)

    def abs(self):
        """Return the element-wise absolute value."""
        return np.ndarray.__abs__(self)

    def any(self, *args, **kwargs):
        """Return whether any elements are True.

        See `numpy.ndarray.any` for details.
        """
        return np.ndarray.any(self, *args, **kwargs)

    def all(self, *args, **kwargs):
        """Return whether all elements are True.

        See `numpy.ndarray.all` for details.
        """
        return np.ndarray.all(self, *args, **kwargs)

    def allclose(self, other, *args, **kwargs):
        """Return whether self and other are nearly element-wise equal.

        See `numpy.allclose` for details.
        """
        return np.allclose(self, other, *args, **kwargs)

    def log(self):
        """Return the element-wise natural logarithm."""
        return np.log(self)

    def exp(self):
        """Return the element-wise exponential."""
        return np.exp(self)

    def sqrt(self):
        """Return the element-wise square root."""
        return np.sqrt(self)

    def average(self):
        """Return the element-wise average."""
        return np.average(self)

    def sign(self):
        """Return the element-wise sign."""
        return np.sign(self)

    def real(self):
        """Return the real part."""
        return super(Tensor, self).real

    def imag(self):
        """Return the imaginary part."""
        return super(Tensor, self).imag

    def sum(self):
        """Return the element-wise sum."""
        return super(Tensor, self).sum().value()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    def isscalar(self):
        """Return whether this `Tensor` is a scalar."""
        return not bool(self.shape)

    def compatible_indices(self, other, i, j):
        """Return True if index `i` of `self` and index `j` of `other` are of
        the same dimension.
        """
        return self.shape[i] == other.shape[j]

    def flip_dir(self, axis):
        """A no-op, that returns a view.

        The corresponding method of `AbelianTensor` flips the direction of an
        index, but directions are meaningless for `Tensors`.
        """
        res = self.view()
        return res

    def expand_dims(self, axis, direction=1):
        """Add to `self` a new singleton index, at position `axis`."""
        res = np.expand_dims(self, axis)
        if not isinstance(res, Tensor):
            res = type(self).from_ndarray(res)
        return res

    def value(self):
        """For a scalar tensor, return the scalar. For a non-scalar one, raise
        a `ValueError`.
        """
        if not self.isscalar():
            raise ValueError("value called on a non-scalar tensor.")
        else:
            return self[()]

    @classmethod
    def check_form_match(
        cls,
        tensor1=None,
        tensor2=None,
        qhape1=None,
        shape1=None,
        dirs1=None,
        qhape2=None,
        shape2=None,
        dirs2=None,
        qodulus=None,
    ):
        """Check that the given two tensors have the same form in the sense
        that, i.e. that their indices have the same dimensions. Instead of
        giving two tensors, two shapes can also be given.
        """
        if tensor1 is not None:
            shape1 = tensor1.shape
        if tensor2 is not None:
            shape2 = tensor2.shape
        return shape1 == shape2

    @classmethod
    def _find_trunc_dim(
        cls,
        S,
        chis=None,
        eps=0,
        break_degenerate=False,
        degeneracy_eps=1e-6,
        trunc_err_func=None,
        norm_sq=None,
    ):
        """A utility function that is used by eigenvalue and singular value
        decompositions.

        Given a information generated by eig and SVD during the decomposition,
        find out what bond dimension we should truncate the decomposition to,
        and what the resulting truncation error is.
        """
        S = np.abs(S)
        if trunc_err_func is None:
            # The user may provide norm_sq if the given S has been pretruncated
            # already. If not, compute it from the given S.
            if norm_sq is None:
                norm_sq = sum(S ** 2)
            trunc_err_func = fct.partial(
                cls.default_trunc_err_func, norm_sq=norm_sq
            )
        # Find the smallest chi for which the error is small enough. If none
        # is found, use the largest chi.
        if sum(S) != 0:
            last_out = S[0]
            for chi in chis:
                if not break_degenerate:
                    # Make sure that we don't break degenerate singular values
                    # by including one but not the other.
                    while 0 < chi < len(S):
                        last_in = S[chi - 1]
                        last_out = S[chi]
                        rel_diff = np.abs(last_in - last_out)
                        avrg = (last_in + last_out) / 2
                        if avrg != 0:
                            rel_diff /= avrg
                        if rel_diff < degeneracy_eps:
                            chi -= 1
                        else:
                            break
                err = trunc_err_func(S, chi)
                if err <= eps or last_out == 0.0:
                    break
        else:
            err = 0
            chi = min(chis)
        return chi, err

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def join_indices(
        self, *inds, return_transposed_shape_data=False, **kwargs
    ):
        """Join indices together in the spirit of reshape.

        `inds` is either a iterable of indices, in which case they are joined,
        or a iterable of iterables of indices, in which case the indices listed
        in each element of `inds` will be joined.

        Before any joining is done the indices are transposed so that for every
        batch of indices to be joined the first remains in place and the others
        are moved to be after in the order given. The order in which the
        batches are given does not matter.

        If `return_transposed_shape_data` is True, then the shape of the tensor
        after transposing but before reshaping is returned as well, in addition
        to None and None, that take the place of `transposed_qhape` and
        `transposed_dirs` of `AbelianTensor`.

        The method does not modify the original tensor.
        """
        # Format index_batches to be a list of lists of indices.
        if isinstance(inds[0], Iterable):
            index_batches = list(map(list, inds))
        else:
            index_batches = [list(inds)]
        # Remove empty batches.
        index_batches = [b for b in index_batches if b]

        # Create the permutation for transposing the tensor. At the same time
        # transpose and sort index_batches.
        # We create trivial one-index batches for all the indices that are not
        # going to be joined, so that all indices are in some batch. Then we
        # sort the batches by the first index in each one.
        joined = set(sum(index_batches, []))
        not_joined = [[i] for i in range(len(self.shape)) if i not in joined]
        all_in_batches = not_joined + index_batches
        all_in_batches.sort(key=opr.itemgetter(0))
        # The batches are now in right order, and we just have to turn this
        # into a permutation of the indices.
        perm = sum(all_in_batches, [])
        # Filter out the trivial batches we added a few lines above.
        index_batches = [batch for batch in all_in_batches if len(batch) > 1]
        # Sort the indices inside each batch according to the permutation perm.
        index_batches = [list(map(perm.index, b)) for b in index_batches]
        self = self.transpose(perm)

        if return_transposed_shape_data:
            transposed_shape = self.shape

        # Find out the shape the tensor should have after joining indices.
        shp = list(self.shape)
        # Traverse the batches in reversed order, because we'll be removing
        # elements from shp, and don't want to mess up the indexing.
        for batch in reversed(index_batches):
            # For each index batch, multiple the dimensions of the indices in
            # the batch to get the new total dimension.
            new_dim = fct.reduce(opr.mul, map(shp.__getitem__, batch))
            # Insert the new total dimension into shp, and remove the
            # dimensions of the indices that the reshape removes.
            shp[batch[0]] = new_dim
            del shp[batch[1] : batch[0] + len(batch)]
        self = self.reshape(shp)

        if return_transposed_shape_data:
            return self, transposed_shape, None, None
        else:
            return self

    def split_indices(self, indices, dims, qims=None, **kwargs):
        """Splits indices in the spirit of reshape.

        `indices` is an iterable of indices to be split. `dims` is an iterable
        of iterables such that ``dims[i]`` is an iterable of lists of
        dimensions, each list giving the dimensions along a new index that will
        come out of splitting ``indices[i]``.

        An example clarifies:
        Suppose `self` has `shape` ``[dim1, dim2, dim3, dim4]``. Suppose then
        that ``indices = [1,3]``, ``dims = [[dimA, dimB], [dimC, dimD]]``.
        Then the resulting tensor will have ``shape = [dim1, dimA, dimB, dim3,
        dimC, dimD]``, assuming that that `dims` and are such that joining
        `dimA` and `dimB` gives `dim2`, etc.

        Instead of a list of indices a single index may be given.
        Correspondingly `dims` should then have one level of depth less as
        well.

        `split_indices` never modifies the original tensor.
        """
        # Format the input so that indices is a list and dim_batches is a list
        # of lists.
        if isinstance(indices, Iterable):
            assert len(indices) == len(dims)
            indices = list(indices)
            dim_batches = list(map(list, dims))
        else:
            indices = [indices]
            dim_batches = [list(dims)]
        dim_batches = [
            [type(self).flatten_dim(dim) for dim in batch]
            for batch in dim_batches
        ]

        if not indices:
            return self.view()

        # Sort indices and dim_batches according to reversed indices. This is
        # necessary for the next step to work.
        indices, dim_batches = zip(
            *sorted(zip(indices, dim_batches), reverse=True)
        )
        # Compute the new shape.
        new_shape = list(self.shape)
        for ind, batch in zip(indices, dim_batches):
            new_shape[ind : ind + 1] = batch

        # Reshape
        res = self.reshape(new_shape)
        return res

    def multiply_diag(self, diag_vect, axis, *args, **kwargs):
        """Multiply by a diagonal matrix on one axis.

        The result of `multiply_diag` is the same as
        ``self.dot(diag_vect.diag(), (axis, 0))``
        This operation is just done without constructing the full diagonal
        matrix.
        """
        if len(diag_vect.shape) != 1:
            raise ValueError(
                "The `diag_vect` argument of multiply_diag must be a vector."
            )
        if axis < 0:
            axis = len(self.shape) + axis
        res = self.swapaxes(-1, axis)
        res = res * diag_vect
        res = res.swapaxes(-1, axis)
        return res

    def trace(self, axis1=0, axis2=1):
        """Return the trace over indices `axis1` and `axis2`."""
        # We assert that the tensor is square with respect to axis1 and axis2,
        # to follow as closely as possible what AbelianTensor does.
        assert self.compatible_indices(self, axis1, axis2)
        trace = super(Tensor, self).trace(axis1=axis1, axis2=axis2)
        return type(self).from_ndarray(trace)

    def dot(self, B, indices):
        """Dot product of tensors.

        See `numpy.tensordot` on how to use this, the interface is exactly the
        same, except that this one is a method, not a function. The original
        tensors are not modified.
        """
        result = np.tensordot(self, B, indices)
        if not isinstance(result, Tensor):
            result = type(self).from_ndarray(result)
        return result

    # This one actually isn't necessary TensorCommon covers this, but the
    # implementation is just some much simpler using np.tensordot.
    def matrix_dot(self, B):
        """Take the dot product of two tensors of order < 3 (i.e. vectors or
        matrices).
        """
        result = np.dot(self, B)
        if not isinstance(result, TensorCommon):
            result = type(self).from_ndarray(result)
        return result

    def matrix_eig(
        self,
        chis=None,
        eps=0,
        print_errors="deprecated",
        hermitian=False,
        break_degenerate=False,
        degeneracy_eps=1e-6,
        sparse=False,
        trunc_err_func=None,
    ):
        """Find eigenvalues and eigenvectors of a matrix.

        The input must be a square matrix.

        If `hermitian` is True the matrix is assumed to be hermitian.

        Truncation works like for SVD, see the documentation there for more.

        If `sparse` is True, a sparse eigenvalue decomposition, using power
        methods from `scipy.sparse.eigs` or `eigsh`, is used. This
        decomposition is done to find ``max(chis)`` eigenvalues, after which
        the decomposition may be truncated further if the truncation error so
        allows. Thus ``max(chis)`` should be much smaller than the full size of
        the matrix, if `sparse` is True.

        The return values is ``S, U, rel_err``, where `S` is a vector of
        eigenvalues and `U` is a matrix that has as its columns the
        eigenvectors. `rel_err` is the truncation error.
        """
        if print_errors != "deprecated":
            msg = (
                "The `print_errors` keyword argument has been deprecated, "
                "and has no effect. Rely instead on getting the error as a "
                "return value, and print it yourself."
            )
            warnings.warn(msg)
        chis = self._matrix_decomp_format_chis(chis, eps)
        mindim = min(self.shape)
        maxchi = max(chis)
        if sparse and maxchi < mindim - 1:
            if hermitian:
                S, U = spsla.eigsh(self, k=maxchi, return_eigenvectors=True)
            else:
                S, U = spsla.eigs(self, k=maxchi, return_eigenvectors=True)
            norm_sq = self.norm_sq()
        else:
            if hermitian:
                S, U = np.linalg.eigh(self)
            else:
                S, U = np.linalg.eig(self)
            norm_sq = None
        order = np.argsort(-np.abs(S))
        S = S[order]
        U = U[:, order]
        # Truncate, if truncation dimensions are given.
        chi, rel_err = type(self)._find_trunc_dim(
            S,
            chis=chis,
            eps=eps,
            break_degenerate=break_degenerate,
            degeneracy_eps=degeneracy_eps,
            trunc_err_func=trunc_err_func,
            norm_sq=norm_sq,
        )
        # Truncate
        S = S[:chi]
        U = U[:, :chi]
        if not isinstance(S, TensorCommon):
            S = type(self).from_ndarray(S)
        if not isinstance(U, TensorCommon):
            U = type(self).from_ndarray(U)
        return S, U, rel_err

    def matrix_svd(
        self,
        chis=None,
        eps=0,
        print_errors="deprecated",
        break_degenerate=False,
        degeneracy_eps=1e-6,
        sparse=False,
        trunc_err_func=None,
    ):
        """Singular value decompose a matrix.

        The optional argument `chis` is a list of bond dimensions. The SVD is
        truncated to one of these dimensions `chi`, meaning that only `chi`
        largest singular values are kept. If `chis` is a single integer (either
        within a singleton list or just as a bare integer) this dimension is
        used. If ``eps == 0``, the largest value in `chis` is used. Otherwise
        the smallest `chi` in `chis` is used, such that the relative error made
        in the truncation is smaller than `eps`. The truncation error is by
        default the Frobenius norm of the difference, but can be specified with
        the keyword agument `trunc_err_func`.

        An exception to the above is made by degenerate singular values. By
        default truncation is never done so that some singular values are
        included while others of the same value are left out. If this is about
        to happen, `chi` is decreased so that none of the degenerate singular
        values are included. This default behavior can be changed with the
        keyword argument `break_degenerate`. The default threshold for when
        singular values are considered degenerate is 1e-6. This can be changed
        with the keyword argument `degeneracy_eps`.

        If `sparse` is True, a sparse SVD, using power methods from
        `scipy.sparse.svds`, is used. This SVD is done to find ``max(chis)``
        singular values, after which the decomposition may be truncated further
        if the truncation error so allows. Thus ``max(chis)`` should be much
        smaller than the full size of the matrix, if `sparse` is True.

        The return value is``U, S, V, rel_err``, where `S` is a vector and `U`
        and `V` are unitary matrices. They are such that ``U.diag(S).V ==
        self``, where the equality is appromixate if there is truncation.
        `rel_err` is the truncation error.
        """
        if print_errors != "deprecated":
            msg = (
                "The `print_errors` keyword argument has been deprecated, "
                "and has no effect. Rely instead on getting the error as a "
                "return value, and print it yourself."
            )
            warnings.warn(msg)
        chis = self._matrix_decomp_format_chis(chis, eps)
        mindim = min(self.shape)
        maxchi = max(chis)
        if sparse and maxchi < mindim - 1:
            U, S, V = spsla.svds(self, k=maxchi, return_singular_vectors=True)
            norm_sq = self.norm_sq()
        else:
            U, S, V = np.linalg.svd(self, full_matrices=False)
            norm_sq = None
        S = Tensor.from_ndarray(S)
        # Truncate, if truncation dimensions are given.
        chi, rel_err = type(self)._find_trunc_dim(
            S,
            chis=chis,
            eps=eps,
            break_degenerate=break_degenerate,
            degeneracy_eps=degeneracy_eps,
            trunc_err_func=trunc_err_func,
            norm_sq=norm_sq,
        )
        # Truncate.
        S = S[:chi]
        U = U[:, :chi]
        V = V[:chi, :]
        if not isinstance(U, TensorCommon):
            U = type(self).from_ndarray(U)
        if not isinstance(V, TensorCommon):
            V = type(self).from_ndarray(V)
        return U, S, V, rel_err
