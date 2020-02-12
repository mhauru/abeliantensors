import numpy as np
import itertools as itt
import functools as fct
import warnings
from collections.abc import Iterable


class TensorCommon:
    """A base class for `Tensor` and `AbelianTensor`, that implements some
    higher level functions that are common to the two.

    Useful also for type checking as in ``isinstance(T, TensorCommon)``.
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Initializing tensors

    @classmethod
    def empty(cls, *args, **kwargs):
        """Initialize a tensor of given form with `np.empty`."""
        return cls.initialize_with(np.empty, *args, **kwargs)

    @classmethod
    def zeros(cls, *args, **kwargs):
        """Initialize a tensor of given form with `np.zeros`."""
        return cls.initialize_with(np.zeros, *args, **kwargs)

    @classmethod
    def ones(cls, *args, **kwargs):
        """Initialize a tensor of given form with `np.ones`."""
        return cls.initialize_with(np.ones, *args, **kwargs)

    @classmethod
    def random(cls, *args, **kwargs):
        """Initialize a tensor of given form with np.random.random_sample."""
        return cls.initialize_with(np.random.random_sample, *args, **kwargs)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    def form_str(self):
        """Return a string that describes the form of the tensor: the `shape`,
        `qhape` and `dirs`.
        """
        s = "shape: %s\nqhape: %s\ndirs: %s" % (
            str(self.shape),
            str(self.qhape),
            str(self.dirs),
        )
        return s

    @staticmethod
    def flatten_shape(shape):
        """Given a `shape` that may have dimensions divided between sectors,
        return a flattened shape, that has just the total dimension of each
        index.
        """
        try:
            return tuple(map(TensorCommon.flatten_dim, shape))
        except TypeError:
            return shape

    @staticmethod
    def flatten_dim(dim):
        """Given a `dim` for a single index that may be divided between
        sectors, return a flattened dim, that has just the total dimension of
        the index.
        """
        try:
            return sum(dim)
        except TypeError:
            return dim

    def norm_sq(self):
        """Return the Frobenius norm squared of the tensor."""
        conj = self.conj()
        all_inds = tuple(range(len(self.shape)))
        norm_sq = self.dot(conj, (all_inds, all_inds))
        return np.abs(norm_sq.value())

    def norm(self):
        """Return the Frobenius norm of the tensor."""
        return np.sqrt(self.norm_sq())

    @classmethod
    def default_trunc_err_func(cls, S, chi, norm_sq=None):
        """The default error function used when truncating decompositions:
        L_2 norm of the discarded singular or eigenvalues ``S[chi:]``, divided
        by the L_2 norm of the whole spectrum `S`.

        A keyword argument `norm_sq` gives the option of specifying the
        Frobneius norm manually, if for instance `S` isn't the full spectrum to
        start with.
        """
        if norm_sq is None:
            norm_sq = sum(S ** 2)
        sum_disc = sum(S[chi:] ** 2)
        err = np.sqrt(sum_disc / norm_sq)
        return err

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def to_matrix(
        self,
        left_inds,
        right_inds,
        dirs=None,
        return_transposed_shape_data=False,
    ):
        """Reshape the tensor into a matrix.

        The reshape is done by transposing `left_inds` to one side of `self`
        and `right_inds` to the other, and joining these indices so that the
        result is a matrix. On both sides, before reshaping, the indices are
        also transposed to the order given in `left`/`right_inds`. If one or
        both of `left`/right_inds is not provided the result is a vector or a
        scalar.

        `dirs` are the directions of the new indices. By default it is `[1,-1]`
        for matrices and `[1]` (respectively `[-1]`) if only `left_inds`
        (respectively `right_inds`) is provided.

        If `return_transposed_shape_data` is True then the `shape`, `qhape` and
        `dirs` of the tensor after all the transposing but before reshaping is
        returned as well.
        """
        if dirs is None:
            if len(left_inds) > 0 and len(right_inds) > 0:
                dirs = [1, -1]
            elif len(right_inds) > 0:
                dirs = [-1]
            elif len(left_inds) > 0:
                dirs = [1]
            else:
                dirs = []

        result = self.join_indices(
            left_inds,
            right_inds,
            dirs=dirs,
            return_transposed_shape_data=return_transposed_shape_data,
        )
        if return_transposed_shape_data:
            (
                result,
                transposed_shape,
                transposed_qhape,
                transposed_dirs,
            ) = result

        # join_indices does not return a matrix with left_inds as the first
        # index and right_inds as the second, so we may have to transpose.
        if left_inds and right_inds and left_inds[0] > right_inds[0]:
            result = result.swapaxes(1, 0)
            # Do the corresponding swap for the transposed form data as well.
            if return_transposed_shape_data:
                ts_left = transposed_shape[: len(right_inds)]
                ts_right = transposed_shape[len(right_inds) :]
                transposed_shape = ts_right + ts_left
                if transposed_qhape is not None:
                    qs_left = transposed_qhape[: len(right_inds)]
                    qs_right = transposed_qhape[len(right_inds) :]
                    transposed_qhape = qs_right + qs_left
                if transposed_dirs is not None:
                    qs_left = transposed_dirs[: len(right_inds)]
                    qs_right = transposed_dirs[len(right_inds) :]
                    transposed_dirs = qs_right + qs_left

        if return_transposed_shape_data:
            return result, transposed_shape, transposed_qhape, transposed_dirs
        else:
            return result

    def from_matrix(
        self,
        left_dims,
        right_dims,
        left_qims=None,
        right_qims=None,
        left_dirs=None,
        right_dirs=None,
    ):
        """Reshape a matrix back into a tensor, given the form data for the
        tensor.

        The counter part of `to_matrix`, `from_matrix` takes in a matrix and
        the `dims`, `qims` and `dirs` lists of the left and right indices that
        the resulting tensor should have. Mainly meant to be used so that one
        first calls `to_matrix`, takes note of the `transposed_shape_data` and
        uses that to reshape the matrix back to a tensor once one is done
        operating on the matrix.
        """
        indices = tuple(range(len(self.shape)))
        final_dims = ()
        final_qims = ()
        final_dirs = ()
        if indices:
            if left_dims:
                final_dims += (left_dims,)
                final_qims += (left_qims,)
                final_dirs += (left_dirs,)
            if right_dims:
                final_dims += (right_dims,)
                final_qims += (right_qims,)
                final_dirs += (right_dirs,)
        if left_qims is right_qims is None:
            final_qims = None
        if left_dirs is right_dirs is None:
            final_dirs = None
        return self.split_indices(
            indices, final_dims, qims=final_qims, dirs=final_dirs
        )

    def dot(self, other, indices):
        """Dot product of tensors.

        See `numpy.tensordot` on how to use this, the interface is exactly the
        same, except that this one is a method, not a function. The original
        tensors are not modified.
        """
        # We want to deal with lists, not tuples or bare integers
        a, b = indices
        if isinstance(a, Iterable):
            a = list(a)
        else:
            a = [a]
        if isinstance(b, Iterable):
            b = list(b)
        else:
            b = [b]
        # Check that 1) the number of contracted indices for self and other
        # match and 2) that the indices are compatible, i.e. okay to contract
        # with each other. In addition raise a warning if the dirs don't match.
        assert len(a) == len(b)
        assert all(
            itt.starmap(fct.partial(self.compatible_indices, other), zip(a, b))
        )
        if (
            self.dirs is not None
            and other.dirs is not None
            and not all(
                self.dirs[i] + other.dirs[j] == 0 for i, j in zip(a, b)
            )
        ):
            warnings.warn("dirs in dot do not match.")

        # Make lists of indices to contract (sum) and not contract (open) for
        # self (s_) and other (o_). Use them to transpose each tensor into a
        # matrix, so that the contraction becomes a matrix product.
        s_sum = a
        s_open = [i for i in range(len(self.shape)) if i not in a]
        o_sum = b
        o_open = [i for i in range(len(other.shape)) if i not in b]
        (
            self,
            s_transposed_shape,
            s_transposed_qhape,
            s_transposed_dirs,
        ) = self.to_matrix(s_open, s_sum, return_transposed_shape_data=True)
        (
            other,
            o_transposed_shape,
            o_transposed_qhape,
            o_transposed_dirs,
        ) = other.to_matrix(o_sum, o_open, return_transposed_shape_data=True)

        self = self.matrix_dot(other)
        del other  # Release memory for the reshaped other.

        # Gather all the necessary form data and reshape the resulting matrix
        # back into a tensor of the correct shape.
        l_dims = s_transposed_shape[: len(s_open)]
        r_dims = o_transposed_shape[len(o_sum) :]
        try:
            l_qims = s_transposed_qhape[: len(s_open)]
            r_qims = o_transposed_qhape[len(o_sum) :]
        except TypeError:
            l_qims = None
            r_qims = None
        try:
            l_dirs = s_transposed_dirs[: len(s_open)]
            r_dirs = o_transposed_dirs[len(o_sum) :]
        except TypeError:
            l_dirs = None
            r_dirs = None
        self = self.from_matrix(
            l_dims,
            r_dims,
            left_qims=l_qims,
            right_qims=r_qims,
            left_dirs=l_dirs,
            right_dirs=r_dirs,
        )
        return self

    def eig(self, a, b, *args, return_rel_err=False, **kwargs):
        """Eigenvalue decompose the tensor.

        Transpose indices `a` to be on one side of `self`, `b` on the other,
        and reshape `self` to a matrix. Then find the eigenvalues and
        eigenvectors of this matrix, and reshape the eigenvectors to have on
        the left side the indices that `self` had on its right side after
        transposing but before reshaping.

        If the keyword argument `hermitian` is True then the matrix that is
        formed after the reshape is assumed to be hermitian.

        Truncation works like with SVD.

        If the keyword argument `sparse` is True, a sparse eigenvalue
        decomposition, using power methods from `scipy.sparse.eigs` or `eigsh`,
        is used. This decomposition is done to find ``max(chis)`` eigenvalues,
        after which the decomposition may be truncated further if the
        truncation error so allows. Thus ``max(chis)`` should be much smaller
        than the full size of the matrix, if `sparse` is True.

        Output is ``S, U, [rel_err]``, where `S` is a vector of eigenvalues and
        `U` is a tensor such that the last index enumerates the eigenvectors of
        `self` in the sense that if ``u_i = U[...,i]`` then
        ``self.dot(u_i, (b, all_indices_of_u_i)) == S[i] * u_i``.
        `rel_err` is relative error in truncation, only returned if
        `return_rel_err` is True.

        The above syntax is precisely correct only for `Tensors`. For
        `AbelianTensors` the idea is the same, but eigenvalues and vectors come
        with quantum numbers so the syntax is slightly different. See
        `AbelianTensor.matrix_eig` for more details about what precisely
        happens.

        The original tensor is not modified by this method.
        """
        # Turn single indices into singleton tuples.
        if not isinstance(a, Iterable):
            a = (a,)
        if not isinstance(b, Iterable):
            b = (b,)
        # Reshape into a matrix and eigenvalue decompose as a matrix.
        (
            self,
            transposed_shape,
            transposed_qhape,
            transposed_dirs,
        ) = self.to_matrix(a, b, return_transposed_shape_data=True)
        S, U, rel_err = self.matrix_eig(*args, **kwargs)
        del self  # Release memory for the reshaped matrix.

        # Collect the necessary form data and reshape U into a tensor.
        U_dims = (transposed_shape[: len(a)], S.shape)
        if transposed_qhape is not None:
            U_qims = (transposed_qhape[: len(a)], S.qhape)
        else:
            U_qims = (None, None)
        if transposed_dirs is not None:
            U_dirs = (transposed_dirs[: len(a)], U.dirs[1:])
        else:
            U_dirs = (None, None)
        U = U.from_matrix(
            *U_dims,
            left_qims=U_qims[0],
            right_qims=U_qims[1],
            left_dirs=U_dirs[0],
            right_dirs=U_dirs[1]
        )
        ret_val = (S, U)
        if return_rel_err:
            ret_val += (rel_err,)
        return ret_val

    def svd(self, a, b, *args, return_rel_err=False, **kwargs):
        """Singular value decompose a tensor.

        Transpose indices `a` to be on one side of `self`, `b` on the other,
        and reshape `self` to a matrix. Then singular value decompose this
        matrix into ``U, S, V``. Finally reshape the unitary matrices to
        tensors that have a new index coming from the SVD, for `U` as the last
        index and for `V` as the first, and `U` to have indices a as its first
        indices and `V` to have indices `b` as its last indices.

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

        If the keyword argument `sparse` is True, a sparse singular value
        decomposition, using power methods from `scipy.sparse.svds`, is used.
        This decomposition is done to find ``max(chis)`` singular values, after
        which the decomposition may be truncated further if the truncation
        error so allows. Thus ``max(chis)`` should be much smaller than the
        full size of the matrix, if `sparse` is True.

        If `return_rel_err` is True then the relative truncation error is also
        returned.

        The return value is ``U, S, V, [rel_err]``. Here `S` is a vector of
        singular values and `U` and `V` are isometric tensors (unitary if the
        matrix that is SVDed is square and there is no truncation).
        ``U . S.diag() . V == self``, up to truncation errors.

        The original tensor is not modified by this method.
        """
        # Turn single indices into singleton tuples.
        if not isinstance(a, Iterable):
            a = (a,)
        if not isinstance(b, Iterable):
            b = (b,)

        (
            self,
            transposed_shape,
            transposed_qhape,
            transposed_dirs,
        ) = self.to_matrix(a, b, return_transposed_shape_data=True)

        U, S, V, rel_err = self.matrix_svd(*args, **kwargs)
        del self  # Release memory for the reshaped matrix.

        # Collect the necessary form data, and reshape U and V into tensors.
        U_dims = (transposed_shape[: len(a)], S.shape)
        V_dims = (S.shape, transposed_shape[len(a) :])
        if transposed_qhape is not None:
            U_qims = (transposed_qhape[: len(a)], S.qhape)
            V_qims = (S.qhape, transposed_qhape[len(a) :])
        else:
            U_qims = (None, None)
            V_qims = (None, None)
        if transposed_dirs is not None:
            U_dirs = (transposed_dirs[: len(a)], U.dirs[1:])
            V_dirs = (V.dirs[:1], transposed_dirs[len(a) :])
        else:
            U_dirs = (None, None)
            V_dirs = (None, None)
        U = U.from_matrix(
            *U_dims,
            left_qims=U_qims[0],
            right_qims=U_qims[1],
            left_dirs=U_dirs[0],
            right_dirs=U_dirs[1]
        )
        V = V.from_matrix(
            *V_dims,
            left_qims=V_qims[0],
            right_qims=V_qims[1],
            left_dirs=V_dirs[0],
            right_dirs=V_dirs[1]
        )

        ret_val = (U, S, V)
        if return_rel_err:
            ret_val += (rel_err,)
        return ret_val

    def _matrix_decomp_format_chis(self, chis, eps):
        """A common function for formatting the truncation parameters of SVD
        and eig. This is meant to be called by the matrix_svd and matrix_eig
        functions of subclasses.
        """
        if chis is None:
            # If chis isn't provided, it should be either the full list of
            # possible dimensions, if eps > 0, or just the full dimension, if
            # eps <= 0.
            min_dim = (
                min(
                    type(self).flatten_dim(self.shape[i])
                    for i in range(len(self.shape))
                )
                + 1
            )
            if eps > 0:
                chis = tuple(range(min_dim))
            else:
                chis = [min_dim]
        else:
            # Make sure chis is a list, and only includes the largest chi if
            # eps <= 0.
            try:
                chis = tuple(chis)
            except TypeError:
                chis = [chis]
            if eps <= 0:
                chis = [max(chis)]
            else:
                chis = sorted(chis)
        return chis

    def split(
        self,
        a,
        b,
        *args,
        return_rel_err=False,
        return_sings=False,
        weight="both",
        **kwargs
    ):
        """Split the tensor into two with an SVD.

        This is like an SVD, but takes the square root of the singular values
        and multiplies both unitaries with it, so that the tensor is split into
        two parts. Values are returned as
        ``US, [S], SV, [rel_err]``,
        where the ones in square brackets are only returned if the
        corresponding arguments, `return_rel_err` and `return_sings`, are True.

        The distribution of ``sqrt(S)`` onto the two sides can be changed with
        the keyword argument `weight`. If ``weight="left"`` (correspondingly
        ``"right"``) then `S` is multiplied into `U` (correspondingly `V`). By
        default ``weight="both"``, in which the square root is evenly
        distributed.
        """
        svd_result = self.svd(
            a, b, *args, return_rel_err=return_rel_err, **kwargs
        )
        U, S, V = svd_result[0:3]

        # Multiply the singular values into U, V or both, as directed by
        # `weight`.
        weight = weight.strip().lower()
        if weight in ("both", "split", "center", "centre", "c", "middle", "m"):
            S_sqrt = S.sqrt()
            U = U.multiply_diag(S_sqrt, -1, direction="right")
            V = V.multiply_diag(S_sqrt, 0, direction="left")
        elif weight in ("left", "l", "a", "u"):
            U = U.multiply_diag(S, -1, direction="right")
        elif weight in ("right", "r", "b", "v"):
            V = V.multiply_diag(S, 0, direction="left")
        else:
            raise ValueError("Unknown value for weight: {}".format(weight))

        if return_sings:
            ret_val = U, S, V
        else:
            ret_val = U, V
        if return_rel_err:
            ret_val += (svd_result[3],)
        return ret_val
