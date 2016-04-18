import numpy as np
import collections
import itertools
import functools
import warnings


class TensorCommon:
    """ A base class for Tensor and AbelianTensor, that implements some
    higher level functions that are common to the two. Useful also for
    type checking as in isinstance(T, TensorCommon).
    """

    @classmethod
    def empty(cls, *args, **kwargs):
        return cls.initialize_with(np.empty, *args, **kwargs)

    @classmethod
    def zeros(cls, *args, **kwargs):
        return cls.initialize_with(np.zeros, *args, **kwargs)

    @classmethod
    def ones(cls, *args, **kwargs):
        return cls.initialize_with(np.ones, *args, **kwargs)

    @classmethod
    def random(cls, *args, **kwargs):
        return cls.initialize_with(np.random.random_sample, *args, **kwargs)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    def form_str(self):
        s = "shape: %s\nqhape: %s\ndirs: %s"%(
                str(self.shape), str(self.qhape), str(self.dirs))
        return s

    @staticmethod
    def flatten_shape(shape):
        try:
            return tuple(map(TensorCommon.flatten_dim, shape))
        except TypeError:
            return shape

    @staticmethod
    def flatten_dim(dim):
        try:
            return sum(dim)
        except TypeError:
            return dim

    def norm_sq(self):
        conj = self.conj()
        all_inds = tuple(range(len(self.shape)))
        norm_sq = self.dot(conj, (all_inds, all_inds))
        return np.abs(norm_sq.value())

    def norm(self):
        return np.sqrt(self.norm_sq())


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def to_matrix(self, left_inds, right_inds, dirs=None,
                  return_transposed_shape_data=False):
        """ Transposes left_inds to one side of self and right_inds to
        the other, and joins these indices so that the result is a
        matrix. On both sides, before reshaping, the indices are also
        transposed to the order given in left/right_inds. If one or both
        of left/right_inds is not provided the result is a vector or a
        scalar. 

        dirs are the directions of the new indices. By default it is
        [1,-1] for matrices and [1] (respectively [-1]) if only
        left_inds (respectively right_inds) is provided.

        If return_transposed_shape_data is True then the shape, qhape
        and dirs of the tensor after all the transposing but before
        reshaping is returned as well.
        """
        if dirs is None:
            if len(left_inds) > 0 and len(right_inds) > 0:
                dirs = [1,-1]
            elif len(right_inds) > 0:
                dirs = [-1]
            elif len(left_inds) > 0:
                dirs = [1]
            else:
                dirs = []

        result = self.join_indices(left_inds, right_inds, dirs=dirs,
                                   return_transposed_shape_data=\
                                           return_transposed_shape_data)
        if return_transposed_shape_data:
            result, transposed_shape, transposed_qhape, transposed_dirs =\
                    result

        # join_indices does not return a matrix with left_inds as the
        # first index and right_inds as the second, so we may have to
        # transpose.
        if left_inds and right_inds and left_inds[0] > right_inds[0]:
            result = result.swapaxes(1,0)
            if return_transposed_shape_data:
                ts_left = transposed_shape[:len(right_inds)]
                ts_right = transposed_shape[len(right_inds):]
                transposed_shape = ts_right + ts_left
                if transposed_qhape is not None:
                    qs_left = transposed_qhape[:len(right_inds)]
                    qs_right = transposed_qhape[len(right_inds):]
                    transposed_qhape = qs_right + qs_left
                if transposed_dirs is not None:
                    qs_left = transposed_dirs[:len(right_inds)]
                    qs_right = transposed_dirs[len(right_inds):]
                    transposed_dirs = qs_right + qs_left

        if return_transposed_shape_data:
            return result, transposed_shape, transposed_qhape, transposed_dirs
        else:
            return result


    def from_matrix(self, left_dims, right_dims,
                    left_qims=None, right_qims=None,
                    left_dirs=None, right_dirs=None):
        """ The counter part of to_matrix, from_matrix takes in a matrix
        and the dims, qims and dirs lists of the left and right indices
        that the resulting tensor should have. Mainly meant to be used
        so that one first calls to_matrix, takes note of the
        transposed_shape_data and uses that to reshape the matrix back
        to a tensor once one is done operating on the matrix.
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
        return self.split_indices(indices, final_dims, qims=final_qims,
                                  dirs=final_dirs)


    def dot(self, other, indices):
        """ Dot product of tensors. See numpy.tensordot on how to use
        this, the interface is exactly the same, except that this one is
        a method, not a function. The original tensors are not modified.
        """
        # We want to deal with lists, not tuples or bare integers
        a,b = indices
        if isinstance(a, collections.Iterable):
            a = list(a)
        else:
            a = [a]
        if isinstance(b, collections.Iterable):
            b = list(b)
        else:
            b = [b]
        # Check that 1) the number of contracted indices for self and
        # other match and 2) that the indices are compatible, i.e. okay
        # to contract with each other. In addition raise a warning if
        # the dirs don't match.
        assert(len(a) == len(b))
        assert(all(itertools.starmap(
            functools.partial(self.compatible_indices, other),
            zip(a, b))))
        if (self.dirs is not None and other.dirs is not None and
                not all(self.dirs[i] + other.dirs[j] == 0
                        for i,j in zip(a,b))):
            warnings.warn("dirs in dot do not match.")

        s_sum = a
        s_open = [i for i in range(len(self.shape)) if i not in a]
        o_sum = b
        o_open = [i for i in range(len(other.shape)) if i not in b]
        self, s_transposed_shape, s_transposed_qhape, s_transposed_dirs =\
                self.to_matrix(s_open, s_sum,
                               return_transposed_shape_data=True)
        other, o_transposed_shape, o_transposed_qhape, o_transposed_dirs =\
                other.to_matrix(o_sum, o_open,
                                return_transposed_shape_data=True)
        self = self.matrix_dot(other)
        del(other)
        l_dims = s_transposed_shape[:len(s_open)]
        r_dims = o_transposed_shape[len(o_sum):]
        try:
            l_qims = s_transposed_qhape[:len(s_open)]
            r_qims = o_transposed_qhape[len(o_sum):]
        except TypeError:
            l_qims = None
            r_qims = None
        try:
            l_dirs = s_transposed_dirs[:len(s_open)]
            r_dirs = o_transposed_dirs[len(o_sum):]
        except TypeError:
            l_dirs = None
            r_dirs = None
        self = self.from_matrix(l_dims, r_dims,
                                left_qims=l_qims, right_qims=r_qims,
                                left_dirs=l_dirs, right_dirs=r_dirs)
        return self


    def eig(self, a, b, chis=None, eps=0, print_errors=0,
            return_rel_err=False, hermitian=False, break_degenerate=False,
            degeneracy_eps=1e-6, norm_type="frobenius"):
        """ Transpose indices a to be on one side of self, b on the
        other, and reshape self to a matrix. Then find the eigenvalues
        and eigenvectors of this matrix, and reshape the eigenvectors to
        have on the left side the indices that self had on its right
        side after transposing but before reshaping.
        
        If hermitian is True then the matrix that is formed after the
        reshape is assumed to be hermitian. 

        Truncation works like with SVD.
        
        Output is S, U, [rel_err], where S is a vector of eigenvalues
        and U is a tensor such that the last index enumerates the
        eigenvectors of self in the sense that if u_i = U[...,i] then
        self.dot(u_i, (b, all_indices_of_u_i)) == S[i] * u_i. rel_err is
        relative error in truncation, only returned if return_rel_err is
        True.

        The above syntax is precisely correct only for Tensors. For
        AbelianTensors the idea is the same, but eigenvalues and vectors
        come with quantum numbers so the syntax is slightly different.
        See AbelianTensor.matrix_eig for more details about what
        precisely happens.

        The original tensor is not modified by this method.
        """
        if not isinstance(a, collections.Iterable):
            a = (a,)
        if not isinstance(b, collections.Iterable):
            b = (b,)
        self, transposed_shape, transposed_qhape, transposed_dirs\
                = self.to_matrix(a, b, return_transposed_shape_data=True)
        S, U, rel_err = self.matrix_eig(chis=chis, eps=eps,
                                        print_errors=print_errors,
                                        hermitian=hermitian,
                                        break_degenerate=break_degenerate,
                                        degeneracy_eps=degeneracy_eps,
                                        norm_type=norm_type)
        del(self)

        U_dims = (transposed_shape[:len(a)], S.shape)
        if transposed_qhape is not None:
            U_qims = (transposed_qhape[:len(a)], S.qhape)
        else:
            U_qims = (None, None)
        if transposed_dirs is not None:
            U_dirs = (transposed_dirs[:len(a)], U.dirs[1:])
        else:
            U_dirs = (None, None)
        U = U.from_matrix(*U_dims,
                          left_qims=U_qims[0], right_qims=U_qims[1],
                          left_dirs=U_dirs[0], right_dirs=U_dirs[1])
        ret_val = (S, U)
        if return_rel_err:
            ret_val += (rel_err,)
        return ret_val


    def svd(self, a, b, chis=None, eps=0, print_errors=0,
            return_rel_err=False, break_degenerate=False, degeneracy_eps=1e-6,
            norm_type="frobenius"):
        """ Transpose indices a to be on one side of self, b on the
        other, and reshape self to a matrix. Then singular value
        decompose this matrix into U, S, V. Finally reshape the unitary
        matrices to tensors that have a new index coming from the SVD,
        for U as the last index and for V as the first, and U to have
        indices a as its first indices and V to have indices b as its
        last indices.

        If eps>0 then the SVD may be truncated if the relative Frobenius
        norm error can be kept below eps.  For this purpose different
        dimensions to truncate to can be tried, and these dimensions
        should be listed in chis. If chis is None then the full range of
        possible dimensions is tried.

        If print_errors > 0 then the truncation error is printed.

        If return_rel_err is True then the relative truncation error is
        also returned.

        norm_type specifies the norm used to measure the error. This
        defaults to "frobenius". The other option is "trace", for trace
        norm.
        
        Output is U, S, V, and possibly rel_err.  Here S is a vector of
        singular values and U and V are isometric tensors (unitary if
        the matrix that is SVDed is square and there is no truncation).
        U . diag(S) . V = self, up to truncation errors.

        The original tensor is not modified by this method.
        """
        if not isinstance(a, collections.Iterable):
            a = (a,)
        if not isinstance(b, collections.Iterable):
            b = (b,)

        self, transposed_shape, transposed_qhape, transposed_dirs =\
                self.to_matrix(a, b, return_transposed_shape_data=True)

        U, S, V, rel_err = self.matrix_svd(chis=chis, eps=eps,
                                           print_errors=print_errors,
                                           break_degenerate=break_degenerate,
                                           degeneracy_eps=degeneracy_eps,
                                           norm_type=norm_type)
        del(self)
        U_dims = (transposed_shape[:len(a)], S.shape)
        V_dims = (S.shape, transposed_shape[len(a):])
        if transposed_qhape is not None:
            U_qims = (transposed_qhape[:len(a)], S.qhape)
            V_qims = (S.qhape, transposed_qhape[len(a):])
        else:
            U_qims = (None, None)
            V_qims = (None, None)
        if transposed_dirs is not None:
            U_dirs = (transposed_dirs[:len(a)], U.dirs[1:])
            V_dirs = (V.dirs[:1], transposed_dirs[len(a):])
        else:
            U_dirs = (None, None)
            V_dirs = (None, None)
        U = U.from_matrix(*U_dims,
                          left_qims=U_qims[0], right_qims=U_qims[1],
                          left_dirs=U_dirs[0], right_dirs=U_dirs[1])
        V = V.from_matrix(*V_dims,
                          left_qims=V_qims[0], right_qims=V_qims[1],
                          left_dirs=V_dirs[0], right_dirs=V_dirs[1])
        ret_val = (U, S, V)
        if return_rel_err:
            ret_val += (rel_err,)
        return ret_val


    def matrix_decomp_format_chis(self, chis, eps):
        """ A common function for formatting the truncation parameters
        of SVD and eig. This is meant to be called by the matrix_svd and
        matrix_eig functions of subclasses.
        """
        if chis is None:
            min_dim = min(type(self).flatten_dim(self.shape[0]),
                          type(self).flatten_dim(self.shape[1])) + 1
            if eps > 0:
                chis = tuple(range(min_dim))
            else:
                chis = [min_dim]
        else:
            try:
                chis = tuple(chis)
            except TypeError:
                chis = [chis]
            if eps == 0:
                chis = [max(chis)]
            else:
                chis = sorted(chis)
        return chis


    def split(self, a, b, chis=None, eps=0, print_errors=0,
              return_rel_err=False, return_sings=False,
              break_degenerate=False, degeneracy_eps=1e-6,
              norm_type="frobenius"):
        """ Split with SVD. Like SVD, but takes the square root of the
        singular values and multiplies both unitaries with it, so that
        the tensor is split into two parts. Values are returned as
        (US, {S}, SV, {rel_err}),
        where the ones in curly brackets are only returned if the
        corresponding arguments are True.
        """
        svd_result = self.svd(a, b, chis=chis, eps=eps,
                              print_errors=print_errors,
                              return_rel_err=return_rel_err,
                              break_degenerate=break_degenerate,
                              degeneracy_eps=degeneracy_eps,
                              norm_type=norm_type)
        U, S, V = svd_result[0:3]
        S_sqrt = S.sqrt()
        U = U.multiply_diag(S_sqrt, -1, direction="right")
        V = V.multiply_diag(S_sqrt, 0, direction="left")
        if return_sings:
            ret_val = U, S, V
        else:
            ret_val = U, V
        if return_rel_err:
            ret_val += (svd_result[3],)
        return ret_val

