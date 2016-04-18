import numpy as np
import collections
import itertools
import operator
from functools import reduce
from tensors.tensorcommon import TensorCommon


class Tensor(TensorCommon, np.ndarray):
    """ This class implements no new functionality beyond ndarrays, but
    simply provides ndarrays the same interface that is used by the
    symmetry preserving tensor classes. Tensors always have qhape == None,
    dirs == None and charge == 0.
    """

    def __new__(cls, shape, *args, qhape=None, charge=None, invar=None,
                dirs=None, **kwargs):
        shape = cls.flatten_shape(shape)
        res = np.ndarray(shape, *args, **kwargs).view(cls)
        return res

    @classmethod
    def initialize_with(cls, numpy_func, shape, *args,
                        qhape=None, charge=None, invar=None, dirs=None,
                        **kwargs):
        shape = cls.flatten_shape(shape)
        res = numpy_func(shape, *args, **kwargs).view(cls)
        return res

    @classmethod
    def eye(cls, dim, qim=None, qodulus=None, *args, **kwargs):
        dim = cls.flatten_dim(dim)
        res = np.eye(dim, *args, **kwargs).view(cls)
        return res

    def diag(self, **kwargs):
        res = np.diag(self).view(Tensor)
        return res

    # Every tensor object has the attributes qhape, dirs and charge just
    # to match the interface of AbelianTensor.
    qhape = None
    dirs = None
    charge = 0


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # To and from normal numpy arrays

    def to_ndarray(self, **kwargs):
        return np.asarray(self.copy())

    @classmethod
    def from_ndarray(cls, a, **kwargs):
        if isinstance(a, np.ndarray):
            res = a.copy().view(cls)
        else:
            res = np.array(a).view(cls)
        return res



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Methods for slicing, setting and getting elements

    fill = np.ndarray.fill



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Operator methods

    conjugate = np.ndarray.conj
    abs = np.ndarray.__abs__
    any = np.ndarray.any
    all = np.ndarray.all
    allclose = np.allclose

    def log(self):
        return np.log(self)

    def exp(self):
        return np.exp(self)

    def sqrt(self):
        return np.sqrt(self)

    def average(self):
        return np.average(self)

    def sign(self):
        return np.sign(self)

    def real(self):
        return super(Tensor, self).real

    def imag(self):
        return super(Tensor, self).imag



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    def compatible_indices(self, other, i, j):
        return self.shape[i] == other.shape[j]

    def flip_dir(self, axis):
        res = self.view()
        return res

    def expand_dims(self, axis, direction=1):
        res = np.expand_dims(self, axis)
        if not isinstance(res, Tensor):
            res = type(self).from_ndarray(res)
        return res

    def value(self):
        """ For a scalar tensor, return the scalar. """
        if self.shape:
            raise ValueError("value called on a non-scalar tensor.")
        else:
            return self[()]

    @classmethod
    def check_form_match(cls, tensor1=None, tensor2=None,
                         qhape1=None, shape1=None, dirs1=None,
                         qhape2=None, shape2=None, dirs2=None,
                         qodulus=None):
        """ Check that the given two tensors have the same form in the
        sense that if their legs are all flipped to point in the same
        direction then both tensors have the same qnums for the same
        indices and with the same dimensions. In stead of giving two
        tensors, sets of qhapes, shapes and dirs and a qodulus can also
        be given.
        """
        if tensor1 is not None:
            shape1 = tensor1.shape
        if tensor2 is not None:
            shape2 = tensor2.shape
        return shape1 == shape2

    @classmethod
    def find_trunc_dim(cls, S, chis=None, eps=0, break_degenerate=False,
                       degeneracy_eps=1e-6, norm_type="frobenius"):
        S = np.abs(S)
        if norm_type=="frobenius":
            S = S**2
            eps = eps**2
        elif norm_type=="trace":
            pass
        else:
            raise ValueError("Unknown norm_type {}".format(norm_type))
        sum_all = sum(S)
        # Find the smallest chi for which the error is small enough.
        # If none is found, use the largest chi.
        if sum_all != 0:
            for chi in chis:
                if not break_degenerate:
                    # Make sure that we don't break degenerate singular
                    # values by including one but not the other.
                    while 0 < chi < len(S):
                        last_in = S[chi-1]
                        last_out = S[chi]
                        rel_diff = np.abs(last_in-last_out)/last_in
                        if rel_diff < degeneracy_eps:
                            chi -= 1
                        else:
                            break
                sum_disc = sum(S[chi:])
                rel_err = sum_disc/sum_all
                if rel_err <= eps:
                    break
            if norm_type=="frobenius":
                rel_err = np.sqrt(rel_err)
        else:
            rel_err = 0
            chi = min(chis)
        return chi, rel_err

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def join_indices(self, *inds, return_transposed_shape_data=False,
                     **kwargs):
        """ Joins indices together in the spirit of reshape. inds is
        either a iterable of indices, in which case they are joined, or
        a iterable of iterables of indices, in which case the indices
        listed in each element of inds will be joined.
        
        Before any joining is done the indices are transposed so that
        for every batch of indices to be joined the first remains in
        place and the others are moved to be after in the order given.
        The order in which the batches are given does not matter.
        
        If return_transposed_shape_data is True, then the shape of the
        tensor after transposing but before reshaping is returned as
        well, in addition to None and None, that take the place of
        transposed_qhape and transposed_dirs of AbelianTensor.
        
        The method does not modify the original tensor.
        """
        # Format index_batches to be a list of lists of indices.
        if isinstance(inds[0], collections.Iterable):
            index_batches = list(map(list, inds))
        else:
            index_batches = [list(inds)]
        # Remove empty batches.
        index_batches = [b for b in index_batches if b]

        # Create the permutation p for transposing.
        joined = set(sum(index_batches, []))
        # Now we only need to insert the indices that are not joined.
        not_joined = [[i] for i in range(len(self.shape)) if i not in joined]
        all_in_batches = not_joined + index_batches
        all_in_batches.sort(key=operator.itemgetter(0))
        p = sum(all_in_batches, [])
        index_batches = [batch for batch in all_in_batches if len(batch) > 1]
        index_batches = [list(map(p.index, batch)) for batch in index_batches]
        self = self.transpose(p)

        if return_transposed_shape_data:
            transposed_shape = self.shape

        shp = list(self.shape)
        for batch in reversed(index_batches):
            new_dim = reduce(operator.mul, map(shp.__getitem__, batch))
            shp[batch[0]] = new_dim
            del shp[batch[1] : batch[0]+len(batch)]
        self = self.reshape(shp)

        if return_transposed_shape_data:
            return self, transposed_shape, None, None
        else:
            return self


    def split_indices(self, indices, dims, qims=None, **kwargs):
        """ Splits indices in the spirit of reshape. Indices is an
        iterable of indices to be split. Dims is an iterable of
        iterables such that dims[i]=dim_batch is an iterable of lists of
        dimensions, each list giving the dimensions along a new index
        that will come out of splitting indices[i].

        An example clarifies:
        Suppose self has shape [dim1, dim2, dim3, dim4]. Suppose then
        that indices = [1,3], dims = [[dimA, dimB], [dimC, dimD]].  Then
        the resulting tensor will have shape [dim1, dimA, dimB, dim3,
        dimC, dimD], assuming that that dims and are such that joining
        dimA and dimB gives qim2, etc.

        In stead of a list of indices a single index may be given.
        Correspondingly dims should then have one level of depth less as
        well.

        split_indices never modifies the original tensor.
        """
        # Formatting the input so that indices is a list and dim_batches
        # is a list of lists.
        if isinstance(indices, collections.Iterable):
            assert(len(indices) == len(dims))
            indices = list(indices)
            dim_batches = list(map(list, dims))
        else:
            indices = [indices]
            dim_batches = [list(dims)]
        dim_batches = [[type(self).flatten_dim(dim) for dim in batch]
                        for batch in dim_batches]

        if not indices:
            return self.view()

        # Sort indices and dim_batches according to reversed indices.
        # This is necessary for the next step to work.
        indices, dim_batches = zip(*sorted(zip(indices, dim_batches),
                                           reverse=True))
        # Compute the new shape.
        new_shape = list(self.shape)
        for ind, batch in zip(indices, dim_batches):
            new_shape[ind:ind+1] = batch

        # Reshape
        res = self.reshape(new_shape)
        return res

    
    def multiply_diag(self, diag_vect, axis, *args, **kwargs):
        """ Multiply self along axis with the diagonal matrix of
        components diag_vect.
        """
        if len(diag_vect.shape) != 1:
            raise ValueError("Second argument of multiply_diag must be a "
                             "vector.")
        if axis < 0:
            axis = len(self.shape) + axis
        res = self.swapaxes(-1, axis)
        res = res * diag_vect
        res = res.swapaxes(-1, axis)
        return res


    def trace(self, axis1=0, axis2=1):
        # We assert that the tensor is square with respect to axis1 and
        # axis2, to follow as closely as possible what AbelianTensor
        # does.
        assert(self.compatible_indices(self, axis1, axis2))
        trace = super(Tensor, self).trace(axis1=axis1, axis2=axis2)
        return type(self).from_ndarray(trace)


    def dot(self, B, indices):
        result = np.tensordot(self, B, indices)
        if not isinstance(result, Tensor):
            result = type(self).from_ndarray(result)
        return result


    def matrix_dot(self, B):
        result = np.dot(self, B)
        if not isinstance(result, TensorCommon):
            result = type(self).from_ndarray(result)
        return result

    def matrix_eig(self, chis=None, eps=0, print_errors=0, hermitian=False,
                   break_degenerate=False, degeneracy_eps=1e-6,
                   norm_type="frobenius"):
        chis = self.matrix_decomp_format_chis(chis, eps)
        if hermitian:
            S, U = np.linalg.eigh(self)
        else:
            S, U = np.linalg.eig(self)
        order = np.argsort(-np.abs(S))
        S = S[order]
        U = U[:,order]
        # Truncate, if truncation dimensions are given.
        chi, rel_err = type(self).find_trunc_dim(
            S, chis=chis, eps=eps, break_degenerate=break_degenerate,
            degeneracy_eps=degeneracy_eps, norm_type=norm_type)
        # Truncate
        S = S[:chi]
        U = U[:,:chi]
        if print_errors > 0:
            print('Relative truncation error in eig: %.3e' % rel_err)
        if not isinstance(S, TensorCommon):
            S = type(self).from_ndarray(S)
        if not isinstance(U, TensorCommon):
            U = type(self).from_ndarray(U)
        return S, U, rel_err

    def matrix_svd(self, chis=None, eps=0, print_errors=0,
                   break_degenerate=False, degeneracy_eps=1e-6,
                   norm_type="frobenius"):
        chis = self.matrix_decomp_format_chis(chis, eps)
        U, S, V = np.linalg.svd(self, full_matrices=False)
        S = Tensor.from_ndarray(S)
        # Truncate, if truncation dimensions are given.
        chi, rel_err = type(self).find_trunc_dim(
            S, chis=chis, eps=eps, break_degenerate=break_degenerate,
            degeneracy_eps=degeneracy_eps, norm_type=norm_type)
        # Truncate
        S = S[:chi]
        U = U[:,:chi]
        V = V[:chi,:]
        if print_errors > 0:
            print('Relative truncation error in SVD: %.3e' % rel_err)
        return U, S, V, rel_err


