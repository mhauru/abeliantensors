import numpy as np
import collections
import itertools
import heapq
import warnings
from functools import reduce
import operator
from copy import deepcopy
from tensors.tensorcommon import TensorCommon

def generate_binary_deferer(op_func):
    def deferer(self, B, *args, **kwargs):
        return type(self).defer_binary_elementwise(self, B, op_func, *args,
                                                   **kwargs)
    return deferer

def generate_unary_deferer(op_func):
    def deferer(self, *args, **kwargs):
        return type(self).defer_unary_elementwise(self, op_func, *args,
                                                  **kwargs)
    return deferer


class AbelianTensor(TensorCommon):
    """ A class for symmetry preserving tensors capabable of handling
    all abelian symmetry groups. Meant to be subclassed to implement
    specific symmetries, which can typically be done by simply fixing
    the qodulus (see below) of the class.
    
    Every AbelianTensor has the following attributes:

    shape: A list of dims, one dim per leg. Every dim is a list of
    integerers that are the dimensions of the different quantum number
    blocks along that leg.

    qhape: A list of qims, one qim per leg. Every qim is a list of
    unique integers that are the quantum numbers (qnums) of that leg.
    The quantum numbers are in one-to-one correspondence with the
    elements of the dims, so that qhape[i][j] and shape[i][j] are the
    qnum and dimension of the same block.

    dirs: A list of integers -1 or 1, one for each leg. 1 means that the
    corresponding leg is outgoing, -1 means incoming. 

    qodulus: An integer or None. If an integer, then all arithmetic on
    the quantum numbers is done modulo qodulus. If None then arithmetic
    on qnums is just usual integer arithmetic.

    sects: A dict of numpy arrays, with combinations of quantum numbers
    as keys. Every key must a tuple of quantum numbers, one for each
    leg, and each one of them being from the qim of that leg. The value
    of the dict at this key is the block (or "sector" or "sect")
    corresponding to these quantum numbers. If the tensor is invariant
    under a symmetry (see invar) then only certain blocks are allowed to
    be set, but even in such a cause not all allowed blocks must be set.
    For the treatement of unset blocks see defval.

    dtype: A numpy dtype, that is the dtype of all the sects.

    defval: The default value that the tensor has everywhere outside the
    blocks set in sects. If the tensor is a scalar with no legs then its
    value is its defval and it has no blocks. Note that many of the
    methods - such as dot and svd - require defval == 0 (and assert
    this). The main use of defval != 0 is to be able to handle tensors
    of boolean values that arise in comparisons.

    charge: An integer such that if invar is True then all the blocks
    set in sects must have keys k such that sum_i k[i]*dirs[i] % qodulus
    == charge.

    invar: A boolean. If True, then the tensor is invariant under the
    symmetry determined by qodulus, in the sense described in the
    definition of charge. If False, this condition is ignored and any
    block can be set. Note that as with defval, many methods require the
    tensor to be invariant and invar == False is mainly meant for
    handling vectors of singular values and eigenvalues. If invar ==
    True then defval must be 0, unless the tensor is a scalar of charge
    0.

    Note that many of these rules are not constantly checked for and can
    be broken by the user. In such cases behavior of the class is not
    quaranteed. The method check_consistency can be used to check that
    the tensor conforms to this definition.

    """

    def __init__(self, shape, qhape=None, qodulus=None, sects=None, dirs=None,
                 dtype=np.float_, defval=0, charge=0, invar=True):
        """ Although qhape is a keyword argument to conform
        to the interface of the Tensor class, it must in fact be set.
        dirs defaults to [1,1,...,1], sects defaults to {}.
        """
        assert(qhape is not None)
        if dirs is None:
            dirs = [1]*len(shape)
            warnings.warn("In __init__, dirs was not given and is thus "
                         "generated to be [[1,...,1], ..., [1,...,1]].")
        shape = list(map(list, shape))
        qhape = list(map(list, qhape))
        if qodulus is not None:
            charge %= qodulus
            qhape = [[q % qodulus for q in qim] for qim in qhape]
        assert(type(self).check_qhape_shape_match(qhape, shape))
        assert(len(shape) == len(dirs))
        if invar and charge != 0:
            assert(defval == 0)
        if sects is None:
            sects = {}

        self.defval = defval
        self.invar = invar
        self.charge = charge
        self.shape = list(map(list, shape))
        self.dirs = dirs
        self.dtype = dtype
        self.qhape = qhape
        self.qodulus = qodulus
        self.sects = sects


    copy = deepcopy
    __copy__ = copy


    def view(self):
        """ A view is otherwise independent but identical to the
        original, but its sects points to the same numpy arrays as the
        sects of the original. In other words changing a whole block is ok,
        but modifying a block in place modifies the original as well.
        """
        view = self.empty_like()
        # Note that this is a shallow copy.
        view.sects = self.sects.copy()
        return view


    def diag(self):
        """ Diag either maps a square matrix to a vector of its
        diagonals or a vector to diagonal square matrix.

        If the input is a vector (which may be non-invariant) with
        qhape=[qim], shape=[dim] and dir=[d], then the output is an
        invariant matrix with qhape=[qim,qim], shape=[dim,dim] and
        dirs[d,-d].

        If the input is a matrix it should be invariant and square in
        the sense that its two indices are compatible, i.e. could be
        contracted with each other. If self.dirs[d,d] then the latter is
        flipped and a warning is raised. The output is then a
        non-invariant vector with dirs=[d].
        """
        assert(len(self.shape) == 1 or len(self.shape) == 2)
        if len(self.shape) == 1:
            dim = self.shape[0]
            qim = self.qhape[0]
            shape = [dim, dim]
            qhape = [qim, qim]
            d = self.dirs[0]
            dirs = [d,-d]
            sects = {}
            for k,v in self.sects.items():
                new_k = (k[0], k[0])
                sects[new_k] = np.diag(v)
            res = type(self)(shape, qhape=qhape, qodulus=self.qodulus,
                             sects=sects, dirs=dirs, dtype=self.dtype)
            return res
        else:
            assert(self.invar)
            assert(self.compatible_indices(self, 0, 1))
            d = self.dirs[0]
            if self.dirs[1] + d != 0:
                warnings.warn("Automatically flipping dir 1 in diag.",
                              stacklevel=2)
                self = self.flip_dir(1)
            dim = self.shape[0]
            qim = self.qhape[0]
            shape = [dim]
            qhape = [qim]
            dirs = [d]
            sects = {}
            for qnum in qim:
                try:
                    diag_block = self[(qnum, qnum)]
                    sects[(qnum,)] = np.diag(diag_block)
                except KeyError:
                    # The diagonal block was not found, so we move on.
                    pass
            res = type(self)(shape, qhape=qhape, qodulus=self.qodulus,
                             sects=sects, dtype=self.dtype, dirs=dirs,
                             invar=False)
            return res


    @classmethod
    def eye(cls, dim, qim=None, qodulus=None, dtype=np.float_):
        """ Outputs an identity tensor of shape=[dim,dim],
        qhape=[qim,qim] and dirs[1,-1].
        """
        assert(cls.check_qim_dim_match(qim, dim))
        dim = list(dim)
        qim = list(qim)
        sects = {}
        for i, qnum in enumerate(qim):
            sects[qnum, qnum] = np.eye(dim[i], dtype=dtype)
        shape = [dim, dim]
        qhape = [qim, qim]
        dirs = [1,-1]
        res = cls(shape, qhape=qhape, qodulus=qodulus, sects=sects, dirs=dirs,
                  dtype=dtype)
        return res

        
    @classmethod
    def initialize_with(cls, numpy_func, shape, *args,
                        qhape=None, qodulus=None, dirs=None, invar=True,
                        charge=0, **kwargs):
        """ initialize_with will be called with different numpy_funcs to
        create initializer functions such as zeros and random. It sets
        all the valid blocks of the new tensor to
        numpy_func(block_shape, *args, **kwargs).
        """
        shape = list(map(list, shape))
        qhape = list(map(list, qhape))
        if qodulus is not None:
            charge %= qodulus
            qhape = [[q % qodulus for q in qim] for qim in qhape]
        assert(cls.check_qhape_shape_match(qhape, shape))
        assert(len(dirs) == len(shape))

        # We use a fancy way of passing optional arguments to __init__
        # to avoid setting defaults in two places.
        opt_args = {"qhape": qhape, "qodulus": qodulus, "charge": charge,
                    "invar": invar, "dirs": dirs}
        try:
            opt_args["dtype"] = kwargs["dtype"]
        except KeyError:
            pass
        res = cls(shape, **opt_args)

        dimcombs = itertools.product(*tuple(shape))
        qimcombs = itertools.product(*tuple(qhape))
        if shape:
            for qcomb, dcomb in zip(qimcombs, dimcombs):
                if res.is_valid_key(qcomb):
                    res[tuple(qcomb)] = numpy_func(dcomb, *args, **kwargs)
        else:
            if res.charge == 0:
                res.defval = numpy_func((), *args, **kwargs)
        return res


    def empty_like(self):
        """ Initializes a tensor that is like a copy of self, but with
        an empty sects.
        """
        res = type(self)(self.shape.copy(), qhape=self.qhape.copy(),
                         qodulus=self.qodulus, dtype=self.dtype,
                         defval=self.defval, invar=self.invar,
                         charge=self.charge, dirs=self.dirs.copy())
        return res



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Methods for slicing, setting and getting elements

    def fill(self, value):
        self.defval = value
        for v in self.sects.values():
            v.fill(value)

    def __getitem__(self, k):
        """ If self[k] is called, then first self.sects[k] is checked.
        If the key is not found, we check if the k still is a valid key
        for this tensor. If yes, a block full of defval is created, set
        to be self[k] and returned. If not, a KeyError is raised, with
        message describing what went wrong.
        """
        try:
            return self.sects.__getitem__(k)
        except KeyError:
            if not isinstance(k, tuple) or not len(k) == len(self.qhape):
                raise KeyError("Malformed block key: %s"%str(k))
            if self.is_valid_key(k):
                # Even though the requested block was not found it's a
                # valid block, so we create it.
                try:
                    block = self.defblock(k)
                except ValueError:
                    raise KeyError("Requested block has non-existent quantum "
                                   "numbers.")
                self[k] = block
                return block
            else:
                raise KeyError("Requested a block forbidden by symmetry: %s"
                               % str(k))

    def value(self):
        """ For a scalar tensor, return the scalar. """
        if self.shape:
            raise ValueError("value called on a non-scalar tensor.")
        else:
            return self.defval

    def __setitem__(self, key, value):
        return self.sects.__setitem__(key, value)

    def __delitem__(self, key):
        return self.sects.__delitem__(key)



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Operator methods

    def __repr__(self):
        r = ("%r(%r, qhape=%r, qodulus=%r, sects=%r, dtype=% rdefval=%r, "
             "invar=%r, charge=%r, dirs=%r"%(
                 type(self), self.shape, self.qhape, self.qodulus, self.sects,
                 self.dtype, self.defval, self.invar, self.charge, self.dirs))
        return r

    def __str__(self, *args, **kwargs):
        r = ("%s object:\n"
             "shape = %s,\n"
             "qhape = %s, qodulus = %s, charge = %s\n"
             "dirs = %s,\n"
             "defval = %s, invar = %s, dtype = %s\n"
             "blocks:")%(str(type(self)), self.shape,
                           self.qhape, self.qodulus, self.charge, 
                           self.dirs,
                           self.defval, self.invar, self.dtype)
        for k,v in self.sects.items():
            r += "\n%s:\n%s"%(k,v)
        return r

    __add__ = generate_binary_deferer(operator.add)
    __sub__ = generate_binary_deferer(operator.sub)
    __mul__ = generate_binary_deferer(operator.mul)
    __truediv__ = generate_binary_deferer(operator.truediv)
    __floordiv__ = generate_binary_deferer(operator.floordiv)
    __mod__ = generate_binary_deferer(operator.mod)
    __divmod__ = generate_binary_deferer(divmod)
    __pow__ = generate_binary_deferer(pow)
    __lshift__ = generate_binary_deferer(operator.lshift)
    __rshift__ = generate_binary_deferer(operator.rshift)
    __and__ = generate_binary_deferer(operator.and_)
    __xor__ = generate_binary_deferer(operator.xor)
    __or__ = generate_binary_deferer(operator.or_)

    def arg_swapper(op):
        def res(a,b, *args, **kwargs):
            return op(b,a, *args, **kwargs)
        return res

    __radd__ = generate_binary_deferer(arg_swapper(operator.add))
    __rsub__ = generate_binary_deferer(arg_swapper(operator.sub))
    __rmul__ = generate_binary_deferer(arg_swapper(operator.mul))
    __rtruediv__ = generate_binary_deferer(arg_swapper(operator.truediv))
    __rfloordiv__ = generate_binary_deferer(arg_swapper(operator.floordiv))
    __rmod__ = generate_binary_deferer(arg_swapper(operator.mod))
    __rdivmod__ = generate_binary_deferer(arg_swapper(divmod))
    __rpow__ = generate_binary_deferer(arg_swapper(pow))
    __rlshift__ = generate_binary_deferer(arg_swapper(operator.lshift))
    __rrshift__ = generate_binary_deferer(arg_swapper(operator.rshift))
    __rand__ = generate_binary_deferer(arg_swapper(operator.and_))
    __rxor__ = generate_binary_deferer(arg_swapper(operator.xor))
    __ror__ = generate_binary_deferer(arg_swapper(operator.or_))

    __eq__ = generate_binary_deferer(operator.eq)
    __ne__ = generate_binary_deferer(operator.ne)
    __lt__ = generate_binary_deferer(operator.lt)
    __le__ = generate_binary_deferer(operator.le)
    __gt__ = generate_binary_deferer(operator.gt)
    __ge__ = generate_binary_deferer(operator.ge)

    __neg__ = generate_unary_deferer(operator.neg)
    __pos__ = generate_unary_deferer(operator.pos)
    __abs__ = generate_unary_deferer(abs)
    __invert__ = generate_unary_deferer(operator.invert)

    def conj(self):
        """ Conjugation complex conjugates, flips the directions of
        all the indices and flips the sign of the charge.
        """
        res = self.defer_unary_elementwise(np.conj)
        res.dirs = list(map(operator.neg, res.dirs))
        res.charge = -res.charge
        if self.qodulus is not None:
            res.charge %= res.qodulus
        return res


    def astype(self, dtype, casting='unsafe', copy=True):
        if not np.can_cast(self.dtype, dtype, casting=casting):
            raise ValueError("Cannot cast {} into {} with casting={}.".
                             format(self.dtype, dtype, casting))
        if copy:
            res = self.copy()
        else:
            res = self
        res.dtype = dtype
        for k, v in res.sects.items():
            res[k] = v.astype(dtype, casting=casting, subok=True, copy=False)
        return res


    conjugate = conj
    sqrt = generate_unary_deferer(np.sqrt)
    sign = generate_unary_deferer(np.sign)
    log = generate_unary_deferer(np.log)
    exp = generate_unary_deferer(np.exp)
    abs = __abs__


    def defer_unary_elementwise(self, op_func, *args, **kwargs):
        """ Produces a new tensor that is like self, but all the blocks
        v have been acted on with op_func(v, *args, **kwargs), as has
        the defval. If defval ends up being mapped to something non-zero
        then the resulting tensor is not invariant and is flagged as
        such.

        This method can be used to create basic element-wise unary
        operations on tensors, such as negation and element-wise
        absolute value.
        """
        res = self.empty_like()
        res.defval = op_func(self.defval, *args, **kwargs)
        if res.defval != 0:
            res.invar = False
        for k,v in self.sects.items():
            res_block = op_func(v, *args, **kwargs)
            res.sects[k] = res_block
        return res


    def defer_binary_elementwise(self, B, op_func, *args, **kwargs):
        """ If both self and B are AbelianTensors, then their blocks and
        defvals are operated on pair-wise with op_func(_, _, *args,
        **kwargs). The two tensors should in this case be of the same
        form: same qnums, dims, etc. If not, either warnings or errors
        are raised, depending on whether the operation can still be
        carried out meaningfully.
        
        If B is not an AbelianTensor then all the blocks and
        the defval of self are operated on with op_func(_, B, *args,
        **kwargs).

        The operation is never in-place, but always a new tensor. The
        new tensor is like self in its attributes, but may be
        non-invariant if its defval ends up being non-zero.

        This method can be used to create element-wise binary operations
        on tensors, such as basic arithmetic and comparisons.
        """
        try:
            res_dtype = np.result_type(self.dtype, B.dtype)
        except AttributeError:
            res_dtype = np.result_type(self.dtype, B)
        res = self.empty_like()
        res.dtype = res_dtype
        if isinstance(B, AbelianTensor):
            # self and B should be of the same form or one should be a
            # scalar. They should also have the same qodulus.
            assert(type(self).check_form_match(tensor1=self, tensor2=B)
                    or not self.shape or not B.shape)
            assert(self.qodulus == B.qodulus)
            # They may have different charges and dirs, but this
            # generates a warning.
            if self.charge != B.charge and self.shape:
                warnings.warn("Binary operation called on non-scalar tensors "
                              "with differing charges (%i and %i)."
                              %(self.charge, B.charge), stacklevel=3)
            relative_dirs = tuple(map(operator.mul, self.dirs, B.dirs))
            for i, d in enumerate(relative_dirs):
                if d != 1:
                    warnings.warn("Automatically flipping dir %i in binary "
                                  "operation."%i, stacklevel=3)
                    B = B.flip_dir(i)

            # Checks are done, move on to operating.
            res.defval = op_func(self.defval, B.defval, *args, **kwargs)
            all_keys = set().union(self.sects, B.sects)
            for k in all_keys:
                # Use B[k] and self[k], but default to defval if key not
                # found.
                a = self.sects.get(k, self.defval)
                b = B.sects.get(k, B.defval)
                res_block = op_func(a, b, *args, **kwargs)
                res.sects[k] = res_block
        else:
            res.defval = op_func(self.defval, B, *args, **kwargs)
            for k,v in self.sects.items():
                res_block = op_func(v, B, *args, **kwargs)
                res.sects[k] = res_block
        if (res.shape or res.charge) and res.defval != 0:
            res.invar = False
        return res


    def any(self):
        """ Check whether any of the elements of the tensor are True.
        """
        for v in self.sects.values():
            if np.any(v):
                return True
        if self.is_full():
            return False
        else:
            return np.any(self.defval)


    def all(self):
        """ Check whether all of the elements of the tensor are True.
        """
        for v in self.sects.values():
            if not np.all(v):
                return False
        if self.is_full():
            return True
        else:
            return np.all(self.defval)


    def allclose(self, B, rtol=1e-05, atol=1e-08):
        """ Check whether all of the elements of the tensors are close
        to each other. See numpy.allclose for explanations of the
        tolerance arguments.
        """
        # self and B should be of the same form and have the same
        # qodulus.
        assert(type(self).check_form_match(tensor1=self, tensor2=B))
        assert(self.qodulus == B.qodulus)
        # They may have different dirs, but this generates a warning.
        relative_dirs = tuple(map(operator.mul, self.dirs, B.dirs))
        for i, d in enumerate(relative_dirs):
            if d != 1:
                warnings.warn("Automatically flipping dir %i in binary "
                              "operation."%i, stacklevel=3)
                B = B.flip_dir(i)

        # Form checks done, move on to comparing blocks.
        all_keys = set().union(self.sects, B.sects)
        for k in all_keys:
            a = self.sects.get(k, self.defval)
            b = B.sects.get(k, B.defval)
            if not np.allclose(a,b):
                return False
        if self.is_full():
            return True
        else:
            return np.allclose(self.defval, B.defval)


    def max(self):
        if 0 in type(self).flatten_shape(self.shape):
            raise ValueError("zero-size array has no maximum")
        if not self.shape:
            return self.defval
        # If not all blocks are set, then the tensor has an element of
        # defval somewhere.
        m = -np.inf if self.is_full() else self.defval
        for v in self.sects.values():
            try:
                n = np.max(v)
            except ValueError:
                # This block was zero-size.
                n = m
            m = max(m, n)
        return m


    def min(self):
        if 0 in type(self).flatten_shape(self.shape):
            raise ValueError("zero-size array has no maximum")
        if not self.shape:
            return self.defval
        # If not all blocks are set, then the tensor has an element of
        # defval somewhere.
        m = np.inf if self.is_full() else self.defval
        for v in self.sects.values():
            try:
                n = np.min(v)
            except ValueError:
                # This block was zero-size.
                n = m
            m = min(m, n)
        return m


    def average(self):
        s = self.sum()
        flat_shape = self.flatten_shape(self.shape)
        num_of_elements = reduce(operator.mul, flat_shape, 1)
        average = s/num_of_elements
        return average


    def real(self):
        res = self.defer_unary_elementwise(np.real)
        res.dtype = np.float_
        return res


    def imag(self):
        res = self.defer_unary_elementwise(np.imag)
        res.dtype = np.float_
        return res


    def sum(self):
        if self.shape:
            if self.defval:
                raise NotImplementedError("Sum of defval != 0 not "
                                          "implemented.")
            s = 0
            for v in self.sects.values():
                s += np.sum(v)
        else:
            s = self.defval
        return s


    def __len__(self):
        """ This works as for numpy arrays: len returns the total
        dimension of the first leg.
        """
        return self.flatten_dim(self.shape[0])


    def __bool__(self):
        if self.shape:
            raise ValueError("The truth value of a tensor with more than one "
                             "element is ambiguous. Use a.any() or a.all()")
        else:
            return bool(self.defval)



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # To and from normal numpy arrays

    def to_ndarray(self):
        """ Returns a corresponding numpy array. The order of the blocks
        in the result is such that along every index the blocks are
        organized according to rising qnum.
        
        Note that this means that the end result changes if the
        directions of some of the legs are flipped before calling
        to_ndarray(). Thus if for example trace or dot is called on the
        resulting numpy array, the result may be different than for the
        AbelianTensor if the contraction requires flipping directions.
        Similarly taking for example traces and diags along axes that
        were not compatible in the AbelianTensor is a perfectly valid
        thing to do for the ndarray, and gives different results.
        
        All these concerns can be avoided by making sure that one only
        calls on the ndarray operations that would have been valid on
        the AbelianTensor without flipping any directions.
        """
        ndshape = type(self).flatten_shape(self.shape)
        res = np.full(ndshape, self.defval, dtype=self.dtype)
        if 0 in ndshape:
            return res
        shp, qhp = type(self).sorted_shape_qhape(tensor=self)
        # ranges is like shape, but every number d is replaced by a
        # tuple (a,a+d) where a is the sum of all the previous entries
        # in the same dim.
        ranges = []
        for dim in shp:
            prv = dim[0]
            r = [(0, prv)]
            for d in dim[1:]:
                nxt = prv + d
                r.append((prv, nxt))
                prv = nxt
            ranges.append(r)
        for k, v in self.sects.items():
            slc = ()
            for i, qnum in enumerate(k):
                r = ranges[i][qhp[i].index(qnum)]
                slc += (slice(r[0], r[1]),)
            res[slc] = v
        return res


    @classmethod
    def from_ndarray(cls, a, shape=None, qhape=None, dirs=None, qodulus=None,
                     invar=True, charge=0):
        """ Takes an ndarray a, and maps it to a corresponding
        AbelianTensor of the form given in the other arguments. Although
        shape and qhape are keyword arguments to maintain a common
        interface with Tensor, they are not optional. The blocks are
        read in the same order as they are written in to_ndarray, i.e.
        rising qnum along every leg. Note hence that the ordering of the
        qnums in the qhape given has no effect.
        """
        if dirs is None:
            warnings.warn("In from_ndarray, dirs was not given and is thus "
                         "generated to be [1,...,1].")
            dirs = [1]*len(shape)
        is_bool = a.dtype == np.bool_
        invar = invar and not is_bool
        shape, qhape = cls.sorted_shape_qhape(shape=shape, qhape=qhape,
                                              qodulus=qodulus)
        res = cls(shape, qhape=qhape, qodulus=qodulus, dtype=a.dtype,
                  invar=invar, charge=charge, dirs=dirs)
        if not a.shape:
            res.defval = a
            return res
        # ranges is like shape, but every number d is replaced by a
        # tuple (a,a+d) where a is the sum of all the previous entries
        # in the same dim.
        ranges = []
        for dim in res.shape:
            prv = dim[0]
            r = [(0, prv)]
            for d in dim[1:]:
                nxt = prv + d
                r.append((prv, nxt))
                prv = nxt
            ranges.append(r)
        for k in itertools.product(*res.qhape):
            if res.is_valid_key(k):
                slc = ()
                for i, qnum in enumerate(k):
                    r = ranges[i][res.qhape[i].index(qnum)]
                    slc += (slice(r[0], r[1]),)
                block = a[slc]
                res.sects[k] = block
        return res



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    def is_valid_key(self, key):
        """ Returns true if self.invar is not True or key is a key for a
        valid block allowed by symmetry. Otherwise False.
        """
        if not self.invar:
            return True
        if len(key) != len(self.qhape):
            return False
        key = map(operator.mul, self.dirs, key)
        s = sum(key)
        if self.qodulus is not None:
            s %= self.qodulus
        return s == self.charge


    def compatible_indices(self, other, i, j):
        """ Returns True if leg i of self may be contracted with leg j
        of other, False otherwise. Flipping of indices is allowed (but
        not done, this is only a check).
        """
        s_d = self.dirs[i]
        s_dim = self.shape[i] 
        s_qim = self.qhape[i]
        o_d = other.dirs[j]
        o_dim = other.shape[j] 
        o_qim = [-1 * s_d * o_d * q for q in other.qhape[j]]
        if other.qodulus is not None:
            o_qim = [q % other.qodulus for q in o_qim]
        o_qimdim = set(zip(o_qim, o_dim))
        s_qimdim = set(zip(s_qim, s_dim))
        res = o_qimdim == s_qimdim
        return res
    

    def flip_dir(self, axis):
        """ Flips the direction of the given axis of self. The operation
        is not in-place, but a view is returned. The quantum numbers
        along given axis are also negated accordingly.
        """
        res = self.empty_like()
        res.dirs[axis] *= -1
        if self.qodulus is None:
            qod_func = lambda q: q
        else:
            qod_func = lambda q: q % self.qodulus
        res.qhape[axis] = [qod_func(-q) for q in res.qhape[axis]]
        keys = set(self.sects.keys())
        while keys:
            k = keys.pop()
            kf = list(k)
            kf[axis] = qod_func(-kf[axis])
            kf = tuple(kf)
            vf = self[k]
            if kf in keys:
                v = self[kf]
                keys.discard(kf)
                res[k] = v
            res[kf] = vf
        return res


    def expand_dims(self, axis, direction=1):
        """ Returns a view of self that has an additional leg at the
        position axis. This leg has only one qnum, 0, and dimension 1.
        The direction of the new leg is a keyword argument 'direction'
        that defaults to 1.
        """
        res = self.empty_like()
        res.shape.insert(axis, [1])
        res.qhape.insert(axis, [0])
        res.dirs.insert(axis, direction)
        if self.shape:
            for k,v in self.sects.items():
                new_k = list(k)
                new_k.insert(axis, 0)
                res[tuple(new_k)] = np.expand_dims(v, axis)
        elif res.charge == 0:
            res[(0,)] = np.array((res.defval,), dtype=res.dtype)
            res.defval = 0
        return res


    @staticmethod
    def sorted_shape_qhape(tensor=None, shape=None, qhape=None, qodulus=[]):
        """ Sort shape and qhape according to ascending qnum along every
        leg. Used by to_ and from_ndarray.
        """
        shape = tensor.shape if shape is None else shape
        qhape = tensor.qhape if qhape is None else qhape
        qodulus = tensor.qodulus if qodulus == [] else qodulus
        sorted_qhp = []
        sorted_shp = []
        for qim, dim in zip(qhape, shape):
            qim, dim = zip(*sorted(zip(qim,dim)))
            sorted_qhp.append(qim)
            sorted_shp.append(dim)
        return sorted_shp, sorted_qhp


    def defblock(self, key):
        """ Returns an ndarray of the size of the block self[key],
        filled with self.defval.  This works regardless of whether
        self[key] is set or not and whether the block is allowed by
        symmetry.
        """
        block_shape = []
        for i,qnum in enumerate(key):
            block_shape.append(self.shape[i][self.qhape[i].index(qnum)])
        block = np.full(block_shape, self.defval, dtype=self.dtype)
        return block


    def is_full(self):
        """ Returns True if the elements in self.sects cover all the
        elements in self.
        """
        elements_in_sects = sum(map(operator.attrgetter("size"),
                                    self.sects.values()))
        elements_in_total = reduce(operator.mul,
                                   type(self).flatten_shape(self.shape),
                                   1)
        res = elements_in_sects >= elements_in_total
        return res


    def check_consistency(self):
        """ Checks that self conforms to the defition given in the
        documentation of the class. If yes, returns True, otherwise
        raises an assertion error.  This method is meant to be used by
        the user (probably for debugging) and is not called anywhere in
        the class.
        """
        try:
            assert(len(self.shape) == len(self.qhape) == len(self.dirs))
            # Qnums must be unique within a qim and correspond
            # one-to-one with dimensions in dim.
            assert(all((len(dim) == len(qim) == len(set(qim))
                   for dim,qim in zip(self.shape, self.qhape))))
            assert(all(d == 1 or d == -1 for d in self.dirs))
            if self.qodulus is not None:
                assert(all(q == q % self.qodulus for q in sum(self.qhape, [])))
            # Check that every sect has a valid key and the correct
            # shape and dtype.
            for k,v in self.sects.items():
                assert(v.dtype == self.dtype)
                assert(self.is_valid_key(k))
                block_shp_real = v.shape
                qnum_inds = tuple(self.qhape[i].index(qnum)
                                  for i, qnum in enumerate(k))
                block_shp_claimed = tuple([self.shape[i][j]
                                          for i, j in enumerate(qnum_inds)])
                assert(block_shp_claimed == block_shp_real)
            # Other checks.
            if self.invar and (self.charge != 0 or self.shape):
                assert(self.defval == 0)
        except:
            raise
        return True

    
    @classmethod
    def check_qim_dim_match(cls, qim, dim):
        """ Check that the given qim and dim match, i.e. are valid for
        the same leg.
        """
        return len(qim) == len(dim)

    @classmethod
    def check_qhape_shape_match(cls, qhape, shape):
        """ Check that the given qhape and shape match, i.e. are valid
        for the same tensor.
        """
        return all(cls.check_qim_dim_match(qim, dim)
                   for qim,dim in zip(qhape,shape))

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
            qhape1 = tensor1.qhape
            shape1 = tensor1.shape
            dirs1 = tensor1.dirs
        if tensor2 is not None:
            qhape2 = tensor2.qhape
            shape2 = tensor2.shape
            dirs2 = tensor2.dirs
        if not (len(qhape1) == len(qhape2) == len(shape1) == len(shape2)
                == len(dirs1) == len(dirs2)):
            return False
        for d1, qim1, dim1, d2, qim2, dim2 in zip(dirs1, qhape1, shape1,
                                                  dirs2, qhape2, shape2):
            # This is almost like compatible_indices, but for the
            # missing minus sign when building o_qim.
            qim2 = [d1 * d2 * q for q in qim2]
            if qodulus is not None:
                qim2 = [q % qodulus for q in qim2]
            qimdim1 = set(zip(qim1, dim1))
            qimdim2 = set(zip(qim2, dim2))
            if not qimdim1 == qimdim2:
                return False
        return True

    @classmethod
    def find_trunc_dim(cls, S, S_sects, minusabs_next_els, dims,
                       chis=None, eps=0, break_degenerate=False,
                       degeneracy_eps=1e-6, norm_type="frobenius"):
        # First, find what chi will be.
        S = -np.sort(-np.abs(S))
        if norm_type=="frobenius":
            S = S**2
            eps = eps**2
        elif norm_type=="trace":
            pass
        else:
            raise ValueError("Unknown norm_type {}".format(norm_type))
        sum_all = np.sum(S)
        if sum_all != 0:
            for chi in chis:
                if not break_degenerate:
                    # Make sure that we don't break degenerate singular
                    # values by including one but not the other.
                    while 0 < chi < len(S):
                        last_eig_in = S[chi-1]
                        last_eig_out = S[chi]
                        rel_diff = np.abs(last_eig_in-last_eig_out)/last_eig_in
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
        # Find out which eigenvalues to keep.
        dim_sum = 0
        while(dim_sum < chi):
            # index_to_add is the index of the block which has
            # the largest eigenvalue that is not yet
            # included in truncation.
            try:
                minusabs_el_to_add, key = heapq.heappop(minusabs_next_els)
            except IndexError:
                # All the dimensions are fully included.
                break
            dims[key] += 1
            this_key_els = S_sects[key][0]
            if dims[key] < len(this_key_els):
                next_el = this_key_els[dims[key]]
                heapq.heappush(minusabs_next_els, (-np.abs(next_el), key))
            dim_sum += 1
        return chi, dims, rel_err

        


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def join_indices(self, *inds, dirs=None,
                     return_transposed_shape_data=False):
        """ Joins indices together in the spirit of reshape. inds is
        either a iterable of indices, in which case they are joined, or
        a iterable of iterables of indices, in which case the indices
        listed in each element of inds will be joined.
        
        Before any joining is done the indices are transposed so that
        for every batch of indices to be joined the first remains in
        place and the others are moved to be after in the order given.
        The order in which the batches are given does not matter.
        
        dirs are the directions of the new indices, defaults to
        [1,...,1]. If a batch of indices to be joined consists of only
        one index, its direction will be flipped to be as in dirs.
        
        If return_transposed_shape_data is True, then the shape, qhape
        and dirs (in this order) of the tensor after transposing but
        before reshaping are returned as well.
        
        The method does not modify the original tensor.
        """
        # Format index_batches to be a list of lists of indices.
        if isinstance(inds[0], collections.Iterable):
            index_batches = list(map(list, inds))
        else:
            index_batches = [list(inds)]
        # Remove empty batches.
        index_batches = [b for b in index_batches if len(b)>0]

        if dirs is None:
            warnings.warn("In join_indices, dirs was not given and is thus "
                         "generated to be [1,...,1].")
            dirs = [1]*len(index_batches)
        else:
            if not isinstance(dirs, collections.Iterable):
                dirs = [dirs]
            assert(len(dirs) == len(index_batches))

        if not index_batches:
            # Nothing to be done. However, join_indices should always
            # return an array independent of the original, so we take a
            # view.
            if return_transposed_shape_data:
                return (self.view(), self.shape.copy(), self.qhape.copy(),
                        self.dirs.copy())
            else:
                return self.view()

        # Group dirs together with index_batches so that they get sorted
        # together.
        index_batches_with_dirs = [b + [d] for b,d in zip(index_batches, dirs)]

        # Create the permutation for transposing the tensor. At the same
        # time transpose and sort index_batches.
        joined = set(sum(index_batches, []))
        not_joined = [[i] for i in range(len(self.shape)) if i not in joined]
        all_batches = not_joined + index_batches_with_dirs
        all_batches.sort(key=operator.itemgetter(0))
        # The a[:-1] conditional statement leaves out the dirs when
        # creating the permutation.
        p = sum((a[:-1] if len(a)>1 else a for a in all_batches), [])
        index_batches_with_dirs = [b for b in all_batches if len(b) > 1]
        dirs = [b[-1] for b in index_batches_with_dirs]
        index_batches = [b[:-1] for b in index_batches_with_dirs]
        index_batches = [list(map(p.index, b)) for b in index_batches]
        res = self.transpose(p)

        if return_transposed_shape_data:
            transposed_shape = res.shape.copy()
            transposed_qhape = res.qhape.copy()
            transposed_dirs = res.dirs.copy()

        # Flip the dirs on single indices in index_batches and remove
        # those batches from the list.
        for i,b in reversed(tuple(enumerate(index_batches))):
            if len(b) == 1:
                if res.dirs[b[0]] != dirs[i]:
                    res = res.flip_dir(b[0])
                del(dirs[i])
                del(index_batches[i])

        if not index_batches:
            # If no indices are left, there is no need to join anything.
            if return_transposed_shape_data:
                return res, transposed_shape, transposed_qhape, transposed_dirs
            else:
                return res
        
        # Find out the remaining, new indices after the joining.
        cumulant = 0
        new_inds = []
        for b in index_batches:
            new_inds.append(b[0] - cumulant)
            cumulant += len(b) - 1

        # Reverse index_batches and dirs for the future so that we first
        # process the indices at the end.
        index_batches.reverse()
        dirs.reverse()

        # For every non-zero block in res, reshape the block and add it
        # to the right key in new_sects. However, every item in
        # new_sects will consist of several blocks that need to be
        # concatenated. Because of this, new_sects[k] is a list of
        # lists [k_part1, k_part2, ..., k_partn, reshaped_block],
        # where k_parts are the qnums of the indices that were joined.
        # Thus by later sorting these lists we get them in the right
        # order for concatenation.
        new_sects = {}
        if res.qodulus is None:
            qod_func = lambda x: x
        else:
            qod_func = lambda x: x % res.qodulus
        # Go through every valid index in stead of every key in sects,
        # because blocks of zeros may be concatenated with other blocks.
        valid_ks = (qcomb for qcomb in itertools.product(*res.qhape)
                    if res.is_valid_key(qcomb))
        del_slcs = [slice(b[1], b[-1]+1) for b in index_batches]
        get_slcs = [slice(b[0], b[-1]+1) for b in index_batches]
        dir_batches = [[res.dirs[i] for i in batch] for batch in index_batches]
        for k in valid_ks:
            v = res[k]
            new_k = list(k)
            new_shp = list(v.shape)
            k_parts = []
            for b, dir_b, dir_new, del_slc, get_slc in\
                    zip(index_batches, dir_batches, dirs, del_slcs, get_slcs):
                k_part = k[get_slc]
                k_parts.append(k_part)
                k_part = map(operator.mul, k_part, dir_b)
                new_qnum = qod_func(sum(k_part) * dir_new)
                new_k[b[0]] = new_qnum
                del(new_k[del_slc])
                new_shp[b[0]] = reduce(operator.mul, v.shape[get_slc])
                del(new_shp[del_slc])
            k_parts.reverse()
            new_k = tuple(new_k)
            l = new_sects.setdefault(new_k, [])
            l.append(k_parts + [v.reshape(new_shp)])

        # Concatenator is a helper function that recursively
        # concatenates the pieces together. It is called once for every
        # index in a batch.
        def concatenator(l, i=0):
            if i == len(l[0])-2:
                l = [el[-1] for el in l]
            else:
                l = [tuple(g)
                     for k,g in itertools.groupby(l, operator.itemgetter(i))]
                l = tuple(map(lambda k: concatenator(k, i=i+1), l))
            return np.concatenate(l, new_inds[i])

        for k,v in new_sects.items():
            # These are the new blocks, just need to concatenate.
            v.sort()
            new_sects[k] = concatenator(v)
        res.sects = new_sects

        # Compute the new shape, qhape and dir.
        for new_d, batch in zip(dirs, index_batches):
            product_of_tuple = lambda l: reduce(operator.mul, l)
            cart_prod_of_dims = itertools.product(*tuple(res.shape[i]
                                                         for i in batch))
            new_dim = list(map(product_of_tuple, cart_prod_of_dims))

            qhps = ([q * res.dirs[i] for q in res.qhape[i]] for i in batch)
            cartesian_product_of_qims = itertools.product(*tuple(qhps))
            new_qim = map(sum, cartesian_product_of_qims)
            new_qim = (q*new_d for q in new_qim)
            new_qim = list(map(qod_func, new_qim))

            # Still need to concatenate.
            # Sort by new_qim.
            if new_qim:
                new_qim, new_dim = zip(*sorted(zip(new_qim, new_dim)))
                new_qim, new_dim = list(new_qim), list(new_dim)
                n = 0
                q = new_qim[n]
                i = 1
                while i < len(new_qim):
                    if new_qim[i] == q:
                        new_dim[n] += new_dim[i]
                        del(new_qim[i])
                        del(new_dim[i])
                    else:
                        n = i
                        q = new_qim[n]
                        i +=1 

            res.shape[batch[0]] = new_dim
            del res.shape[batch[1] : batch[0] + len(batch)]
            res.qhape[batch[0]] = new_qim
            del res.qhape[batch[1] : batch[0] + len(batch)]
            res.dirs[batch[0]] = new_d
            del res.dirs[batch[1] : batch[0] + len(batch)]

        if return_transposed_shape_data:
            return res, transposed_shape, transposed_qhape, transposed_dirs
        else:
            return res


    def split_indices(self, indices, dims, qims=None, dirs=None):
        """ Splits indices in the spirit of reshape. Indices is an
        iterable of indices to be split. Dims is an iterable of
        iterables such that dims[i]=dim_batch is an iterable of lists of
        dimensions, each list giving the dimensions along a new index
        that will come out of splitting indices[i]. qims correspondingly
        gives the qims of the new indices, and dirs gives the new
        directions.

        An example clarifies:
        Suppose self has shape [dim1, dim2, dim3, dim4], qhape [qim1,
        qim2, qim3, qim4] and dirs [d1,d2,d3,d4]. Suppose then that
        indices = [1,3], dims = [[dimA, dimB], [dimC, dimD]], qims =
        [[qimA, qimB], [qimC, qimD]] and dirs = [[dA, dB] [dC, dD]].
        Then the resulting tensor will have shape [dim1, dimA, dimB,
        dim3, dimC, dimD], qhape [qim1, qimA, qimB, qim3, qimC, qimD]
        and dirs [d1, dA, dB, d3, dC, dD].  All this assuming that that dims
        and qims are such that joining qimA and qimB gives qim2, etc.

        In stead of a list of indices a single index may be given.
        Correspondingly dims, qims and dirs should then have one level
        of depth less as well.

        split_indices never modifies the original tensor.
        """
        # Formatting the input so that indices is a list and dim_batches
        # and dim_batches are lists of lists.
        if isinstance(indices, collections.Iterable):
            assert(len(indices) == len(dims) == len(qims))
            indices = list(indices)
            dim_batches = list(map(list, dims))
            qim_batches = list(map(list, qims))
            if dirs is None:
                warnings.warn("In split_indices, dirs was not given and is "
                             "thus generated to be full of ones.")
                dir_batches = list(map(lambda dim: [1]*len(dim), dims))
            else:
                dir_batches = dirs
        else:
            indices = [indices]
            dim_batches = [list(dims)]
            qim_batches = [list(qims)]
            if dirs is None:
                warnings.warn("In split_indices, dirs was not given and is "
                             "thus generated to be [[1,...,1]].")
                dir_batches = [[1]*len(dims)]
            else:
                dir_batches = [dirs]

        if not indices:
            return self.view()

        # Reverse sort according to indices, so that we always handle
        # the last index first. This way when removing elements from
        # different iterables we always remove from the end and don't
        # mess up the indexing.
        indices, dim_batches, qim_batches, dir_batches =\
               zip(*sorted(zip(indices, dim_batches, qim_batches, dir_batches),
                            reverse=True))

        if self.qodulus is None:
            qod_func = lambda x: x
        else:
            qod_func = lambda x: x % self.qodulus

        # Step one: Build split_data. It will be a dictionary with keys
        # (ind, qnum), so that
        # split_data[ind, qnum] = [(qcomb1, dcomb1, dcum1),
        #                          (qcomb2, dcomb2, dcum2),
        #                          ... ]
        # Each of these tuples 1,2, ... will correspond to a new block
        # in the split tensor.  Every qcomb is a tuple of quantum
        # numbers, one for each new index coming out of ind, such that
        # their combination is qnum.  Every dcomb is a tuple of
        # corresponding dimensions. Every dcum is a tuple of two numbers
        # (a,b) such that when slicing a block of self, the slice along
        # ind to that gets the block in question in dcum[0]:dcum[1].
        split_data = {}
        for ind, dims, qims, dir_batch in zip(indices, dim_batches,
                                              qim_batches, dir_batches):
            # Find combinations of qims that give the right qnum
            dimcombs = itertools.product(*dims)
            qimcombs = itertools.product(*qims)
            for qcomb, dcomb in zip(qimcombs, dimcombs):
                qcomb_flipped = tuple(map(operator.mul, qcomb, dir_batch))
                qnum = qod_func(sum(qcomb_flipped)*self.dirs[ind])
                if self.qodulus is not None:
                    qnum %= self.qodulus
                if qnum in self.qhape[ind]:
                    try:
                        split_data[ind, qnum].add((qcomb, dcomb))
                    except KeyError:
                        split_data[ind, qnum] = {(qcomb, dcomb)}

        # Sort split_data items and add dcums.
        for k,v in split_data.items():
            v = sorted(v)
            cumulant = 0
            for i, el in enumerate(v):
                prod = reduce(operator.mul, el[1])
                v[i] += ((cumulant, cumulant + prod),)
                cumulant += prod
            split_data[k] = v

        # Step two: Use split_data to do the splitting.
        new_sects = {}
        for k,v in self.sects.items():
            # For every block v that is to be split, we need the
            # corresponding splitting data for every index that is being
            # split.
            datas = map(lambda i: split_data[i, k[i]], indices) 
            # We then take the cartesian product of these datas over the
            # indices being split. That is, every member of the product
            # corresponds to one of the indices.
            for p in itertools.product(*datas):
                # p is a tuple of lists, one list for each ind, where
                # each list consists of three tuples: a qcomb, a dcomb
                # and a cumdim. For each p there will be a block in the
                # new tensor.
                #
                # Build the key for the new block, the slice that gets
                # the new block from v, and the shape that the new block
                # shall be reshaped to.
                s = slice(None)
                slc = [s]*len(self.shape)
                new_key = list(k)
                block_shape = list(v.shape)
                for i, t in enumerate(p):
                    qcomb = t[0]
                    dcomb = t[1]
                    cumdim = t[2]
                    new_key[indices[i]:indices[i]+1] = qcomb
                    block_shape[indices[i]:indices[i]+1] = dcomb
                    slc[indices[i]] = slice(cumdim[0], cumdim[1])
                new_key = tuple(new_key)
                # Blam!
                new_sects[new_key] = v[slc].reshape(block_shape)
        res = self.empty_like()
        res.sects = new_sects

        # Create the new shape, qhape and dirs.
        for ind, dim_b, qim_b, dir_b in zip(indices, dim_batches, qim_batches,
                                            dir_batches):
            res.shape[ind:ind+1] = dim_b
            res.qhape[ind:ind+1] = qim_b
            res.dirs[ind:ind+1] = dir_b

        return res


    def swapaxes(self, i, j):
        """ Swaps two legs, returns a view. """
        res = self.empty_like()
        keys = set(self.sects.keys())
        while keys:
            k = keys.pop()
            kt = list(k)
            kt[i], kt[j] = kt[j], kt[i]
            kt = tuple(kt)
            keys.discard(kt)
            v = self[k].swapaxes(i,j)
            try:
                vt = self[kt].swapaxes(i,j)
                res[k], res[kt] = vt, v
            except KeyError:
                res[kt] = v
        res.shape[i], res.shape[j] = res.shape[j], res.shape[i]
        res.qhape[i], res.qhape[j] = res.qhape[j], res.qhape[i]
        res.dirs[i], res.dirs[j] = res.dirs[j], res.dirs[i]
        return res


    def transpose(self, p=(1,0)):
        """ Transposes legs of self, returns a view. """
        res = self.empty_like()
        for k,v in self.sects.items():
            kt = tuple(map(k.__getitem__, p))
            res.sects[kt] = v.transpose(p)
        res.shape = list(map(self.shape.__getitem__, p))
        res.qhape = list(map(self.qhape.__getitem__, p))
        res.dirs = list(map(self.dirs.__getitem__, p))
        return res


    def trace(self, axis1=0, axis2=1):
        """ Takes a trace of self over axis1 and axis2. This differs
        from the usual trace in the sense that it is more like
        connecting the two legs and contracting. This means that if the
        legs axis1 and axis2 don't have the same dim and qim the
        function will raise an error. If the dirs don't match (both are
        1 or both are -1) then one of them is flipped and a warning is
        raised.

        Note that the diagonal consists always of blocks with the same
        qnum on axis1 and axis2 (once dirs are opposite). This means
        that the trace of an invariant charge != 0 tensor is always a
        zero-tensor.
        """
        assert(self.compatible_indices(self, axis1, axis2))
        if axis1 < axis2:
            axis1, axis2 = axis2, axis1
        if self.dirs[axis1] + self.dirs[axis2] != 0:
            warnings.warn("Automatically flipping dir %i in trace."%axis1,
                          stacklevel=2)
            self = self.flip_dir(axis1)
        res = self.empty_like()
        del(res.shape[axis1])
        del(res.shape[axis2])
        del(res.qhape[axis1])
        del(res.qhape[axis2])
        del(res.dirs[axis1])
        del(res.dirs[axis2])

        # We could avoid looping over the whole dictionary by
        # constructing the cartesiand product of all the qims but not
        # qhape[axis2], and then infering what the qnum on axis2 should
        # be. I suspect that this would be slower, but I haven't tried.
        for k,v in self.sects.items():
            qnum = k[axis1] - k[axis2]
            if self.qodulus is not None:
                qnum %= self.qodulus
            if qnum == 0:
                new_k = tuple(i for j,i in enumerate(k)
                              if j != axis1 and j != axis2)
                if new_k in res.sects:
                    res[new_k] += v.trace(axis1=axis1, axis2=axis2)
                else:
                    res[new_k] = v.trace(axis1=axis1, axis2=axis2)
        if not res.shape:
            # The result is a scalar.
            try:
                res.defval = res[()]
                res.sects = {}
            except KeyError:
                # There was no () block, so the tensor is 0 by symmetry.
                pass
        return res


    def multiply_diag(self, diag_vect, axis, direction="r"):
        """ The result of multiply_diag is the same as
        self.dot(diag_vect.diag(), (axis, 0)) if direction = "right" or
        "r" (the diagonal matrix comes from the right) or
        self.dot(diag_vect.diag(), (axis, 1)) if direction = "left" or
        "l"
        """
        assert(diag_vect.qodulus == self.qodulus)
        assert(diag_vect.charge == 0)
        assert(len(diag_vect.shape) == 1)
        assert(direction in {"r", "l", "left", "right"})

        res = self.empty_like()
        if axis < 0:
            axis = len(self.shape) + axis
        right = direction == "r" or direction == "right"

        if self.qodulus is None:
            qod_func = lambda x: x
        else:
            qod_func = lambda x: x % self.qodulus

        # Flip axes as needed.
        if ((right and self.dirs[axis] != -diag_vect.dirs[0]) or
                (not right and self.dirs[axis] != diag_vect.dirs[0])):
            warnings.warn("Automatically flipping dir 0 of diag_vect in "
                          "multiply_diag.",
                          stacklevel=2)
            diag_vect = diag_vect.flip_dir(0)

        for k,v in self.sects.items():
            q_sum = k[axis]
            k_new = list(k)
            k_new[axis] = q_sum
            k_new = tuple(k_new)
            v = np.swapaxes(v, -1, axis)
            v = v * diag_vect[(q_sum,)]
            v = np.swapaxes(v, -1, axis)
            res[k_new] = v

        res.qhape[axis] = [qod_func(q + diag_vect.charge)
                            for q in diag_vect.qhape[0]]
        res.charge = qod_func(self.charge + diag_vect.charge)
        return res


    def matrix_dot(self, other):
        """ Takes the dot product of two tensors of rank < 3. If self or
        other is a matrix, it must be invariant and have defval == 0.
        """
        assert(self.qodulus == other.qodulus)

        # The following essentially a massive case statement on whether
        # self and other are scalar, vectors or matrices. Unwieldly, but
        # efficient and clear.
        if not self.shape and not other.shape:
            return self*other
        else:
            res_dtype = np.result_type(self.dtype, other.dtype)
            if self.qodulus is None:
                qod_func = lambda x: x
            else:
                qod_func = lambda x: x % self.qodulus
            res_charge = qod_func(self.charge + other.charge)
            res_invar = self.invar and other.invar

            # Vector times vector
            if len(self.shape) == 1 and len(other.shape) == 1:
                assert(self.compatible_indices(other, 0, 0))
                if self.dirs[0] + other.dirs[0] != 0:
                    warnings.warn("Automatically flipping dir 0 of other in "
                                  "dot.")
                    other = other.flip_dir(0)
                res = 0
                for qnum in self.qhape[0]:
                    try:
                        a = self[(qnum,)]
                        b = other[(qnum,)]
                    except KeyError:
                        continue
                    prod = np.dot(a,b)
                    if prod:
                        res += np.dot(a,b)
                res = type(self)([], qhape=[], qodulus=self.qodulus, sects={},
                                 defval=res, dirs=[], dtype=res_dtype,
                                 charge=res_charge, invar=res_invar)
            else:
                res_sects = {}

                # Vector times matrix
                if len(self.shape) == 1:
                    assert(other.invar)
                    assert(other.defval == 0)
                    assert(self.compatible_indices(other, 0, 0))
                    if self.dirs[0] + other.dirs[0] != 0:
                        warnings.warn("Automatically flipping dir 0 of self "
                                      "in dot.")
                        self = self.flip_dir(0)
                    res_shape = [other.shape[1]]
                    res_qhape = [other.qhape[1]]
                    res_dirs = [other.dirs[1]]
                    flux = -other.dirs[0] * other.dirs[1]
                    for sum_qnum in self.qhape[0]:
                        b_qnum = qod_func(sum_qnum*flux +
                                          other.dirs[1]*other.charge)
                        try:
                            a = self[(sum_qnum,)]
                            b = other[(sum_qnum, b_qnum)]
                            res_sects[(b_qnum,)] = np.dot(a,b)
                        except KeyError:
                            # One of the blocks was zero so the resulting
                            # block will be zero.
                            continue

                # Matrix times vector
                elif len(other.shape) == 1:
                    assert(self.invar)
                    assert(self.defval == 0)
                    assert(self.compatible_indices(other, 1, 0))
                    if self.dirs[1] + other.dirs[0] != 0:
                        warnings.warn("Automatically flipping dir 0 of other "
                                      "in dot.")
                        other = other.flip_dir(0)
                    res_shape = [self.shape[0]]
                    res_qhape = [self.qhape[0]]
                    res_dirs = [self.dirs[0]]
                    flux = -self.dirs[0] * self.dirs[1]
                    for sum_qnum in self.qhape[1]:
                        a_qnum = qod_func(sum_qnum*flux +
                                          self.dirs[0]*self.charge)
                        try:
                            a = self[(a_qnum, sum_qnum)]
                            b = other[(sum_qnum,)]
                            res_sects[(a_qnum,)] = np.dot(a,b)
                        except KeyError:
                            # One of the blocks was zero so the resulting
                            # block will be zero.
                            continue

                # Matrix times matrix
                else:
                    assert(self.invar and other.invar)
                    assert(self.defval == other.defval == 0)
                    assert(self.compatible_indices(other, 1, 0))
                    if self.dirs[1] + other.dirs[0] != 0:
                        warnings.warn("Automatically flipping dir 0 of other "
                                      "in dot.")
                        other = other.flip_dir(0)
                    res_shape = [self.shape[0], other.shape[1]]
                    res_qhape = [self.qhape[0], other.qhape[1]]
                    res_dirs = [self.dirs[0], other.dirs[1]]
                    a_flux = -self.dirs[0] * self.dirs[1]
                    b_flux = -other.dirs[0] * other.dirs[1]
                    for sum_qnum in self.qhape[1]:
                        a_qnum = qod_func(sum_qnum*a_flux +
                                          self.dirs[0]*self.charge)
                        b_qnum = qod_func(sum_qnum*b_flux +
                                          other.dirs[1]*other.charge)
                        try:
                            a = self[a_qnum, sum_qnum]
                            b = other[sum_qnum, b_qnum]
                            res_sects[a_qnum, b_qnum] = np.dot(a,b)
                        except KeyError:
                            # One of the blocks was zero so the resulting
                            # block will be zero.
                            continue
                res = type(self)(res_shape, qhape=res_qhape,
                                 qodulus=self.qodulus, sects=res_sects,
                                 dtype=res_dtype, dirs=res_dirs,
                                 charge=res_charge, invar=res_invar)
        return res


    def matrix_eig(self, chis=None, eps=0, print_errors=0, hermitian=False,
                   break_degenerate=False, degeneracy_eps=1e-6,
                   norm_type="frobenius"):
        """ Find eigenvalues and eigenvectors of a matrix. The input
        must have defval == 0, invar == True, charge == 0 and must be
        square in the sense that the dimensions must have the same qim
        and dim and opposing dirs.

        If hermitian is True the matrix is assumed to be hermitian.

        Truncation works like for SVD.

        The output is in the form S, U, where S is a non-invariant
        vector of eigenvalues and U is a matrix that has its columns the
        eigenvectors. Both have the same dim and qim as self.
        """
        chis = self.matrix_decomp_format_chis(chis, eps)
        np_func = np.linalg.eigh if hermitian else np.linalg.eig
        assert(self.defval == 0)
        assert(self.invar)
        assert(self.charge == 0)
        assert(self.dirs[0] + self.dirs[1] == 0)
        assert(set(zip(self.qhape[0], self.shape[0])) ==
               set(zip(self.qhape[1], self.shape[1])))
        if self.qodulus is None:
            qod_func = lambda x: x
        else:
            qod_func = lambda x: x % self.qodulus

        S_dtype = np.float_ if hermitian else np.complex_
        U_dtype = self.dtype if hermitian else np.complex_

        eigdecomps = {}
        dims = {}
        minusabs_next_eigs = []
        all_eigs = []
        for k,v in self.sects.items():
            if 0 not in v.shape:
                s, u = np_func(v)
                order = np.argsort(-np.abs(s))
                s = s[order]
                u = u[:,order]
                s = s.astype(S_dtype)
                u = u.astype(U_dtype)
                eigdecomp = (s, u)
            else:
                shp = v.shape
                m = min(shp)
                u = np.empty((shp[0], m), dtype=U_dtype)
                s = np.empty((m,), dtype=S_dtype)
                eigdecomp = (u, s)
            eigdecomps[k] = eigdecomp
            dims[k] = 0
            all_eigs.append(s)
            if 0 not in s.shape:
                heapq.heappush(minusabs_next_eigs, (-np.abs(s[0]), k))
        try:
            all_eigs = np.concatenate(all_eigs)
        except ValueError:
            # all_eigs == []
            all_eigs = np.array((0,))
        
        # Truncate, if truncation dimensions are given.
        chi, dims, rel_err = type(self).find_trunc_dim(
            all_eigs, eigdecomps, minusabs_next_eigs, dims,
            chis=chis, eps=eps, break_degenerate=break_degenerate,
            degeneracy_eps=degeneracy_eps, norm_type=norm_type)

        if print_errors > 0:
            print('Relative truncation error in eig: '
                  '%.3e' % rel_err)

        # Truncate each block and create the dim for the new index.
        new_dim = []
        new_qim = []
        eigdecomps = {k:v for k,v in eigdecomps.items() if dims[k] > 0}
        for k,v in eigdecomps.items():
            d = dims[k]
            if d>0:
                new_dim.append(d)
                new_qim.append(k[0])
                eigdecomps[k] = (v[0][:d], v[1][:,:d])
            else:
                del(eigdecomps[k])

        # Initialize U, S.
        d = self.dirs[0]
        S = type(self)([new_dim], qhape=[new_qim], dirs=[d],
                       qodulus=self.qodulus, dtype=S_dtype, invar=False,
                       charge=0)
        U = type(self)([self.shape[0], new_dim],
                       qhape=[self.qhape[0], new_qim],
                       dirs=[d,-d],
                       qodulus=self.qodulus, dtype=U_dtype,
                       charge=0)

        # Set the blocks of U, S and V.
        for k,v in eigdecomps.items():
            S[(k[0],)] = v[0]
            k_U = (k[0], k[0])
            U[k_U] = v[1]

        return S, U, rel_err


    def matrix_svd(self, chis=None, eps=0, print_errors=0,
                   break_degenerate=False, degeneracy_eps=1e-6,
                   norm_type="frobenius"):
        """ SVD a matrix. The matrix must have invar == True and defval
        == 0.

        The optional argument chis is a list of bond dimensions. The SVD
        is truncated to one of these dimensions chi, meaning that only
        chi largest singular values are kept. If chis is a single
        integer (either within a singleton list or just as a bare
        integer) this dimension is used. If no eps==0, the largest value
        in chis is used. Otherwise the smallest chi in chis is used,
        such that the relative error made in the truncation is smaller
        than eps.

        An exception to the above is degenerate singular values. By
        default truncation is never done so that some singular values
        are included while others of the same value are left out. If
        this is about to happen chi is decreased so that none of the
        degenerate singular values is included. This default behavior
        can be changed with the keyword argument break_degenerate=True.
        The default threshold for when singular values are considered
        degenerate is 1e-6. This can be changed with the keyword
        argument degeneracy_eps.

        If print_errors > 0 truncation error is printed.

        norm_type specifies the norm used to measure the error. This
        defaults to "frobenius". The other option is "trace", for trace
        norm.

        The method returns the tuple U, S, V, rel_err, where S is a
        non-invariant vector and U and V are unitary matrices. They are
        such that U.diag(S).V = self, where the equality is appromixate
        if there is truncation. U and S have always charge 0, but V has
        the same charge as self. U has dirs [d,-d] where d =
        self.dirs[0], but V has the same dirs as self. rel_err is the
        relative Frobenius norm error caused by the truncation.
        """
        chis = self.matrix_decomp_format_chis(chis, eps)
        assert(self.defval == 0)
        assert(self.invar)
        if self.qodulus is None:
            qod_func = lambda x: x
        else:
            qod_func = lambda x: x % self.qodulus

        svds = {}
        dims = {}
        minus_next_sings = []
        all_sings = []
        for k,v in self.sects.items():
            if 0 not in v.shape:
                u, s, v = np.linalg.svd(v, full_matrices=False)
            else:
                shp = v.shape
                m = min(shp)
                u = np.empty((shp[0], m), dtype=self.dtype)
                s = np.empty((m,), dtype=np.float_)
                v = np.empty((m, shp[1]), dtype=self.dtype)
            svd = (s, u, v)
            svds[k] = svd
            dims[k] = 0
            sings = svd[0]
            all_sings.append(sings)
            if 0 not in sings.shape:
                heapq.heappush(minus_next_sings, (-sings[0], k))
        try:
            all_sings = np.concatenate(all_sings)
        except ValueError:
            # all_sings == []
            all_sings = np.array((0,))
        
        # Truncate, if truncation dimensions are given.
        chi, dims, rel_err = type(self).find_trunc_dim(
            all_sings, svds, minus_next_sings, dims,
            chis=chis, eps=eps, break_degenerate=break_degenerate,
            degeneracy_eps=degeneracy_eps, norm_type=norm_type)

        if print_errors > 0:
            print('Relative truncation error in SVD: %.3e' % rel_err)

        # Truncate each block and create the dim for the new index.
        new_dim = []
        new_qim = []
        svds = {k:v for k,v in svds.items() if dims[k] > 0}
        for k,v in svds.items():
            d = dims[k]
            if d>0:
                new_dim.append(d)
                new_qim.append(k[0])
                svds[k] = (v[0][:d], v[1][:,:d], v[2][:d,:])
            else:
                del(svds[k])

        # Initialize U, S, V.
        d = self.dirs[0]
        U = type(self)([self.shape[0], new_dim],
                       qhape=[self.qhape[0], new_qim],
                       dirs=[d,-d],
                       qodulus=self.qodulus, dtype=self.dtype,
                       charge=0)
        S = type(self)([new_dim], qhape=[new_qim], dirs=[d],
                       qodulus=self.qodulus, dtype=np.float_, invar=False,
                       charge=0)
        V = type(self)([new_dim, self.shape[1]],
                       qhape=[new_qim, self.qhape[1]],
                       dirs=[d, self.dirs[1]],
                       qodulus=self.qodulus, dtype=self.dtype,
                       charge=self.charge)

        # Set the blocks of U, S and V.
        for k,v in svds.items():
            k_U = (k[0], k[0])
            S[(k[0],)] = v[0]
            U[k_U] = v[1]
            V[k] = v[2]

        return U, S, V, rel_err

