import numpy as np
import sys
import warnings
from timer import Timer
from scon import scon
from custom_parser import parse_argv
from tensors.ndarray_svd import svd, eig
from tensors.tensorcommon import TensorCommon
from tensors.tensor import Tensor
from tensors.symmetrytensors import TensorZ2, TensorU1, TensorZ3

""" A test suite for the tensors package. """

# We do not accept warnings in the test, so raise them as errors.
warnings.simplefilter("error", UserWarning)

pars = parse_argv(sys.argv,
                  # Format is: (name_of_argument, type, default)
                  ("test_to_and_from_ndarray", "bool", True),
                  ("test_arithmetic_and_comparison", "bool", True),
                  ("test_transposing", "bool", True),
                  ("test_splitting_and_joining", "bool", True),
                  ("test_to_and_from_matrix", "bool", True),
                  ("test_diag", "bool", True),
                  ("test_trace", "bool", True),
                  ("test_multiply_diag", "bool", True),
                  ("test_product", "bool", True),
                  ("test_svd", "bool", True),
                  ("test_eig", "bool", True),
                  ("test_split", "bool", True),
                  ("test_miscellaneous", "bool", True),
                  ("test_expand_dims_product", "bool", True),
                  ("test_scon_svd_scon", "bool", True),
                  ("n_iters", "int", 500),
                  ("classes", "word_list", ["Tensor", "TensorZ2", "TensorU1",
                                            "TensorZ3"]))
pars = vars(pars)

classes = []
for w in pars["classes"]:
    if w.strip().lower() == "tensor":
        classes.append(Tensor)
    elif w.strip().lower() == "tensorz2":
        classes.append(TensorZ2)
    elif w.strip().lower() == "tensoru1":
        classes.append(TensorU1)
    elif w.strip().lower() == "tensorz3":
        classes.append(TensorZ3)

timer = Timer()
timer.start()

def test_internal_consistency(T):
    if not isinstance(T, (Tensor, np.generic, np.ndarray)):
        T.check_consistency()


for cls in classes:

    print("\nTesting class " + str(cls))

    if cls == TensorZ2:
        n_qnums = 2
    elif cls == TensorZ3:
        n_qnums = 3
    else:
        n_qnums = None

    def rshape(n=None, chi=None, nlow=0, nhigh=5, chilow=0, chihigh=3):
        if n is None:
            n = np.random.randint(nlow, high=nhigh)
        shape = []
        for i in range(n):
            if n_qnums is None:
                dim_dim = np.random.randint(low=1, high=5)
            else:
                dim_dim = n_qnums
            if chi is None:
                dim = []
                for _ in range(dim_dim):
                    dim.append(np.random.randint(0, high=chihigh))
            else:
                dim = [chi]*dim_dim
            s = sum(dim)
            if s < chilow:
                j = np.random.randint(0, len(dim))
                dim[j] += chilow - s
            shape.append(dim)
        return shape 

    def rqhape(shape, qlow=-3, qhigh=3):
        if n_qnums is not None:
            qlow = 0
            qhigh = n_qnums-1
        try:
            assert(all(len(dim) <= qhigh-qlow+1 for dim in shape))
        except TypeError:
            pass
        possible_qnums = range(qlow, qhigh+1)
        try:
            qhape = [list(np.random.choice(possible_qnums, len(dim),
                                           replace=False))
                     for dim in shape]
        except TypeError:
            qhape = None
        return qhape 

    def rdirs(shape=None, length=None):
        if shape is not None:
            length = len(shape)
        dirs = np.random.randint(low=0, high=2, size=length)
        dirs = list(2*dirs - 1)
        return dirs 

    def rcharge():
        return np.random.randint(low=0, high=4)

    def rtensor(shape=None, qhape=None, n=None, chi=None,
                nlow=0, nhigh=5, chilow=0, chihigh=6,
                charge=None, dirs=None, cmplx=True, **kwargs):
        if shape is None:
            shape = rshape(n=n, chi=chi, nlow=nlow, nhigh=nhigh, chilow=chilow,
                           chihigh=chihigh)
        if qhape is None:
            qhape = rqhape(shape)
        if dirs is None:
            dirs = rdirs(shape=shape)
        elif dirs == 1:
            dirs = [1]*len(shape)
        if charge is None:
            charge = rcharge()

        real = cls.random(shape, qhape=qhape, dirs=dirs, charge=charge,
                          **kwargs)
        if cmplx:
            imag = cls.random(shape, qhape=qhape, dirs=dirs, charge=charge,
                              **kwargs)
            res = real + 1j*imag
        else:
            res = real
        return res

    def test_with_np(func, T, S, T_np, S_np):
        tensor_res = func(S,T)
        np_res = func(S_np, T_np)
        np_res = type(tensor_res).from_ndarray(np_res, shape=tensor_res.shape,
                                               qhape=tensor_res.qhape,
                                               dirs=tensor_res.dirs,
                                               charge=tensor_res.charge)
        return (tensor_res == np_res).all()


    if pars["test_to_and_from_ndarray"]:
        for iter_num in range(pars["n_iters"]):
            T = rtensor()
            T_np = T.to_ndarray()
            S = cls.from_ndarray(T_np, shape=T.shape, qhape=T.qhape,
                                 dirs=T.dirs, charge=T.charge)
            test_internal_consistency(T)
            test_internal_consistency(S)
            assert((T==S).all())
        print('Done testing to and from ndarray.')


    if pars["test_arithmetic_and_comparison"]:
        for iter_num in range(pars["n_iters"]):
            s = rshape()
            q = rqhape(s)
            d = rdirs(shape=s)
            T = rtensor(shape=s, qhape=q, dirs=d, cmplx=False)
            c = T.charge
            T_np = T.to_ndarray()
            S = rtensor(shape=s, qhape=q, dirs=d, charge=c, cmplx=False)
            S_np = S.to_ndarray()
            assert(((S+T) - T).allclose(S))
            assert(((-S)+S).allclose(cls.zeros(s, qhape=q, dirs=d, charge=c)))
            assert((0*S).allclose(cls.zeros(s, qhape=q, dirs=d, charge=c)))
            assert((S*0).allclose(cls.zeros(s, qhape=q, dirs=d, charge=c)))
            assert((S*cls.zeros(s, qhape=q, dirs=d, charge=c)).allclose(
                cls.zeros(s, qhape=q, dirs=d, charge=c)))
            assert((cls.zeros(s, qhape=q, dirs=d, charge=c)*S).allclose(
                cls.zeros(s, qhape=q, dirs=d, charge=c)))
            assert((S*cls.ones(s, qhape=q, dirs=d, charge=c)).allclose(S))
            assert((cls.ones(s, qhape=q, dirs=d, charge=c)*S).allclose(S))
            assert(((S*2)/2).allclose(S))
            assert((2*(S/2)).allclose(S))
            assert(((S+2) - 2).allclose(S))
            assert((T==T).all())
            assert(not (T>T).any())
            assert(test_with_np(lambda a,b: a+b, T, S, T_np, S_np))
            assert(test_with_np(lambda a,b: a-b, T, S, T_np, S_np))
            assert(test_with_np(lambda a,b: a*b, T, S, T_np, S_np))
            assert(test_with_np(lambda a,b: a>b, T, S, T_np, S_np))
            assert(test_with_np(lambda a,b: a==b, T, S, T_np, S_np))
        print('Done testing arithmetic and comparison.')


    if pars["test_transposing"]:
        for iter_num in range(pars["n_iters"]):
            T = rtensor(nlow=1)
            shp = T.shape
            i = np.random.randint(low=0, high=len(shp))
            j = np.random.randint(low=0, high=len(shp))
            S = T.copy()
            S = S.swapaxes(i,j)
            T = T.swapaxes(j,i)
            assert((S==T).all())
            test_internal_consistency(T)
            T = T.swapaxes(i,i)
            assert((S==T).all())
            test_internal_consistency(T)
            T = T.transpose(range(len(shp)))
            assert((S==T).all())
            test_internal_consistency(T)
            perm = list(range(len(shp)))
            np.random.shuffle(perm)
            T_copy = T.copy()
            T = T.transpose(perm)
            T_tr_np = T.to_ndarray()
            T_np_tr = np.transpose(T_copy.to_ndarray(), perm)
            assert(np.all(T_tr_np == T_np_tr))
        print('Done testing transposing.')


    if pars["test_splitting_and_joining"]:
        # First join and then split two indices, compare with original.
        for iter_num in range(pars["n_iters"]):
            T = rtensor(nlow=2)
            T_orig = T.copy()
            shp = T.shape
            qhp = T.qhape

            i = np.random.randint(low=0, high=len(shp))
            j = i
            while j==i:
                j = np.random.randint(low=0, high=len(shp))
            i_dim = shp[i]
            j_dim = shp[j]
            try:
                i_qim = qhp[i]
                j_qim = qhp[j]
            except TypeError:
                i_qim = None
                j_qim = None
            if T.dirs is not None:
                di, dj = T.dirs[i], T.dirs[j]
            else:
                di, dj = None, None
            new_d = rdirs(length=1)[0]
            T_joined = T.join_indices(i,j, dirs=new_d)
            assert((T == T_orig).all())
            T = T_joined
            test_internal_consistency(T)

            if j < i:
                i_new = i-1
            else:
                i_new = i
            j_new = i_new + 1

            if T.dirs is not None:
                assert(T.dirs[i_new] == new_d)

            if i != j:
                T_before_split = T.copy()
                T_split = T.split_indices(i_new, (i_dim, j_dim),
                                          qims=(i_qim, j_qim), dirs=(di, dj))
                assert((T_before_split == T).all())
                T = T_split
                test_internal_consistency(T)
                while j_new != j:
                    if j_new > j:
                        T = T.swapaxes(j_new, j_new-1)
                        j_new = j_new - 1
                    else:
                        T = T.swapaxes(j_new, j_new+1)
                        j_new = j_new + 1
                test_internal_consistency(T)
            assert((T_orig==T).all())

        #TODO
        # First split then join two indices, compare with original.
        #for iter_num in range(pars["n_iters"]):
        #    shp = rshape(nlow=1)
        #    i = np.random.randint(low=0, high=len(shp))
        #    m_dim = rshape(n=1)[0]
        #    n_dim = rshape(n=1)[0]
        #    i_dim = cls.combined_dim(m_dim, n_dim)
        #    shp[i] = i_dim
        #    T = rtensor(shape=shp)
        #    T_orig = T.copy()
        #    T = T.split_indices(i, (m_dim, n_dim))
        #    test_internal_consistency(T)
        #    T = T.join_indices(i,i+1)
        #    test_internal_consistency(T)
        #    assert((T_orig==T).all())

        # First join then split many indices, don't compare.
        for iter_num in range(pars["n_iters"]):
            T = rtensor(nlow=1)
            T_orig = T.copy()
            shp = T.shape

            batch_sizes = []
            while True:
                new_size = np.random.randint(low=1, high=len(T.shape)+1)
                if sum(batch_sizes) + new_size <= len(T.shape):
                    batch_sizes.append(new_size)
                else:
                    break
            index_batches = []
            sum_inds = list(np.random.choice(range(len(T.shape)),
                                             size=sum(batch_sizes),
                                             replace=False))
            cumulator = 0
            for b_n in batch_sizes:
                index_batches.append(sum_inds[cumulator:cumulator+b_n])
                cumulator += b_n

            not_joined = sorted(set(range(len(T.shape))) - set(sum_inds))
            batch_firsts = [batch[0] for batch in index_batches]
            remaining_indices = sorted(not_joined + batch_firsts)
            batch_new_indices = [remaining_indices.index(i)
                                 for i in batch_firsts]
            dim_batches = [[T.shape[i] for i in batch]
                           for batch in index_batches]
            if T.qhape is not None:
                qim_batches = [[T.qhape[i] for i in batch]
                               for batch in index_batches]
            else:
                qim_batches = None
            if T.dirs is not None:
                dir_batches = [[T.dirs[i] for i in batch]
                               for batch in index_batches]
            else:
                dir_batches = None
            new_dirs = rdirs(length=len(index_batches))

            T = T.join_indices(*tuple(index_batches), dirs=new_dirs)
            test_internal_consistency(T)

            T = T.split_indices(batch_new_indices, dim_batches,
                                qims=qim_batches, dirs=dir_batches)
            test_internal_consistency(T)
        print('Done testing splitting and joining.')


    if pars["test_to_and_from_matrix"]:
        for iter_num in range(pars["n_iters"]):
            T = rtensor()
            T_orig = T.copy()
            n = np.random.randint(low=0, high=len(T.shape)+1)
            if n:
                i_list = list(np.random.choice(len(T.shape), size=n,
                                               replace=False))
            else:
                i_list = []
            i_list_compl = sorted(set(range(len(T.shape))) - set(i_list))
            T_matrix, T_transposed_shape, T_transposed_qhape, T_transposed_dirs =\
                    T.to_matrix(i_list, i_list_compl, return_transposed_shape_data=True)
            assert((T == T_orig).all())
            T = T_matrix
            test_internal_consistency(T)
            T_orig = T_orig.transpose(i_list + i_list_compl)
            assert(T_transposed_shape == T_orig.shape)
            l_dims = T_transposed_shape[:len(i_list)]
            r_dims = T_transposed_shape[len(i_list):]
            if T_transposed_qhape is not None:
                l_qims = T_transposed_qhape[:len(i_list)]
                r_qims = T_transposed_qhape[len(i_list):]
            else:
                l_qims = None
                r_qims = None
            if T_transposed_dirs is not None:
                l_dirs = T_transposed_dirs[:len(i_list)]
                r_dirs = T_transposed_dirs[len(i_list):]
            else:
                l_dirs = None
                r_dirs = None
            T_matrix = T.copy()
            T_tensor = T.from_matrix(l_dims, r_dims,
                                     left_qims=l_qims, right_qims=r_qims,
                                     left_dirs=l_dirs, right_dirs=r_dirs,)
            assert((T == T_matrix).all())
            T = T_tensor 
            test_internal_consistency(T)
            assert((T == T_orig).all())
        print('Done testing to and from matrix.')


    if pars["test_diag"]:
        for iter_num in range(pars["n_iters"]):
            # Vectors to matrices
            T = rtensor(n=1, invar=False)
            T_np = T.to_ndarray()
            T_diag = T.diag()
            T_np_diag = np.diag(T_np)
            T_np_diag = type(T).from_ndarray(T_np_diag, shape=T_diag.shape,
                                             qhape=T_diag.qhape,
                                             dirs=T_diag.dirs,
                                             charge=T_diag.charge)

            assert(T_np_diag.allclose(T_diag))
            # Matrices to vectors
            shp = rshape(n=2)
            shp[1] = shp[0]
            qhp = rqhape(shape=shp)
            qhp[1] = qhp[0]
            dirs = rdirs(shape=shp)
            dirs[1] = -dirs[0]
            T = rtensor(shape=shp, qhape=qhp, dirs=dirs)
            T_np = T.to_ndarray()
            T_diag = T.diag()
            T_np_diag = np.diag(T_np)
            T_np_diag = type(T).from_ndarray(T_np_diag, shape=T_diag.shape,
                                             qhape=T_diag.qhape,
                                             dirs=T_diag.dirs,
                                             charge=T_diag.charge,
                                             invar=False)
            assert(T_np_diag.allclose(T_diag))
        print('Done testing diag.')


    if pars["test_trace"]:
        for iter_num in range(pars["n_iters"]):
            shp = rshape(nlow=2)
            qhp = rqhape(shape=shp)
            dirs = rdirs(shape=shp)
            charge = rcharge()
            i = np.random.randint(low=0, high=len(shp))
            j = np.random.randint(low=0, high=len(shp))
            while i==j:
                j = np.random.randint(low=0, high=len(shp))
            shp[j] = shp[i]
            dirs[j] = -dirs[i]
            qhp[j] = qhp[i]
            T = rtensor(shape=shp, qhape=qhp, dirs=dirs, charge=charge)
            T_np = T.to_ndarray()
            tr = T.trace(axis1=i, axis2=j)
            np_tr = np.trace(T_np, axis1=i, axis2=j)
            test_internal_consistency(tr)
            np_tr_tensor = type(T).from_ndarray(np_tr, shape=tr.shape,
                                                qhape=tr.qhape, dirs=tr.dirs,
                                                charge=tr.charge)
            assert(np_tr_tensor.allclose(tr))
        print('Done testing traces.')


    if pars["test_multiply_diag"]:
        for iter_num in range(pars["n_iters"]):
            right = np.random.randint(low=0, high=2)
            T = rtensor(nlow=1, chilow=1)
            T_orig = T.copy()
            i = np.random.randint(low=0, high=len(T.shape))

            D_shape = [T.shape[i]]
            D_qhape = None if T.qhape is None else [T.qhape[i]]
            D_dirs = None if T.dirs is None else [T.dirs[i] * (1 - 2*right)]
            D = rtensor(shape=D_shape, qhape=D_qhape, dirs=D_dirs,
                        invar=False, charge=0)
            
            T_np = T.to_ndarray()
            D_np = D.to_ndarray()
            prod_np = np.tensordot(T_np, np.diag(D_np), (i,1-right))
            perm = list(range(len(prod_np.shape)))
            d = perm.pop(-1)
            perm.insert(i, d)
            prod_np = np.transpose(prod_np, perm)

            direction = "right" if right else "left"
            TD = T.multiply_diag(D, i, direction=direction)
            assert((T == T_orig).all())
            T = TD
            test_internal_consistency(T)
            assert(np.allclose(T.to_ndarray(), prod_np))
        print('Done testing multiply_diag.')


    if pars["test_product"]:
        for iter_num in range(pars["n_iters"]):
            shp1 = rshape(nlow=1)
            n = np.random.randint(low=1, high=len(shp1)+1)
            if n:
                i_list = list(np.random.choice(len(shp1), size=n, replace=False))
            else:
                i_list = []
            shp2 = rshape(nlow=n, chilow=1)
            if n:
                j_list = list(np.random.choice(len(shp2), size=n, replace=False))
            else:
                j_list = []
            for k in range(n):
                # A summation index should not have dimension 0
                dim1 = shp1[i_list[k]]
                if np.sum(dim1) < 1:
                    dim1 = rshape(n=1, chilow=1)[0]
                    shp1[i_list[k]] = dim1
                shp2[j_list[k]] = dim1
            qhp1 = rqhape(shp1)
            qhp2 = rqhape(shp2)
            if qhp1 is not None:
                for k in range(n):
                    qhp2[j_list[k]] = qhp1[i_list[k]]
            T1 = rtensor(shape=shp1, qhape=qhp1)
            T1_orig = T1.copy()
            if T.dirs is not None:
                dirs2 = rdirs(shape=shp2)
                for i,j in zip(i_list, j_list):
                    dirs2[j] = -T1.dirs[i]
            else:
                dirs2 = None
            T2 = rtensor(shape=shp2, qhape=qhp2, dirs=dirs2)
            T2_orig = T2.copy()
            T1_np = T1.to_ndarray()
            T2_np = T2.to_ndarray()
            T = T1.dot(T2, (i_list, j_list))
            assert((T1 == T1_orig).all())
            assert((T2 == T2_orig).all())
            test_internal_consistency(T)
            i_list_compl = sorted(set(range(len(shp1))) - set(i_list))
            j_list_compl = sorted(set(range(len(shp2))) - set(j_list))
            product_shp = [shp1[i] for i in i_list_compl]\
                          + [shp2[j] for j in j_list_compl]
            if type(T) == Tensor:
                product_shp = Tensor.flatten_shape(product_shp)
            assert(T.shape == product_shp)
            T_np = np.tensordot(T1_np, T2_np, (i_list, j_list))
            assert(np.allclose(T_np, T.to_ndarray()))

            # Products of non-invariant vectors.
            n1 = np.random.randint(1,3)
            T1 = rtensor(n=n1, chilow=1, invar=(n1!=1))

            n2 = np.random.randint(1,3)
            shp2 = rshape(n=n2, chilow=1)
            qhp2 = rqhape(shape=shp2)
            dirs2 = rdirs(shape=shp2)
            c2 = rcharge()
            shp2[0] = T1.shape[-1]
            if T1.qhape is not None:
                qhp2[0] = T1.qhape[-1]
                dirs2[0] = -T1.dirs[-1]
            T2 = rtensor(shape=shp2, qhape=qhp2, dirs=dirs2, charge=c2,
                         invar=(n2!=1))

            T1_orig = T1.copy()
            T2_orig = T2.copy()
            test_internal_consistency(T1)
            test_internal_consistency(T2)
            T1_np = T1.to_ndarray()
            T2_np = T2.to_ndarray()
            T = T1.dot(T2, (n1-1, 0))
            assert((T1 == T1_orig).all())
            assert((T2 == T2_orig).all())
            test_internal_consistency(T)
            T_np = np.tensordot(T1_np, T2_np, (n1-1, 0))
            assert(np.allclose(T_np, T.to_ndarray()))
        print('Done testing products.')


    if pars["test_svd"]:
        for iter_num in range(pars["n_iters"]):
            T = rtensor(nlow=2, chilow=1)
            T_orig = T.copy()
            T_np = T.to_ndarray()
            n = np.random.randint(low=1, high=len(T.shape))
            if n:
                i_list = list(np.random.choice(len(T.shape), size=n, replace=False))
            else:
                i_list = []
            i_list_compl = sorted(set(range(len(T.shape))) - set(i_list))
            np.random.shuffle(i_list_compl)

            # (Almost) no truncation.
            U, S, V = T.svd(i_list, i_list_compl, eps=1e-15)
            assert((T == T_orig).all())
            test_internal_consistency(U)
            test_internal_consistency(S)
            test_internal_consistency(V)
            US = U.dot(S.diag(), (len(i_list), 0))
            USV = US.dot(V, (len(i_list), 0))
            T = T.transpose(i_list+i_list_compl)
            assert(USV.allclose(T))

            U_np_svd, S_np_svd, V_np_svd = svd(T_np, i_list, i_list_compl, eps=1e-15)
            U_svd_np, S_svd_np, V_svd_np = U.to_ndarray(), S.to_ndarray(), V.to_ndarray()

            order = np.argsort(-S_svd_np)
            S_svd_np = S_svd_np[order]
            U_svd_np = U_svd_np[...,order]
            V_svd_np = V_svd_np[order,...]
            # abs is needed because of gauge freedom in SVD. We assume here
            # that there are no degenerate singular values.
            assert(np.allclose(np.abs(U_np_svd), np.abs(U_svd_np)))
            assert(np.allclose(np.abs(S_np_svd), np.abs(S_svd_np)))
            assert(np.allclose(np.abs(V_np_svd), np.abs(V_svd_np)))

            # Truncation.
            chi = np.random.randint(low=1, high=6)
            chis = list(range(chi+1))
            eps = 1e-5
            U, S, V, rel_err = T_orig.svd(i_list, i_list_compl, chis=chis,
                                          eps=eps, return_rel_err=True)
            test_internal_consistency(U)
            test_internal_consistency(S)
            test_internal_consistency(V)
            assert(rel_err<eps or sum(type(S).flatten_shape(S.shape)) == chi)
            US = U.dot(S.diag(), (len(i_list), 0))
            USV = US.dot(V, (len(i_list), 0))
            err = (USV - T).norm()
            T_norm = T_orig.norm()
            if T_norm != 0:
                true_rel_err = err/T_norm
            else:
                true_rel_err = 0
            if rel_err > 1e-7 or true_rel_err > 1e-7:
                # If this doesnt' hold we run into machine epsilon
                # because of a square root.
                assert(np.abs(rel_err - true_rel_err)/(rel_err+true_rel_err) < 1e-7)
            else:
                assert(USV.allclose(T))

            U_np_svd, S_np_svd, V_np_svd, np_rel_err = svd(T_np, i_list, i_list_compl, chis=chis,
                                                           eps=eps, return_rel_err=True)
            assert(np.allclose(rel_err, np_rel_err, atol=1e-7))
            assert(np.allclose(-np.sort(-S.to_ndarray()), S_np_svd))
        print('Done testing SVD.')
            

    if pars["test_eig"]:
        for iter_num in range(pars["n_iters"]):
            n = np.random.randint(low=1, high=3)
            shp = rshape(n=n*2, chilow=1, chihigh=4)
            qhp = rqhape(shape=shp)
            dirs = [1]*len(shp)
            i_list = list(np.random.choice(len(shp), size=n, replace=False))
            i_list_compl = sorted(set(range(len(shp))) - set(i_list))
            np.random.shuffle(i_list_compl)
            for i,j in zip(i_list, i_list_compl):
                shp[j] = shp[i].copy()
                qhp[j] = qhp[i].copy()
                dirs[j] = -1
            T = rtensor(shape=shp, qhape=qhp, dirs=dirs, charge=0)
            T_orig = T.copy()
            T_np = T.to_ndarray()

            # No truncation, non-hermitian
            S, U = T.eig(i_list, i_list_compl)
            assert((T == T_orig).all())
            test_internal_consistency(S)
            test_internal_consistency(U)

            S_np_eig, U_np_eig = eig(T_np, i_list, i_list_compl)
            S_eig_np, U_eig_np = S.to_ndarray(), U.to_ndarray()

            order = np.argsort(-S_eig_np)
            S_eig_np = S_eig_np[order]
            U_eig_np = U_eig_np[...,order]
            order = np.argsort(-S_np_eig)
            S_np_eig = S_np_eig[order]
            U_np_eig = U_np_eig[...,order]
            assert(np.allclose(S_np_eig, S_eig_np))
            assert(np.allclose(np.abs(U_np_eig), np.abs(U_eig_np)))

            # Truncation, non-hermitian
            chi = np.random.randint(low=1, high=6)
            chis = list(range(chi+1))
            eps = 1e-5
            S, U, rel_err = T.eig(i_list, i_list_compl, chis=chis, eps=eps,
                                  return_rel_err=True)
            assert((T == T_orig).all())
            test_internal_consistency(S)
            test_internal_consistency(U)

            S_np_eig, U_np_eig, rel_err_np = eig(T_np, i_list, i_list_compl,
                                                 chis=chis, eps=eps,
                                                 return_rel_err=True)
            S_eig_np, U_eig_np = S.to_ndarray(), U.to_ndarray()

            order = np.argsort(-S_eig_np)
            S_eig_np = S_eig_np[order]
            U_eig_np = U_eig_np[...,order]
            order = np.argsort(-S_np_eig)
            S_np_eig = S_np_eig[order]
            U_np_eig = U_np_eig[...,order]
            assert(np.allclose(S_np_eig, S_eig_np))
            assert(np.allclose(np.abs(U_np_eig), np.abs(U_eig_np)))
            assert(np.allclose(rel_err, rel_err_np))
            assert(rel_err<eps or sum(type(S).flatten_shape(S.shape)) == chi)

            # No truncation, hermitian
            T_scon_list = list(range(-len(T.shape), 0))
            T_conj_scon_list = [i - 100 for i in T_scon_list]
            for counter, i in enumerate(i_list_compl):
                T_scon_list[i] = counter+1
                T_conj_scon_list[i] = counter+1

            T = scon((T, T.conjugate()), (T_scon_list, T_conj_scon_list))
            T_orig = T.copy()
            T_np = T.to_ndarray()
            i_list = list(range(len(i_list_compl)))
            i_list_compl = [len(i_list) + i for i in i_list]

            S, U = T.eig(i_list, i_list_compl, hermitian=True)
            assert((T == T_orig).all())
            test_internal_consistency(S)
            test_internal_consistency(U)

            S_np_eig, U_np_eig = eig(T_np, i_list, i_list_compl,
                                     hermitian=True)
            S_eig_np, U_eig_np = S.to_ndarray(), U.to_ndarray()

            order = np.argsort(-S_eig_np)
            S_eig_np = S_eig_np[order]
            U_eig_np = U_eig_np[...,order]
            order = np.argsort(-S_np_eig)
            S_np_eig = S_np_eig[order]
            U_np_eig = U_np_eig[...,order]
            assert(np.allclose(S_np_eig, S_eig_np))
            assert(np.allclose(np.abs(U_np_eig), np.abs(U_eig_np)))

            # Truncation, hermitian
            chi = np.random.randint(low=1, high=6)
            chis = list(range(chi+1))
            eps = 1e-5
            S, U, rel_err = T.eig(i_list, i_list_compl, chis=chis, eps=eps,
                                  hermitian=True, return_rel_err=True)
            assert((T == T_orig).all())
            test_internal_consistency(S)
            test_internal_consistency(U)

            S_np_eig, U_np_eig, rel_err_np = eig(T_np, i_list, i_list_compl,
                                                 chis=chis, eps=eps,
                                                 hermitian=True,
                                                 return_rel_err=True)
            S_eig_np, U_eig_np = S.to_ndarray(), U.to_ndarray()

            order = np.argsort(-S_eig_np)
            S_eig_np = S_eig_np[order]
            U_eig_np = U_eig_np[...,order]
            order = np.argsort(-S_np_eig)
            S_np_eig = S_np_eig[order]
            U_np_eig = U_np_eig[...,order]
            assert(np.allclose(S_np_eig, S_eig_np))
            assert(np.allclose(np.abs(U_np_eig), np.abs(U_eig_np)))
            assert(np.allclose(rel_err, rel_err_np))
            assert(rel_err<eps or sum(type(S).flatten_shape(S.shape)) == chi)

            l = len(U.shape)
            V_permutation = (l-1,) + tuple(range(l-1))
            V = U.conjugate().transpose(V_permutation)
            US = U.dot(S.diag(), (len(i_list), 0))
            USV = US.dot(V, (len(i_list), 0))
            err = (USV - T).norm()
            T_norm = T_orig.norm()
            if T_norm != 0:
                true_rel_err = err/T_norm
            else:
                true_rel_err = 0
            if rel_err > 1e-7 or true_rel_err > 1e-7:
                # If this doesnt' hold we run into machine epsilon
                # because of a square root.
                assert(np.abs(rel_err - true_rel_err)/(rel_err+true_rel_err) < 1e-7)
            else:
                assert(USV.allclose(T))
        print('Done testing eig.')
            

    if pars["test_split"]:
        for iter_num in range(pars["n_iters"]):
            T = rtensor(nlow=2, chilow=1)
            T_orig = T.copy()
            n = np.random.randint(low=1, high=len(T.shape))
            i_list = []
            while len(i_list) < n:
                i_list.append(np.random.randint(low=0, high=len(T.shape)))
                i_list = list(set(i_list))
            i_list_compl = sorted(set(range(len(T.shape))) - set(i_list))
            np.random.shuffle(i_list)
            np.random.shuffle(i_list_compl)

            chi = np.random.randint(low=1, high=10)
            eps = 10**(-1*np.random.randint(low=2, high=10))
            svd_res = T.svd(i_list, i_list_compl, chis=chi, eps=eps)
            assert((T == T_orig).all())
            U, S, V = svd_res[0:3]
            test_internal_consistency(U)
            test_internal_consistency(S)
            test_internal_consistency(V)
            US = U.dot(S.sqrt().diag(), (len(i_list), 0))
            SV = V.dot(S.sqrt().diag(), (0, 1))
            perm = list(range(len(SV.shape)))
            d = perm.pop(-1)
            perm.insert(0, d)
            SV = SV.transpose(perm)
            split_res = T.split(i_list, i_list_compl, chis=chi, eps=eps, return_sings=True)
            assert(US.allclose(split_res[0]))
            assert(S.allclose(split_res[1]))
            assert(SV.allclose(split_res[2]))
        print('Done testing splitting tensors.')


    if pars["test_miscellaneous"]:
        for iter_num in range(pars["n_iters"]):
            # Test norm
            shp = rshape()
            for dim in shp:
                if all([d == 0 for d in dim]):
                    dim[0] = 1
            T = rtensor(shape=shp)
            T_np = T.to_ndarray()
            T_norm = T.norm()
            n = len(T.shape)
            all_inds = tuple(range(n))
            T_np_norm = np.sqrt(np.tensordot(T_np, T_np.conj(), (all_inds, all_inds)))
            assert(np.allclose(T_norm, T_np_norm))

            # Test min, max and average
            shp = rshape()
            for dim in shp:
                if all([d == 0 for d in dim]):
                    dim[0] = 1
            T = rtensor(shape=shp, cmplx=False)
            T_np = T.to_ndarray()
            T_max = T.max()
            T_np_max = np.max(T_np)
            assert(T_max == T_np_max)

            shp = rshape()
            for dim in shp:
                if all([d == 0 for d in dim]):
                    dim[0] = 1
            T = rtensor(shape=shp, cmplx=False)
            T_np = T.to_ndarray()
            T_min = T.min()
            T_np_min = np.min(T_np)
            assert(T_min == T_np_min)
            
            shp = rshape()
            for dim in shp:
                if all([d == 0 for d in dim]):
                    dim[0] = 1
            T = rtensor(shape=shp)
            T_np = T.to_ndarray()
            T_average = T.average()
            T_np_average = np.average(T_np)
            assert(np.allclose(T_average, T_np_average))

            # Test expand_dim
            T = rtensor()
            T_orig = T.copy()
            axis = np.random.randint(0, high=len(T.shape)+1)
            T_np = T.to_ndarray()
            T_expanded = T.expand_dims(axis)
            assert((T == T_orig).all())
            T = T_expanded
            test_internal_consistency(T)
            T_np = np.expand_dims(T_np, axis)
            T_np_T = type(T).from_ndarray(T_np, shape=T.shape, qhape=T.qhape,
                                          dirs=T.dirs, charge=T.charge)
            test_internal_consistency(T_np_T)
            assert(T.allclose(T_np_T))

            # Test eye
            dim = rshape(n=1)[0]
            qim = rqhape(shape=[dim])[0]
            T = cls.eye(dim, qim=qim)
            T_np = np.eye(T.flatten_dim(dim))
            T_np = type(T).from_ndarray(T_np, shape=T.shape, qhape=T.qhape,
                                        dirs=T.dirs, charge=T.charge)
            assert((T == T_np).all())

            # Test flip_dir
            T = rtensor(nlow=1)
            T_orig = T.copy()
            i = np.random.randint(low=0, high=len(T.shape))
            T_flipped = T.flip_dir(i)
            assert((T == T_orig).all())
            test_internal_consistency(T_flipped)
            T_flipped = T_flipped.flip_dir(i)
            assert((T == T_flipped).all())
        print('Done testing miscellaneous.')


    if pars["test_expand_dims_product"]:
        for iter_num in range(pars["n_iters"]):
            T1 = rtensor()
            T2 = rtensor()
            axis1 = np.random.randint(0, high=len(T1.shape)+1)
            axis2 = np.random.randint(0, high=len(T2.shape)+1)
            T1_np = T1.to_ndarray()
            T2_np = T2.to_ndarray()
            T1 = T1.expand_dims(axis1, direction=1)
            T2 = T2.expand_dims(axis2, direction=-1)
            T1_np = np.expand_dims(T1_np, axis1)
            T2_np = np.expand_dims(T2_np, axis2)
            T = T1.dot(T2, (axis1, axis2))
            test_internal_consistency(T)
            T_np = np.tensordot(T1_np, T2_np, (axis1, axis2))
            T_np_T = type(T).from_ndarray(T_np, shape=T.shape, qhape=T.qhape,
                                          dirs=T.dirs, charge=T.charge)
            test_internal_consistency(T_np_T)
            assert(T.allclose(T_np_T))
        print('Done testing expand_dims products.')


    if pars["test_scon_svd_scon"]:
        for iter_num in range(pars["n_iters"]):
            # Create a random scon contraction
            n_tensors = np.random.randint(low=1, high=4)
            shapes = []
            qhapes = []
            dirss = []
            charges = []
            indices = set()
            for i in range(n_tensors):
                shp = rshape(nhigh=4, chilow=1)
                shapes.append(shp)
                qhapes.append(rqhape(shape=shp))
                dirss.append(rdirs(shape=shp))
                charges.append(rcharge())
                for j in range(len(shp)):
                    indices.add((i,j))

            scon_lists = []
            index_numbers = set(range(-len(indices), 0))
            for shp in shapes:
                scon_list = []
                for index in shp:
                    scon_list.append(index_numbers.pop())
                scon_lists.append(scon_list)

            n_contractions = np.random.randint(low=0, high=int(len(indices)/2)+1)
            for counter in range(1, n_contractions+1):
                t1, i1 = indices.pop()
                t2, i2 = indices.pop()
                shapes[t2][i2] = shapes[t1][i1]
                qhapes[t2][i2] = qhapes[t1][i1]
                dirss[t2][i2] = -dirss[t1][i1]
                scon_lists[t1][i1] = counter
                scon_lists[t2][i2] = counter

            tensors = []
            np_tensors = []
            for shape, qhape, dirs, charge in zip(shapes, qhapes, dirss, charges):
                tensor = rtensor(shape, qhape=qhape, dirs=dirs, charge=charge)
                np_tensor = tensor.to_ndarray()
                tensors.append(tensor)
                np_tensors.append(np_tensor)
            
            T = scon(tensors, scon_lists)
            test_internal_consistency(T)
            np_T = scon(np_tensors, scon_lists)
            np_T = type(T).from_ndarray(np_T, shape=T.shape, qhape=T.qhape,
                                        dirs=T.dirs, charge=T.charge)
            assert(T.allclose(np_T))

            if len(T.shape) > 1:
                # SVD the result of the contraction
                n_svd_inds = np.random.randint(low=1, high=len(T.shape))
                if n_svd_inds:
                    i_list = list(np.random.choice(len(T.shape), size=n_svd_inds,
                                                   replace=False))
                else:
                    i_list = []
                i_list_compl = sorted(set(range(len(T.shape))) - set(i_list))
                np.random.shuffle(i_list_compl)
                U, S, V = T.svd(i_list, i_list_compl, eps=1e-15)

                # scon U, S and V with S to get the norm_sq of S.
                S_diag = S.diag().conjugate()
                U = U.conjugate()
                V = V.conjugate()
                U_left_inds = [i+1 for i in i_list]
                V_right_inds = [j+1 for j in i_list_compl]
                norm_sq_scon = scon((T, U, S_diag, V),
                                    (list(range(1, len(T.shape)+1)),
                                     U_left_inds + [100],
                                     [100,101],
                                     [101] + V_right_inds))
                norm_sq = S.norm_sq()
                assert(np.allclose(norm_sq, norm_sq_scon.value()))
        print('Done testing scon_svd_scon.')

