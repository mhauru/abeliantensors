"""The main test suite for abeliantensors."""
import numpy as np
import pytest
from ncon import ncon
from .ndarray_decomp import svd, eig
from abeliantensors import Tensor

# # # # # # # # # # # # # # # # # # # #
# Utilities that tests use


def check_with_np(func, T, S, T_np, S_np):
    """Given a function `func` that can take as arguments two `TensorCommon`
    instances or two NumPy arrays, check that `func(T, S)` is the same as
    `func(T_np, S_np)` converted to the type of `T` and `S`.
    """
    tensor_res = func(S, T)
    np_res = func(S_np, T_np)
    np_res = type(tensor_res).from_ndarray(
        np_res,
        shape=tensor_res.shape,
        qhape=tensor_res.qhape,
        dirs=tensor_res.dirs,
        charge=tensor_res.charge,
    )
    return (tensor_res == np_res).all()


def check_internal_consistency(T):
    """If `T` is a symmetric tensor, check that its form data is consistent.
    """
    if not isinstance(T, (Tensor, np.generic, np.ndarray)):
        T.check_consistency()


# # # # # # # # # # # # # # # # # # # #
# The actual tests


def test_to_and_from_ndarray(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Test converting random tensors to ndarrays and back, checking that this
    leaves them invariant.
    """
    for iter_num in range(n_iters):
        T = rtensor()
        T_np = T.to_ndarray()
        S = tensorclass.from_ndarray(
            T_np, shape=T.shape, qhape=T.qhape, dirs=T.dirs, charge=T.charge,
        )
        check_internal_consistency(T)
        check_internal_consistency(S)
        assert (T == S).all()


def test_arithmetic_and_comparison(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Test basic arthmetic and comparison operations."""
    for iter_num in range(n_iters):
        # Create two tensors with the same shape, qhape, and dirs, but possibly
        # different charges.
        s = rshape()
        q = rqhape(s)
        d = rdirs(shape=s)
        T = rtensor(shape=s, qhape=q, dirs=d, cmplx=False)
        c = T.charge
        T_np = T.to_ndarray()
        S = rtensor(shape=s, qhape=q, dirs=d, charge=c, cmplx=False)
        S_np = S.to_ndarray()
        assert ((S + T) - T).allclose(S)
        assert ((-S) + S).allclose(
            tensorclass.zeros(s, qhape=q, dirs=d, charge=c)
        )
        assert (0 * S).allclose(
            tensorclass.zeros(s, qhape=q, dirs=d, charge=c)
        )
        assert (S * 0).allclose(
            tensorclass.zeros(s, qhape=q, dirs=d, charge=c)
        )
        assert (S * tensorclass.zeros(s, qhape=q, dirs=d, charge=c)).allclose(
            tensorclass.zeros(s, qhape=q, dirs=d, charge=c)
        )
        assert (tensorclass.zeros(s, qhape=q, dirs=d, charge=c) * S).allclose(
            tensorclass.zeros(s, qhape=q, dirs=d, charge=c)
        )
        assert (S * tensorclass.ones(s, qhape=q, dirs=d, charge=c)).allclose(S)
        assert (tensorclass.ones(s, qhape=q, dirs=d, charge=c) * S).allclose(S)
        assert ((S * 2) / 2).allclose(S)
        assert (2 * (S / 2)).allclose(S)
        assert ((S + 2) - 2).allclose(S)
        assert (T == T).all()
        assert not (T > T).any()
        assert check_with_np(lambda a, b: a + b, T, S, T_np, S_np)
        assert check_with_np(lambda a, b: a - b, T, S, T_np, S_np)
        assert check_with_np(lambda a, b: a * b, T, S, T_np, S_np)
        assert check_with_np(lambda a, b: a > b, T, S, T_np, S_np)
        assert check_with_np(lambda a, b: a == b, T, S, T_np, S_np)


def test_transposing(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Test transposing and swapaxes on random tensors, checking consistency
    and comparing agaings np.transpose.
    """
    for iter_num in range(n_iters):
        # Create a random tensor with at least one index.
        T = rtensor(nlow=1)
        shp = T.shape
        # Pick to random indices.
        i = np.random.randint(low=0, high=len(shp))
        j = np.random.randint(low=0, high=len(shp))
        # Check that the order of swapaxes arguments doesn't matter.
        S = T.copy()
        S = S.swapaxes(i, j)
        T = T.swapaxes(j, i)
        assert (S == T).all()
        check_internal_consistency(T)
        # Check that trivial swaps and transposes are noops.
        T = T.swapaxes(i, i)
        assert (S == T).all()
        check_internal_consistency(T)
        T = T.transpose(range(len(shp)))
        assert (S == T).all()
        check_internal_consistency(T)
        # Make a random permutation, check that its done correctly using
        # np.transpose to compare.
        perm = list(range(len(shp)))
        np.random.shuffle(perm)
        T_copy = T.copy()
        T = T.transpose(perm)
        T_tr_np = T.to_ndarray()
        T_np_tr = np.transpose(T_copy.to_ndarray(), perm)
        assert np.all(T_tr_np == T_np_tr)


def test_splitting_and_joining_two(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """First join and then split back two indices of a random tensors. Check
    that this leaves the tensor invariant.
    """
    for iter_num in range(n_iters):
        # First join and then split two indices, compare with original.
        T = rtensor(nlow=2)
        T_orig = T.copy()
        shp = T.shape
        qhp = T.qhape

        # Pick to random indices (that must be different).
        i = np.random.randint(low=0, high=len(shp))
        j = i
        while j == i:
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
        # Join the indices, with the new direction being random.
        new_d = rdirs(length=1)[0]
        T_joined = T.join_indices(i, j, dirs=new_d)
        # Check that this didn't affect the original tensor.
        assert (T == T_orig).all()
        T = T_joined
        check_internal_consistency(T)

        if j < i:
            i_new = i - 1
        else:
            i_new = i
        j_new = i_new + 1

        if T.dirs is not None:
            assert T.dirs[i_new] == new_d

        T_before_split = T.copy()
        # Split the indices back to how they were.
        T_split = T.split_indices(
            i_new, (i_dim, j_dim), qims=(i_qim, j_qim), dirs=(di, dj)
        )
        # Check that this didn't modify the argument.
        assert (T_before_split == T).all()
        T = T_split
        check_internal_consistency(T)
        # Rotate the split indices back to their original places.
        while j_new != j:
            if j_new > j:
                T = T.swapaxes(j_new, j_new - 1)
                j_new = j_new - 1
            else:
                T = T.swapaxes(j_new, j_new + 1)
                j_new = j_new + 1
        check_internal_consistency(T)
        # Check that we are back where we started.
        assert (T_orig == T).all()


def test_splitting_and_joining_many(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """First join then split many indices of random tensors. Don't compare to
    the original one though, because doing the permutations right would just be
    so much work.
    """
    for iter_num in range(n_iters):
        T = rtensor(nlow=1)  # Random tensor with at least one index.

        # Generate random sizes for the index batches to be joined.
        batch_sizes = []
        while True:
            new_size = np.random.randint(low=1, high=len(T.shape) + 1)
            if sum(batch_sizes) + new_size <= len(T.shape):
                batch_sizes.append(new_size)
            else:
                break
        # Generate the random index batches. We first generate a list of all
        # the indices that will be joined, then split it into batches.
        index_batches = []
        sum_inds = list(
            np.random.choice(
                range(len(T.shape)), size=sum(batch_sizes), replace=False
            )
        )
        cumulator = 0
        for b_n in batch_sizes:
            index_batches.append(sum_inds[cumulator : cumulator + b_n])
            cumulator += b_n

        # Figure out the remaining indices after the join, and the all the
        # shape information involved.
        not_joined = sorted(set(range(len(T.shape))) - set(sum_inds))
        batch_firsts = [batch[0] for batch in index_batches]
        remaining_indices = sorted(not_joined + batch_firsts)
        batch_new_indices = [remaining_indices.index(i) for i in batch_firsts]
        dim_batches = [[T.shape[i] for i in batch] for batch in index_batches]
        if T.qhape is not None:
            qim_batches = [
                [T.qhape[i] for i in batch] for batch in index_batches
            ]
        else:
            qim_batches = None
        if T.dirs is not None:
            dir_batches = [
                [T.dirs[i] for i in batch] for batch in index_batches
            ]
        else:
            dir_batches = None
        new_dirs = rdirs(length=len(index_batches))

        # First join, then split back, and check that the operation goes
        # through and returns an internally consistent tensor.
        T = T.join_indices(*tuple(index_batches), dirs=new_dirs)
        check_internal_consistency(T)
        T = T.split_indices(
            batch_new_indices, dim_batches, qims=qim_batches, dirs=dir_batches,
        )
        check_internal_consistency(T)


def test_to_and_from_matrix(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Reshape random tensors into matrices and back, check that this leaves
    them invariant.
    """
    for iter_num in range(n_iters):
        T = rtensor()
        T_orig = T.copy()
        # Partition the indices of T into two sets, i_list and its complement.
        n = np.random.randint(low=0, high=len(T.shape) + 1)
        if n:
            i_list = list(
                np.random.choice(len(T.shape), size=n, replace=False)
            )
        else:
            i_list = []
        i_list_compl = sorted(set(range(len(T.shape))) - set(i_list))
        # Reshape T into a matrix.
        (
            T_matrix,
            T_transposed_shape,
            T_transposed_qhape,
            T_transposed_dirs,
        ) = T.to_matrix(
            i_list, i_list_compl, return_transposed_shape_data=True
        )
        assert (T == T_orig).all()
        T = T_matrix
        check_internal_consistency(T)

        # Permute the indices of T_orig as they were permuted by to_matrix.
        T_orig = T_orig.transpose(i_list + i_list_compl)
        assert T_transposed_shape == T_orig.shape

        # Reshape the matrix back into a tensor.
        l_dims = T_transposed_shape[: len(i_list)]
        r_dims = T_transposed_shape[len(i_list) :]
        if T_transposed_qhape is not None:
            l_qims = T_transposed_qhape[: len(i_list)]
            r_qims = T_transposed_qhape[len(i_list) :]
        else:
            l_qims = None
            r_qims = None
        if T_transposed_dirs is not None:
            l_dirs = T_transposed_dirs[: len(i_list)]
            r_dirs = T_transposed_dirs[len(i_list) :]
        else:
            l_dirs = None
            r_dirs = None
        T_matrix = T.copy()
        T_tensor = T.from_matrix(
            l_dims,
            r_dims,
            left_qims=l_qims,
            right_qims=r_qims,
            left_dirs=l_dirs,
            right_dirs=r_dirs,
        )
        assert (T == T_matrix).all()
        T = T_tensor
        check_internal_consistency(T)
        # Check that we are back where we started.
        assert (T == T_orig).all()


def test_diag_vector_to_matrix(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Generate a random vector, turn it into a diagonal matrix, compare
    against NumPy.
    """
    for iter_num in range(n_iters):
        T = rtensor(n=1, invar=False)
        T_np = T.to_ndarray()
        T_diag = T.diag()
        T_np_diag = np.diag(T_np)
        T_np_diag = type(T).from_ndarray(
            T_np_diag,
            shape=T_diag.shape,
            qhape=T_diag.qhape,
            dirs=T_diag.dirs,
            charge=T_diag.charge,
        )
        assert T_np_diag.allclose(T_diag)


def test_diag_matrix_to_vector(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Generate a random matrix, extract its diagonal, compare against NumPy.
    """
    for iter_num in range(n_iters):
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
        T_np_diag = type(T).from_ndarray(
            T_np_diag,
            shape=T_diag.shape,
            qhape=T_diag.qhape,
            dirs=T_diag.dirs,
            charge=T_diag.charge,
            invar=False,
        )
        assert T_np_diag.allclose(T_diag)


def test_trace(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Generate a random tensor with at least two indices, trace over two of
    them, and compare against NumPy.
    """
    for iter_num in range(n_iters):
        shp = rshape(nlow=2)
        qhp = rqhape(shape=shp)
        dirs = rdirs(shape=shp)
        charge = rcharge()
        i = np.random.randint(low=0, high=len(shp))
        j = np.random.randint(low=0, high=len(shp))
        while i == j:
            j = np.random.randint(low=0, high=len(shp))
        shp[j] = shp[i]
        dirs[j] = -dirs[i]
        qhp[j] = qhp[i]
        T = rtensor(shape=shp, qhape=qhp, dirs=dirs, charge=charge)
        T_np = T.to_ndarray()
        tr = T.trace(axis1=i, axis2=j)
        np_tr = np.trace(T_np, axis1=i, axis2=j)
        check_internal_consistency(tr)
        np_tr = type(T).from_ndarray(
            np_tr,
            shape=tr.shape,
            qhape=tr.qhape,
            dirs=tr.dirs,
            charge=tr.charge,
        )
        assert np_tr.allclose(tr)


def test_multiply_diag(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Generate a random tensor and diagonal matrix, use multiply_diag to
    multiply the them, compare against NumPy.
    """
    for iter_num in range(n_iters):
        # Generate a random tensor, and index to multiply on, and whether to do
        # it from the right or the left.
        T = rtensor(nlow=1, chilow=1)
        T_orig = T.copy()
        i = np.random.randint(low=0, high=len(T.shape))
        right = np.random.randint(low=0, high=2)

        # Generate the random vector.
        D_shape = [T.shape[i]]
        D_qhape = None if T.qhape is None else [T.qhape[i]]
        D_dirs = None if T.dirs is None else [T.dirs[i] * (1 - 2 * right)]
        D = rtensor(
            shape=D_shape, qhape=D_qhape, dirs=D_dirs, invar=False, charge=0
        )

        # Do the multiplication using NumPy.
        T_np = T.to_ndarray()
        D_np = D.to_ndarray()
        prod_np = np.tensordot(T_np, np.diag(D_np), (i, 1 - right))
        # Permute the index back to its original place.
        perm = list(range(len(prod_np.shape)))
        d = perm.pop(-1)
        perm.insert(i, d)
        prod_np = np.transpose(prod_np, perm)

        # Compare multiply_diag to NumPy.
        direction = "right" if right else "left"
        TD = T.multiply_diag(D, i, direction=direction)
        assert (T == T_orig).all()
        T = TD
        check_internal_consistency(T)
        assert np.allclose(T.to_ndarray(), prod_np)


def test_product_invariant(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Generate two invariant tensors, contract them over a random set of legs,
    and compare with NumPy.
    """
    for iter_num in range(n_iters):
        shp1 = rshape(nlow=1)  # Shape of the first tensor
        # Choose how many indices to contract order, and which indices of
        # tensor #1 those should be.
        n = np.random.randint(low=1, high=len(shp1) + 1)
        if n:
            i_list = list(np.random.choice(len(shp1), size=n, replace=False))
        else:
            i_list = []
        # Generate the shape of the second tensor, and which indices it should
        # be contracted over.
        shp2 = rshape(nlow=n)
        if n:
            j_list = list(np.random.choice(len(shp2), size=n, replace=False))
        else:
            j_list = []
        # Make sure contracted indices have a dimension of at least 1.
        for k in range(n):
            dim1 = shp1[i_list[k]]
            if np.sum(dim1) < 1:
                dim1 = rshape(n=1, chilow=1)[0]
                shp1[i_list[k]] = dim1
            shp2[j_list[k]] = dim1
        # Generate tensor #1.
        qhp1 = rqhape(shp1)
        qhp2 = rqhape(shp2)
        if qhp1 is not None:
            for k in range(n):
                qhp2[j_list[k]] = qhp1[i_list[k]]
        T1 = rtensor(shape=shp1, qhape=qhp1)
        T1_orig = T1.copy()
        # Generate tensor #2.
        if T1.dirs is not None:
            dirs2 = rdirs(shape=shp2)
            for i, j in zip(i_list, j_list):
                dirs2[j] = -T1.dirs[i]
        else:
            dirs2 = None
        T2 = rtensor(shape=shp2, qhape=qhp2, dirs=dirs2)
        T2_orig = T2.copy()
        # Do the product.
        T1_np = T1.to_ndarray()
        T2_np = T2.to_ndarray()
        T = T1.dot(T2, (i_list, j_list))
        assert (T1 == T1_orig).all()
        assert (T2 == T2_orig).all()
        check_internal_consistency(T)
        # Assert that the result has the right shape.
        i_list_compl = sorted(set(range(len(shp1))) - set(i_list))
        j_list_compl = sorted(set(range(len(shp2))) - set(j_list))
        product_shp = [shp1[i] for i in i_list_compl] + [
            shp2[j] for j in j_list_compl
        ]
        if type(T) == Tensor:
            product_shp = Tensor.flatten_shape(product_shp)
        assert T.shape == product_shp
        # Do the product using NumPy and compare.
        T_np = np.tensordot(T1_np, T2_np, (i_list, j_list))
        assert np.allclose(T_np, T.to_ndarray())


def test_product_noninvariant(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Generate two tensors that are either matrices and vectors. If they are
    vectors, make them *not* be invariant. Contract them and compare with
    NumPy.
    """
    for iter_num in range(n_iters):
        # Generate tensor #1.
        n1 = np.random.randint(1, 3)
        T1 = rtensor(n=n1, chilow=1, invar=(n1 != 1))

        # Generate tensor #2.
        n2 = np.random.randint(1, 3)
        shp2 = rshape(n=n2, chilow=1)
        qhp2 = rqhape(shape=shp2)
        dirs2 = rdirs(shape=shp2)
        c2 = rcharge()
        # The last index of T1 will be contracted with the first index of T2,
        # so make those match.
        shp2[0] = T1.shape[-1]
        if T1.qhape is not None:
            qhp2[0] = T1.qhape[-1]
            dirs2[0] = -T1.dirs[-1]
        T2 = rtensor(
            shape=shp2, qhape=qhp2, dirs=dirs2, charge=c2, invar=(n2 != 1)
        )

        # Do the product and compare.
        T1_orig = T1.copy()
        T2_orig = T2.copy()
        check_internal_consistency(T1)
        check_internal_consistency(T2)
        T1_np = T1.to_ndarray()
        T2_np = T2.to_ndarray()
        T = T1.dot(T2, (n1 - 1, 0))
        assert (T1 == T1_orig).all()
        assert (T2 == T2_orig).all()
        check_internal_consistency(T)
        T_np = np.tensordot(T1_np, T2_np, (n1 - 1, 0))
        assert np.allclose(T_np, T.to_ndarray())


# We test SVD with a tiny amount of truncation and substantial amoun of
# truncation, as well as enforcing full bond dimension with chis. We do not
# test eps=0, because there may be singular values that are 0 by symmetry,
# which makes comparing with NumPy unfair.
@pytest.mark.parametrize("eps", [1e-15, 1e-3])
@pytest.mark.parametrize("truncate", [True, False])
def test_svd(
    n_iters,
    tensorclass,
    n_qnums,
    rshape,
    rqhape,
    rdirs,
    rcharge,
    rtensor,
    eps,
    truncate,
):
    """Generate a random  tensor, SVD it with or without truncation, and
    reconstruct it from the SVD. Check that the reconstructing matches up to
    truncation error, that truncation error is correctly reported, and compare
    with NumPy.
    """
    if truncate:
        chi = np.random.randint(low=1, high=6)
        chis = list(range(chi + 1))
    else:
        chis = None
    for iter_num in range(n_iters):
        # Generate a random tensor with at least two indices, and partition the
        # indices into two groups.
        T = rtensor(nlow=2, chilow=1)
        T_orig = T.copy()
        T_np = T.to_ndarray()
        n = np.random.randint(low=1, high=len(T.shape))
        if n:
            i_list = list(
                np.random.choice(len(T.shape), size=n, replace=False)
            )
        else:
            i_list = []
        i_list_compl = sorted(set(range(len(T.shape))) - set(i_list))
        np.random.shuffle(i_list_compl)

        # Do the SVD and compare the U*S*V to T.
        U, S, V, rel_err = T.svd(
            i_list, i_list_compl, chis=chis, eps=eps, return_rel_err=True
        )
        assert (T == T_orig).all()
        check_internal_consistency(U)
        check_internal_consistency(S)
        check_internal_consistency(V)
        US = U.dot(S.diag(), (len(i_list), 0))
        USV = US.dot(V, (len(i_list), 0))
        err = (USV - T.transpose(i_list + i_list_compl)).norm()
        T_norm = T.norm()
        if T_norm != 0:
            true_rel_err = err / T_norm
        else:
            true_rel_err = 0
        # Check that the reported error is the same as the actual error.
        # Allow a mismatch up to 1e-7, because a square root brings machine
        # epsilon to around 1e-8
        assert np.abs(rel_err - true_rel_err) < 1e-7
        # If we did not use the full bond dimension allowd, the error incurred
        # should be smaller than eps.
        assert rel_err <= eps or sum(type(S).flatten_shape(S.shape)) == chi

        # Do the same SVD with NumPy and compare.
        U_np_svd, S_np_svd, V_np_svd, np_rel_err = svd(
            T_np,
            i_list,
            i_list_compl,
            chis=chis,
            eps=eps,
            return_rel_err=True,
        )
        U_svd_np, S_svd_np, V_svd_np = (
            U.to_ndarray(),
            S.to_ndarray(),
            V.to_ndarray(),
        )
        order = np.argsort(-S_svd_np)
        S_svd_np = S_svd_np[order]
        U_svd_np = U_svd_np[..., order]
        V_svd_np = V_svd_np[order, ...]
        # There's a gauge freedom in SVD, so find the gauge transformation that
        # maps between U_svd_np and U_np_svd, and revert that transformation.
        g = np.tensordot(U_svd_np.conjugate(), U_np_svd, (range(n), range(n)))
        U_svd_np = np.tensordot(U_svd_np, g, ([-1], [0]))
        V_svd_np = np.tensordot(g.conjugate(), V_svd_np, ([0], [0]))
        # Check that the gauge transformation commutes with the matrix of
        # singular values.
        S_np_mat = np.diag(S_np_svd)
        assert np.allclose(np.dot(g, S_np_mat), np.dot(S_np_mat, g))
        assert np.allclose(U_np_svd, U_svd_np)
        assert np.allclose(S_np_svd, S_svd_np)
        assert np.allclose(V_np_svd, V_svd_np)
        # atol=1e-7 because a square root brings machine epsilon to around 1e-8
        assert np.allclose(rel_err, np_rel_err, atol=1e-7)


@pytest.mark.parametrize("eps", [1e-15, 1e-3])
@pytest.mark.parametrize("truncate", [True, False])
@pytest.mark.parametrize("hermitian", [True, False])
def test_eig(
    n_iters,
    tensorclass,
    n_qnums,
    rshape,
    rqhape,
    rdirs,
    rcharge,
    rtensor,
    eps,
    truncate,
    hermitian,
):
    if truncate:
        chi = np.random.randint(low=1, high=6)
        chis = list(range(chi + 1))
    else:
        chis = None
    for iter_num in range(n_iters):
        # Generate a tensor that is square when indices in i_list and
        # i_list_compl are joined.
        n = np.random.randint(low=1, high=3)
        shp = rshape(n=n * 2, chilow=1, chihigh=4)
        qhp = rqhape(shape=shp)
        dirs = [1] * len(shp)
        i_list = list(np.random.choice(len(shp), size=n, replace=False))
        i_list_compl = sorted(set(range(len(shp))) - set(i_list))
        np.random.shuffle(i_list_compl)
        for i, j in zip(i_list, i_list_compl):
            shp[j] = shp[i].copy()
            qhp[j] = qhp[i].copy()
            dirs[j] = -1
        T = rtensor(shape=shp, qhape=qhp, dirs=dirs, charge=0)
        if hermitian:
            T_transpose = T.copy()
            for i, j in zip(i_list, i_list_compl):
                T_transpose = T_transpose.swapaxes(i, j)
            T = T + T_transpose.conjugate()
        T_orig = T.copy()
        T_np = T.to_ndarray()

        # Find eigenvalues and vectors.
        S, U, rel_err = T.eig(
            i_list,
            i_list_compl,
            eps=eps,
            chis=chis,
            hermitian=hermitian,
            return_rel_err=True,
        )
        assert (T == T_orig).all()
        check_internal_consistency(S)
        check_internal_consistency(U)
        S_eig_np, U_eig_np = S.to_ndarray(), U.to_ndarray()

        # Do the same SVD with NumPy and compare.
        S_np_eig, U_np_eig, rel_err_np = eig(
            T_np,
            i_list,
            i_list_compl,
            chis=chis,
            eps=eps,
            hermitian=hermitian,
            return_rel_err=True,
        )
        order = np.argsort(-S_eig_np)
        S_eig_np = S_eig_np[order]
        U_eig_np = U_eig_np[..., order]
        order = np.argsort(-S_np_eig)
        S_np_eig = S_np_eig[order]
        U_np_eig = U_np_eig[..., order]
        # There's a gauge freedom in the decomposition (for instance, phases of
        # eigenvectors), so find the gauge transformation that maps between
        # U_svd_np and U_np_svd, and revert that transformation.
        g = np.tensordot(U_eig_np.conjugate(), U_np_eig, (range(n), range(n)))
        # We should only transform vectors within subspaces corresponding to
        # degenerate eigenvalues, so enforce g to be 0 outside those blocks.
        SX, SY = np.meshgrid(S_np_eig, S_np_eig)
        degeneracy_eps = 1e-6
        fltr = np.exp(-(abs(SX - SY) ** 2) / degeneracy_eps)
        g = g * fltr
        U_eig_np = np.tensordot(U_eig_np, g, ([-1], [0]))
        assert np.allclose(S_np_eig, S_eig_np)
        assert np.allclose(U_np_eig, U_eig_np)
        assert np.allclose(rel_err, rel_err_np)
        # If we did not use the full bond dimension allowd, the error incurred
        # should be smaller than eps.
        assert rel_err < eps or sum(type(S).flatten_shape(S.shape)) == chi

        # If the tensor was Hermitian, we should be able to reconstruct the
        # original tensor as U*S*U^dagger.
        if hermitian:
            l = len(U.shape)
            Udg_permutation = (l - 1,) + tuple(range(l - 1))
            Udg = U.conjugate().transpose(Udg_permutation)
            US = U.dot(S.diag(), (len(i_list), 0))
            USUdg = US.dot(Udg, (len(i_list), 0))
            err = (USUdg - T.transpose(i_list + i_list_compl)).norm()
            T_norm = T.norm()
            if T_norm != 0:
                true_rel_err = err / T_norm
            else:
                true_rel_err = 0
            # Check that the reported error is the same as the actual error.
            # Allow a mismatch up to 1e-7, because a square root brings machine
            # epsilon to around 1e-8
            assert np.abs(rel_err - true_rel_err) < 1e-7


def test_split(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Use both `split` and SVD to decompose a random tensor, and compare the
    results.
    """
    for iter_num in range(n_iters):
        # Generate a random tensor with at least two indices, and a random
        # bipartition of its indices.
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

        # Use both SVD and `split` to decompose the tensor with a random amount
        # of truncation, check that the results match.
        chi = np.random.randint(low=1, high=10)
        eps = 10 ** (-1 * np.random.randint(low=2, high=10))
        svd_res = T.svd(i_list, i_list_compl, chis=chi, eps=eps)
        assert (T == T_orig).all()
        U, S, V = svd_res[0:3]
        check_internal_consistency(U)
        check_internal_consistency(S)
        check_internal_consistency(V)
        mid = S.sqrt().diag()
        US = U.dot(mid, (len(i_list), 0))
        SV = mid.dot(V, (1, 0))
        split_res = T.split(
            i_list, i_list_compl, chis=chi, eps=eps, return_sings=True
        )
        assert US.allclose(split_res[0])
        assert S.allclose(split_res[1])
        assert SV.allclose(split_res[2])


def test_norm(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Compute the Frobenius norm of a tensor, compare with NumPy."""
    for iter_num in range(n_iters):
        T = rtensor()
        T_np = T.to_ndarray()
        T_norm = T.norm()
        n = len(T.shape)
        all_inds = tuple(range(n))
        T_np_norm = np.sqrt(
            np.tensordot(T_np, T_np.conj(), (all_inds, all_inds))
        )
        assert np.allclose(T_norm, T_np_norm)


def test_max(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Get the maximum element of a real tensor, compare with NumPy."""
    for iter_num in range(n_iters):
        T = rtensor(chilow=1, cmplx=False)
        T_np = T.to_ndarray()
        T_max = T.max()
        T_np_max = np.max(T_np)
        assert T_max == T_np_max


def test_min(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Get the minimum element of a real tensor, compare with NumPy."""
    for iter_num in range(n_iters):
        T = rtensor(chilow=1, cmplx=False)
        T_np = T.to_ndarray()
        T_min = T.min()
        T_np_min = np.min(T_np)
        assert T_min == T_np_min


def test_average(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Get the average element of a real tensor, compare with NumPy."""
    for iter_num in range(n_iters):
        T = rtensor(chilow=1)
        T_np = T.to_ndarray()
        T_average = T.average()
        T_np_average = np.average(T_np)
        assert np.allclose(T_average, T_np_average)


def test_expand_dim(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Use `expand_dim` to a give a random tensor a trivial extra index, and
    compare the outcome with NumPy.
    """
    for iter_num in range(n_iters):
        T = rtensor()
        T_orig = T.copy()
        axis = np.random.randint(0, high=len(T.shape) + 1)
        T_np = T.to_ndarray()
        T_expanded = T.expand_dims(axis)
        assert (T == T_orig).all()
        T = T_expanded
        check_internal_consistency(T)
        T_np = np.expand_dims(T_np, axis)
        T_np_T = type(T).from_ndarray(
            T_np, shape=T.shape, qhape=T.qhape, dirs=T.dirs, charge=T.charge,
        )
        check_internal_consistency(T_np_T)
        assert T.allclose(T_np_T)


def test_eye(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Create an identity matrix, compare with NumPy."""
    for iter_num in range(n_iters):
        dim = rshape(n=1)[0]
        qim = rqhape(shape=[dim])[0]
        T = tensorclass.eye(dim, qim=qim)
        T_np = np.eye(T.flatten_dim(dim))
        T_np = type(T).from_ndarray(
            T_np, shape=T.shape, qhape=T.qhape, dirs=T.dirs, charge=T.charge,
        )
        assert (T == T_np).all()


def test_flip_dir(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Test that flipping the direction of an index twice is a noop."""
    for iter_num in range(n_iters):
        T = rtensor(nlow=1)
        T_orig = T.copy()
        i = np.random.randint(low=0, high=len(T.shape))
        T_flipped = T.flip_dir(i)
        assert (T == T_orig).all()
        check_internal_consistency(T_flipped)
        T_flipped = T_flipped.flip_dir(i)
        assert (T == T_flipped).all()


def test_expand_dims_product(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Test contracting over a trivial index created with `expand_dims` by
    comparing with NumPy.
    """
    for iter_num in range(n_iters):
        T1 = rtensor()
        T2 = rtensor()
        axis1 = np.random.randint(0, high=len(T1.shape) + 1)
        axis2 = np.random.randint(0, high=len(T2.shape) + 1)
        T1_np = T1.to_ndarray()
        T2_np = T2.to_ndarray()
        T1 = T1.expand_dims(axis1, direction=1)
        T2 = T2.expand_dims(axis2, direction=-1)
        T1_np = np.expand_dims(T1_np, axis1)
        T2_np = np.expand_dims(T2_np, axis2)
        T = T1.dot(T2, (axis1, axis2))
        check_internal_consistency(T)
        T_np = np.tensordot(T1_np, T2_np, (axis1, axis2))
        T_np_T = type(T).from_ndarray(
            T_np, shape=T.shape, qhape=T.qhape, dirs=T.dirs, charge=T.charge,
        )
        check_internal_consistency(T_np_T)
        assert T.allclose(T_np_T)


def test_ncon_svd_ncon(
    n_iters, tensorclass, n_qnums, rshape, rqhape, rdirs, rcharge, rtensor
):
    """Create a random NCon contraction, do it, and compare to doing the same
    contraction with `ndarrays`. If the result has more than one index, SVD it,
    and use the SVD to compute its norm squared with another NCon contraction.

    The point of this test is to mimic a typical sequence in a tensor network
    algorithm, where decompositions and contractions follow each other.
    """
    for iter_num in range(n_iters):
        # Create random form data for a random number of tensors (at most 4),
        # and make a set that lists all tuples of ``(i, j)`` where `i` numbers
        # tensors and `j` numbers the indices of the `i`th tensor.
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
                indices.add((i, j))

        # Give each index of each tensor a negative number that will be its
        # ncon contraction number if it left uncontracted.
        ncon_lists = []
        index_numbers = set(range(-len(indices), 0))
        for shp in shapes:
            ncon_list = []
            for index in shp:
                ncon_list.append(index_numbers.pop())
            ncon_lists.append(ncon_list)

        # Pick a random number of pairs of indices to be contracted, give them
        # the same, positive index number, and change their form data to match
        # so that they can be contracted with each other.
        n_contractions = np.random.randint(
            low=0, high=int(len(indices) / 2) + 1
        )
        for counter in range(1, n_contractions + 1):
            t1, i1 = indices.pop()
            t2, i2 = indices.pop()
            shapes[t2][i2] = shapes[t1][i1]
            qhapes[t2][i2] = qhapes[t1][i1]
            dirss[t2][i2] = -dirss[t1][i1]
            ncon_lists[t1][i1] = counter
            ncon_lists[t2][i2] = counter

        # Create the tensors according to the form data we now have.
        tensors = []
        np_tensors = []
        for shape, qhape, dirs, charge in zip(shapes, qhapes, dirss, charges):
            tensor = rtensor(shape, qhape=qhape, dirs=dirs, charge=charge)
            np_tensor = tensor.to_ndarray()
            tensors.append(tensor)
            np_tensors.append(np_tensor)

        # Do the contraction. Compare with doing it with NumPy arrays.
        T = ncon(tensors, ncon_lists)
        check_internal_consistency(T)
        np_T = ncon(np_tensors, ncon_lists)
        np_T = type(T).from_ndarray(
            np_T, shape=T.shape, qhape=T.qhape, dirs=T.dirs, charge=T.charge,
        )
        assert T.allclose(np_T)

        if len(T.shape) > 1:
            # SVD the result of the contraction
            n_svd_inds = np.random.randint(low=1, high=len(T.shape))
            if n_svd_inds:
                i_list = list(
                    np.random.choice(
                        len(T.shape), size=n_svd_inds, replace=False
                    )
                )
            else:
                i_list = []
            i_list_compl = sorted(set(range(len(T.shape))) - set(i_list))
            np.random.shuffle(i_list_compl)
            U, S, V = T.svd(i_list, i_list_compl, eps=1e-15)

            # ncon U, S and V with S to get the norm_sq of S.
            S_diag = S.diag().conjugate()
            U = U.conjugate()
            V = V.conjugate()
            U_left_inds = [i + 1 for i in i_list]
            V_right_inds = [j + 1 for j in i_list_compl]
            norm_sq_ncon = ncon(
                (T, U, S_diag, V),
                (
                    list(range(1, len(T.shape) + 1)),
                    U_left_inds + [100],
                    [100, 101],
                    [101] + V_right_inds,
                ),
            ).value()
            norm_sq_S = S.norm_sq()
            norm_sq = T.norm_sq()
            # Check that different ways of computing the norm all give the same
            # result.
            assert np.allclose(norm_sq, norm_sq_ncon)
            assert np.allclose(norm_sq, norm_sq_S)
