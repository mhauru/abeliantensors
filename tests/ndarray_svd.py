import numpy as np
import collections


def svd(T, a, b, chis=None, eps=0, print_errors=0,
        return_rel_err=False, break_degenerate=False, degeneracy_eps=1e-6):
    """ Reshapes the tensor T have indices a on one side and indices b
    on the other, SVDs it as a matrix and reshapes the parts back to the
    original form. a and b should be iterables of integers that number
    the indices of T. 

    The optional argument chis is a list of bond dimensions. The SVD is
    truncated to one of these dimensions chi, meaning that only chi
    largest singular values are kept. If chis is a single integer
    (either within a singleton list or just as a bare integer) this
    dimension is used. If no eps is given, the largest value in chis is
    used. Otherwise the smallest chi in chis is used, such that the
    relative error made in the truncation is smaller than eps.

    An exception to the above is degenerate singular values. By default
    truncation is never done so that some singular values are included
    while others of the same value are left out. If this is about to
    happen chi is decreased so that none of the degenerate singular
    values is included. This default behavior can be changed with the
    keyword argument break_degenerate=True. The default threshold for
    when singular values are considered degenerate is 1e-6. This can be
    changed with the keyword argument degeneracy_eps.

    If print_errors > 0 truncation errors are printed, with more
    information as print_errors increases through 1,2 and 3.

    By default the function returns the tuple (U,s,V), where in matrix
    terms U.diag(s).V = T, where the equality is appromixate if there is
    truncation. If return_rel_err = True a fourth value is returned,
    which is the ratio sum_of_discarded_singular_values /
    sum_of_all_singular_values.
    """

    # We want to deal with lists, not tuples or bare integers
    if isinstance(a, collections.Iterable):
        a = list(a)
    else:
        a = [a]
    if isinstance(b, collections.Iterable):
        b = list(b)
    else:
        b = [b]
    assert(len(a) + len(b) == len(T.shape))

    # Permute the indices of T to the right order
    perm = tuple(a+b)
    T_matrix = np.transpose(T, perm)
    # The lists shp_a and shp_b list the dimensions of the bonds in a and b
    shp = T_matrix.shape
    shp_a = shp[:len(a)]
    shp_b = shp[-len(b):]

    # Compute the dimensions of the the matrix that will be formed when
    # indices of a and b are joined together.
    dim_a = 1
    for s in shp_a:
        dim_a = dim_a * s
    dim_b = 1
    for s in shp_b:
        dim_b = dim_b * s
    # Create the matrix and SVD it.
    T_matrix = np.reshape(T_matrix, (dim_a, dim_b))
    U, s, V = np.linalg.svd(T_matrix, full_matrices=False)

    # Format the truncation parameters to canonical form.
    if chis is None:
        max_dim = min(dim_a, dim_b)
        if eps > 0:
            # Try all possible chis.
            chis = list(range(max_dim+1))
        else:
            chis = [max_dim]
    else:
        if isinstance(chis, collections.Iterable):
            chis = list(chis)
        else:
            chis = [chis]
        if eps == 0:
            chis = [max(chis)]
        else:
            chis = sorted(chis)

    # Truncate, if truncation dimensions are given.
    s_sq = s**2
    sum_all_sq = sum(s_sq)
    # Find the smallest chi for which the error is small enough.
    # If none is found, use the largest chi.
    for chi in chis:
        if not break_degenerate:
            # Make sure that we don't break degenerate singular
            # values by including one but not the other.
            while 0 < chi < len(s_sq):
                last_eig_in = s_sq[chi-1]
                last_eig_out = s_sq[chi]
                rel_diff = np.abs(last_eig_in-last_eig_out)/last_eig_in
                if rel_diff < degeneracy_eps:
                    chi -= 1
                else:
                    break
        sum_disc_sq = sum((s_sq)[chi:])
        if sum_all_sq != 0:
            rel_err_sq = sum_disc_sq/sum_all_sq
        else:
            rel_err_sq = 0
        if rel_err_sq <= eps**2:
            break
    # Truncate
    s = s[:chi]
    U = U[:,:chi]
    V = V[:chi,:]

    if print_errors > 0:
        print('----- Tensor SVD error -----')
        print('- Relative truncation error (Frobenius norm) from the '
              'singular values: %.3e' % np.sqrt(rel_err_sq))
        if print_errors > 3:
            print('- sum(|s^2_discarded|): %.3e' % sum_disc_sq)
            print('- sum(|s^2_all|): %.3e' % sum_all_sq)
        if print_errors > 2:
            # Reconstruct the original T_matrix with the truncated
            # U, s, V and print the error in the reconstruction. The
            # error is computed as the Frobenius norm so it should
            # match the ratio of the discarded eigenvalues printed
            # earlier.
            reco = np.einsum('ij,j->ij', U, s)
            reco = np.dot(reco, V)
            reco_err = np.linalg.norm(T_matrix-reco)
            normalization = np.linalg.norm(T_matrix)
            reco_rel_err = reco_err/normalization
            print('- Relative truncation error (Frobenius norm) from a '
                  'reconstruction: %.3e' % reco_rel_err)
        print('----------------------------')

    # Reshape U and V to tensors with shapes matching the shape of T and
    # return.
    U_tens = np.reshape(U, shp_a + (-1,))
    V_tens = np.reshape(V, (-1,) + shp_b)
    ret_val = U_tens, s, V_tens
    if return_rel_err:
        ret_val = ret_val + (np.sqrt(rel_err_sq),)
    return ret_val


def eig(T, a, b, chis=None, eps=0, print_errors=0, return_rel_err=False,
        hermitian=False, break_degenerate=False, degeneracy_eps=1e-6):
    """ Like svd, but for finding the eigenvalues and left eigenvectors.
    See the documentation of svd.

    The only notable differences are the keyword option hermitian (by
    default False), that specifies whether the matrix that is obtained
    by reshaping T is known to be hermitian, and the return values,
    which are of the form S, U, (rel_err), where S includes the
    eigenvalues and U[...,i] is the left eigenvector corresponding to
    S[i]. The first legs of U are compatible with the legs b of T.
    """
    # We want to deal with lists, not tuples or bare integers
    if isinstance(a, collections.Iterable):
        a = list(a)
    else:
        a = [a]
    if isinstance(b, collections.Iterable):
        b = list(b)
    else:
        b = [b]
    assert(len(a) + len(b) == len(T.shape))

    # Permute the indices of T to the right order
    perm = tuple(a+b)
    T_matrix = np.transpose(T, perm)
    # The lists shp_a and shp_b list the dimensions of the bonds in a and b
    shp = T_matrix.shape
    shp_a = shp[:len(a)]
    shp_b = shp[-len(b):]

    # Compute the dimensions of the the matrix that will be formed when
    # indices of a and b are joined together.
    dim_a = 1
    for s in shp_a:
        dim_a = dim_a * s
    dim_b = 1
    for s in shp_b:
        dim_b = dim_b * s
    # Create the matrix and eigenvalue decompose it.
    T_matrix = np.reshape(T_matrix, (dim_a, dim_b))
    if hermitian:
        S, U = np.linalg.eigh(T_matrix)
    else:
        S, U = np.linalg.eig(T_matrix)

    order = np.argsort(-np.abs(S))
    S = S[order]
    U = U[:,order]

    # Format the truncation parameters to canonical form.
    if chis is None:
        max_dim = min(dim_a, dim_b)
        if eps > 0:
            # Try all possible chis.
            chis = list(range(max_dim+1))
        else:
            chis = [max_dim]
    else:
        if isinstance(chis, collections.Iterable):
            chis = list(chis)
        else:
            chis = [chis]
        if eps == 0:
            chis = [max(chis)]
        else:
            chis = sorted(chis)

    # Truncate, if truncation dimensions are given.
    S_abs_sq = np.abs(S)**2
    sum_all_abs_sq = sum(S_abs_sq)
    # Find the smallest chi for which the error is small enough.
    # If none is found, use the largest chi.
    for chi in chis:
        if not break_degenerate:
            # Make sure that we don't break degenerate eigenvalues by
            # including one but not the other.
            while 0 < chi < len(S_abs_sq):
                last_eig_in = S_abs_sq[chi-1]
                last_eig_out = S_abs_sq[chi]
                rel_diff = np.abs(last_eig_in-last_eig_out)/last_eig_in
                if rel_diff < degeneracy_eps:
                    chi -= 1
                else:
                    break
        sum_disc_abs_sq = sum((S_abs_sq)[chi:])
        if sum_all_abs_sq != 0:
            rel_err_sq = sum_disc_abs_sq/sum_all_abs_sq
        else:
            rel_err_sq = 0
        if rel_err_sq <= eps**2:
            break
    # Truncate
    S = S[:chi]
    U = U[:,:chi]

    if print_errors > 0:
        print('----- Tensor SVD error -----')
        print('- Relative truncation error (Frobenius norm) from the '
              'singular values: %.3e' % np.sqrt(rel_err_sq))
        if print_errors > 3:
            print('- sum(|S^2_discarded|): %.3e' % sum_disc_abs_sq)
            print('- sum(|S^2_all|): %.3e' % sum_all_abs_sq)

    # Reshape U to a tensor with shape matching the shape of T and
    # return.
    U_tens = np.reshape(U, shp_a + (-1,))
    ret_val = S, U_tens
    if return_rel_err:
        ret_val = ret_val + (np.sqrt(rel_err_sq),)
    return ret_val

