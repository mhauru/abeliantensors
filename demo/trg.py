import numpy as np
from ncon import ncon


def trg_step(A, log_fact, chi, eps=1e-7):
    """Take a TRG step.

    A is a tensor and log_fact is a scalar factor, so that A*exp(log_fact) is
    the tensor that defines the model. The log_fact bit is necessary to keep
    the norm of A reasonable and avoid numerical erros and overflows. chi is
    the maximum allowed bond dimension, eps is a truncation error threshold.

    The index numbering convention for A is like this:
       2
       │
    1──A──0
       │
       3

    Returns A_new, log_fact_new, err
    where A_new and log_fact_new define the same model but now coarse-grained
    by a factor of sqrt(2), and err is the sum of truncation errors incurred.
    """
    # Split A in two different ways, diagonally.
    NW, SE, err1 = A.split(
        [1, 2], [0, 3], chis=chi, eps=eps, return_rel_err=True
    )
    NE, SW, err2 = A.split(
        [0, 2], [1, 3], chis=chi, eps=eps, return_rel_err=True
    )
    # Contract the pieces to form the new tensor.
    A_new = ncon(
        (SE, SW, NE, NW), ([-2, 1, 11], [-3, 1, 12], [2, 11, -4], [2, 12, -1])
    )
    # Normalize
    n = A_new.norm()
    if n > 0:
        A_new /= n
        log_fact_new = 2 * log_fact + np.log(n)
    else:
        log_fact_new = 2 * log_fact

    err = err1 + err2
    return A_new, log_fact_new, err
