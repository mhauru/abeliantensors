"""A script that runs TRG on the Ising model, and measures performance with and
without explicit symmetries for the tensors.
"""
import numpy as np
import timeit
from scipy import integrate
from ncon import ncon
from matplotlib import pyplot as plt
import seaborn as sns
from abeliantensors import Tensor, TensorZ2
from trg import trg_step


def ising_tensor(J, h, beta):
    """Return as np.ndarray the tensor that defines the classical square
    lattice Ising model at coupling J, external field h, and inverse
    temperature beta.
    """
    # Construct the Hamiltonian
    ham_interaction = np.array([[1, -1], [-1, 1]])
    ham_onsite = np.array([[-1, 0], [0, 1]])
    ham = -J * ham_interaction + h * ham_onsite
    # The tensor is a four matrices of Boltzmann weights, connected by 3-way
    # delta tensors.
    boltz = np.exp(-beta * ham)
    A = np.einsum("ab,bc,cd,da->acdb", boltz, boltz, boltz, boltz)
    # Rotate the indices with a Hadamard transformation, to put them in a basis
    # where the Z2 symmetry is obvious.
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    A = ncon(
        (A, hadamard, hadamard, hadamard, hadamard),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [-3, 3], [-4, 4]),
    )
    return A


def ising_exact_f(J, h, beta):
    """Return the exact free energy per site of the infinite classical square
    lattice Ising model at coupling J, external field h, and inverse
    temperature beta.
    """
    if h != 0:
        msg = "Free energy of Ising model with H != 0 " "not implemented."
        raise NotImplementedError(msg)
    sinh = np.sinh(2 * beta * J)
    cosh = np.cosh(2 * beta * J)

    def integrand(theta):
        res = np.log(
            cosh ** 2
            + sinh ** 2
            * np.sqrt(1 + sinh ** (-4) - 2 * sinh ** (-2) * np.cos(theta))
        )
        return res

    integral, err = integrate.quad(integrand, 0, np.pi)
    f = -(np.log(2) / 2 + integral / (2 * np.pi)) / beta
    return f


def main():
    """The main function, that runs the simulation.

    Here's how this goes: For both Tensor and TensorZ2 classes, so with and
    without explicit symmetries, we construct the tensor for the critical Ising
    model. We run TRG on it for a number of iterations that should be
    sufficient for convergence. We print out the error we find for the free
    energy per site, because why not, even though measuring accuracy isn't the
    goal here. Both Tensor and TensorZ2 should give the exact same number.

    We then use timeit to run one extra iteration of TRG, and time how long it
    takes. We run it a number of times and take the fastest time, to avoid
    spurious timings because of e.g. other processes. The reason we only do
    this for the converged tensor is that we are only interested in the time it
    takes for a TRG iteration once the bond dimension is full, the first few
    iterations would only skew the timings.

    We do the above for many different TRG bond dimensions, and finally make a
    plot of how long it took to run one iteration of TRG as a function of bond
    dimension, both for Tensor and TensorZ2.
    """
    # Couplings for the critical Ising model.
    J = 1.0
    h = 0.0
    beta = np.log(1 + np.sqrt(2)) / (2 * J)
    # Exact free energy to compare to.
    f_exact = ising_exact_f(J, h, beta)
    # Initial tensor to start from, as an ndarray.
    A0_np = ising_tensor(J, h, beta)

    # Bond dimensions to do TRG with.
    chis = range(2, 71, 1)
    # Truncation threshold in TRG
    eps = 1e-7
    # Number of iterations to run for, before starting performance measurement.
    n_iters = 14
    # The tensor classes to compare.
    tensorclasses = (Tensor, TensorZ2)

    # Number of times to run the same iteration when doing timings. The minimum
    # time is taken as the result.
    n_reps = 5

    timings = {}
    for cls in tensorclasses:
        timings[cls] = []
        # Turn the numpy array tensor into a Tensor or TensorZ2 instance.
        A0 = cls.from_ndarray(
            A0_np, shape=[[1, 1]] * 4, qhape=[[0, 1]] * 4, dirs=[1, -1, 1, -1]
        )
        for chi in chis:
            print("Starting chi = {}.".format(chi))
            A = A0
            log_fact = 0
            for i in range(1, n_iters + 1):
                A, log_fact, err = trg_step(A, log_fact, chi, eps)
            # Compute the partition function and free energy per site.
            Z = ncon(A, [1, 1, 2, 2]).value()
            logZ = np.log(Z) + log_fact
            # The normalization is by 2 ** (i + 1) because each step of
            # coarge-graining reduces the number of tensors by a factor
            # of 2, and the original tensor corresponded to two lattice
            # sites.
            f = (-logZ / beta) / 2 ** (i + 1)
            f_err = (f - f_exact) / f_exact
            msg = (
                "Done converging chi = {}. Error in free energy: {:.5e}\n"
                "Starting timeit."
            ).format(chi, f_err)
            print(msg)

            timing = timeit.repeat(
                "trg_step(A, log_fact, chi, eps)",
                setup=("from trg import trg_step"),
                number=1,
                globals=locals(),
                repeat=5,
            )
            timing = sorted(timing)
            msg = "Timings for chi = {}:\n{}".format(chi, timing)
            timings[cls].append(min(timing))

    # The plotting. We use seaborn's default style.
    sns.set_style("darkgrid")
    sns.set_context("paper")
    plt.figure()
    for cls in tensorclasses:
        plt.loglog(chis, timings[cls], label=cls, marker="o", ms=2)
    # For our choice of bond dimensions, the default ticks only consist of
    # [10], so manually set them to something more informative.
    ticks = [min(chis)] + list(range(10, max(chis) + 1, 10))
    plt.xticks(ticks, ticks)
    plt.title("Time for a single TRG iteration, with and without symmetries")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Bond dimension")
    plt.legend()
    plt.savefig("latest_trg_performance.pdf")


if __name__ == "__main__":
    main()
