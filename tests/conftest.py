"""Define utility functions and set up pytest fixtures for the test suite."""
import pytest
import numpy as np
from abeliantensors import Tensor
from abeliantensors import TensorZ2, TensorU1, TensorZ3


@pytest.fixture(scope="module")
def tensorclass(request):
    """A pytest fixture that returns the tensor class currently being tested.
    """
    return request.param


def pytest_addoption(parser):
    """Add command line options for setting the tensorclass(es) to test and
    the number of times to repeat each test.
    """
    parser.addoption(
        "--tensorclass",
        action="append",
        default=[],
        help="Tensor class(es) to run tests on.",
    )
    parser.addoption(
        "--n_iters",
        type=int,
        default=100,
        help="Number of times to run each test on new random input.",
    )


def parse_tensorclass(s):
    """Take a string representing a tensorclass, such as "TensorZ2", and return
    the corresponding class.
    """
    s = s.lower().strip()
    if s == "tensor":
        return Tensor
    elif s == "tensorz2":
        return TensorZ2
    elif s == "tensorz3":
        return TensorZ3
    elif s == "tensoru1":
        return TensorU1
    else:
        msg = "Unknown tensor class name: {}".format(s)
        raise ValueError(msg)


def pytest_generate_tests(metafunc):
    """Set up passing the command line arguments for n_iters and for the
    tensorclass fixture, and give the latter's default value.
    """
    default_classes = ["Tensor", "TensorZ2", "TensorU1", "TensorZ3"]
    tensorclass_opts = (
        metafunc.config.getoption("tensorclass") or default_classes
    )
    tensorclasses = map(parse_tensorclass, tensorclass_opts)
    if "tensorclass" in metafunc.fixturenames:
        metafunc.parametrize("tensorclass", tensorclasses, indirect=True)
    n_iters = metafunc.config.getoption("n_iters")
    if "n_iters" in metafunc.fixturenames:
        metafunc.parametrize("n_iters", [n_iters])


@pytest.fixture
def n_qnums(tensorclass):
    """Return the number of different possible quantum numbers for the given
    tensorclass, or None if the answer is infinite or undefined.
    """
    if tensorclass == TensorZ2:
        n_qnums = 2
    elif tensorclass == TensorZ3:
        n_qnums = 3
    else:
        n_qnums = None
    return n_qnums


@pytest.fixture
def rshape(n_qnums):
    """Return a function that generates random shapes for symmetric tensors."""

    def _rshape(n=None, chi=None, nlow=0, nhigh=5, chilow=0, chihigh=3):
        """Return a random shape for a symmetric tensor.

        `n` is the number of indices, `chi` is the bond dimension of each
        symmetry sector. If either one is None, they are random generated using
        the bounds `nlow`, `nhigh`, `chilow`, and `chihigh`.
        """
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
                dim = [chi] * dim_dim
            s = sum(dim)
            if s < chilow:
                j = np.random.randint(0, len(dim))
                dim[j] += chilow - s
            shape.append(dim)
        return shape

    return _rshape


@pytest.fixture
def rqhape(n_qnums):
    """Return a function that generates random `qhape`s for symmetric tensors.
    """

    def _rqhape(shape, qlow=-3, qhigh=3):
        """Return a random `qhape` for a symmetric tensor.

        `shape` is the corresponding shape. When generating a `qhape` for
        the `TensorZN` class, quantum numbers are randomly picked from
        `range(0, n_qnums)`. For `TensorU1`, they are randomly picked from the
        `range(qlow, qhigh+1)`.
        """
        if n_qnums is not None:
            qlow = 0
            qhigh = n_qnums - 1
        try:
            assert all(len(dim) <= qhigh - qlow + 1 for dim in shape)
        except TypeError:
            # This happens when dim is a single number.
            pass
        possible_qnums = range(qlow, qhigh + 1)
        try:
            qhape = [
                list(np.random.choice(possible_qnums, len(dim), replace=False))
                for dim in shape
            ]
        except TypeError:
            # This happens when dim is a single number.
            qhape = None
        return qhape

    return _rqhape


@pytest.fixture
def rdirs():
    """Return a function that generates random `dirs`s for symmetric tensors.
    """

    def _rdirs(shape=None, length=None):
        """Return a random `dirs` for a symmetric tensor.

        `shape` is the corresponding shape. Alternatively, just the length of
        of the `dirs` can be passed.
        """
        if shape is not None:
            length = len(shape)
        dirs = np.random.randint(low=0, high=2, size=length)
        dirs = list(2 * dirs - 1)
        return dirs

    return _rdirs


@pytest.fixture
def rcharge():
    """Return a function that generates random charges for symmetric tensors.
    """

    def _rcharge(low=0, high=4):
        """Return a random charge for a symmetric tensor, i.e. a random integer
        between `low` and `high`, inclusive.
        """
        return np.random.randint(low=low, high=high)

    return _rcharge


@pytest.fixture
def rtensor(tensorclass, n_qnums, rqhape, rshape, rdirs, rcharge):
    """Return a function that generates a random tensor of the given
    tensorclass, using the given fixtures for generating form data.
    """

    def _rtensor(
        shape=None,
        qhape=None,
        n=None,
        chi=None,
        nlow=0,
        nhigh=5,
        chilow=0,
        chihigh=6,
        charge=None,
        dirs=None,
        cmplx=True,
        **kwargs
    ):
        """Return a random tensor of the given form data.

        Full form data (`shape`, `qhape`, `dirs`, `charge`) can be provided, in
        which case only the elements of the tensor are random. Alternatively,
        random form data can also be generated. In that case, `n` is the number
        of indices (alternatively a random number between `nlow` and `nhigh`)
        and `chi` is the bond dimension of all sectors (alternatively random
        numbers between `chilow` and `chihigh`).

        Partial form data can also be given. For instance, if one only gives
        the `shape` argument, random `qhape`, `dirs`, and `charge` are
        generated that match it.

        `cmplx` sets whether the tensor should be complex instead of
        real, and is by default True.
        """
        if shape is None:
            shape = rshape(
                n=n,
                chi=chi,
                nlow=nlow,
                nhigh=nhigh,
                chilow=chilow,
                chihigh=chihigh,
            )
        if qhape is None:
            qhape = rqhape(shape)
        if dirs is None:
            dirs = rdirs(shape=shape)
        elif dirs == 1:
            dirs = [1] * len(shape)
        if charge is None:
            charge = rcharge()

        real = tensorclass.random(
            shape, qhape=qhape, dirs=dirs, charge=charge, **kwargs
        )
        if cmplx:
            imag = tensorclass.random(
                shape, qhape=qhape, dirs=dirs, charge=charge, **kwargs
            )
            res = real + 1j * imag
        else:
            res = real
        return res

    return _rtensor
