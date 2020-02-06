import pytest
import numpy as np
from tensors import Tensor
from tensors import TensorZ2, TensorU1, TensorZ3


@pytest.fixture(scope="module")
def tensorclass(request):
    return request.param


def pytest_addoption(parser):
    parser.addoption(
        "--tensorclass",
        action="append",
        default=[],
        help="Tensor class(es) to run tests on.",
    )


def parse_tensorclass(s):
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
    default_opts = ["Tensor", "TensorZ2", "TensorU1", "TensorZ3"]
    tensorclass_opts = metafunc.config.getoption("tensorclass") or default_opts
    tensorclasses = map(parse_tensorclass, tensorclass_opts)
    if "tensorclass" in metafunc.fixturenames:
        metafunc.parametrize("tensorclass", tensorclasses, indirect=True)


@pytest.fixture
def n_qnums(tensorclass):
    if tensorclass == TensorZ2:
        n_qnums = 2
    elif tensorclass == TensorZ3:
        n_qnums = 3
    else:
        n_qnums = None
    return n_qnums


@pytest.fixture
def rshape(n_qnums):
    def _rshape(n=None, chi=None, nlow=0, nhigh=5, chilow=0, chihigh=3):
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
    def _rqhape(shape, qlow=-3, qhigh=3):
        if n_qnums is not None:
            qlow = 0
            qhigh = n_qnums - 1
        try:
            assert all(len(dim) <= qhigh - qlow + 1 for dim in shape)
        except TypeError:
            pass
        possible_qnums = range(qlow, qhigh + 1)
        try:
            qhape = [
                list(np.random.choice(possible_qnums, len(dim), replace=False))
                for dim in shape
            ]
        except TypeError:
            qhape = None
        return qhape

    return _rqhape


@pytest.fixture
def rdirs():
    def _rdirs(shape=None, length=None):
        if shape is not None:
            length = len(shape)
        dirs = np.random.randint(low=0, high=2, size=length)
        dirs = list(2 * dirs - 1)
        return dirs

    return _rdirs


@pytest.fixture
def rcharge():
    def _rcharge():
        return np.random.randint(low=0, high=4)

    return _rcharge


@pytest.fixture
def rtensor(tensorclass, n_qnums, rqhape, rshape, rdirs, rcharge):
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
