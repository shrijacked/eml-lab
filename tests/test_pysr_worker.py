import numpy as np
import pytest

from eml_lab.pysr_worker import _target_values


def test_target_values_support_univariate_targets() -> None:
    x = np.array([1.0, 2.0], dtype=np.float64)

    assert np.allclose(_target_values("identity", x), x)
    assert np.allclose(_target_values("exp", x), np.exp(x))
    assert np.allclose(_target_values("ln", x), np.log(x))
    assert np.allclose(_target_values("square", x), x * x)
    assert np.allclose(_target_values("sin", x), np.sin(x))


def test_target_values_reject_unknown_targets() -> None:
    with pytest.raises(ValueError, match="Unsupported PySR worker target"):
        _target_values("mul", np.array([1.0], dtype=np.float64))
