import torch

from eml_lab.operators import StabilityConfig, StabilityStats, eml_exact, eml_train
from eml_lab.targets import get_target, ln_tree, sample_inputs


def test_eml_exact_recovers_exp_identity() -> None:
    x = torch.linspace(-1, 1, 32, dtype=torch.float64).to(torch.complex128)
    y = torch.ones_like(x)

    assert torch.allclose(eml_exact(x, y), torch.exp(x), atol=1e-12, rtol=1e-12)


def test_eml_exact_recovers_e_constant() -> None:
    one = torch.ones((), dtype=torch.complex128)

    assert torch.allclose(eml_exact(one, one), torch.exp(one), atol=1e-12, rtol=1e-12)


def test_paper_ln_identity_matches_torch_log() -> None:
    spec = get_target("ln")
    inputs = sample_inputs(spec, points=64)
    predicted = ln_tree().evaluate(inputs)

    assert torch.allclose(predicted, torch.log(inputs["x"]), atol=1e-12, rtol=1e-12)


def test_eml_train_returns_stability_stats() -> None:
    x = torch.tensor([100.0 + 0j], dtype=torch.complex128)
    y = torch.tensor([0.0 + 0j], dtype=torch.complex128)

    value, stats = eml_train(x, y, StabilityConfig(), return_stats=True)

    assert isinstance(stats, StabilityStats)
    assert torch.isfinite(value).all()
    assert stats.exp_clipped == 1
    assert stats.log_nudged == 1
