import torch

from dhundh.physics.advection import advective_divergence, van_leer_slopes


def test_slopes_follow_a_linear_ramp():
    P = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    assert torch.allclose(van_leer_slopes(P), torch.tensor([1.0, 1.0, 1.0]))


def test_slopes_vanish_at_extrema():
    P = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    assert torch.equal(van_leer_slopes(P), torch.zeros(3))


def test_uniform_field_has_zero_divergence_under_uniform_wind():
    C = torch.full((2, 8, 10), 42.0)
    u = torch.full((8, 10), 3.0)
    v = torch.full((8, 10), -1.5)
    div = advective_divergence(C, u, v, 1000.0, 1000.0, "periodic")
    assert torch.allclose(div, torch.zeros_like(div), atol=1e-12)


def test_periodic_flux_form_conserves_mass_exactly():
    torch.manual_seed(1)
    C = torch.rand(3, 16, 20) * 100
    u = torch.randn(16, 20) * 5
    v = torch.randn(16, 20) * 5
    div = advective_divergence(C, u, v, 750.0, 900.0, "periodic")
    total = div.sum(dim=(-1, -2))
    assert torch.allclose(total, torch.zeros(3), atol=1e-10)


def test_positive_wind_moves_mass_downstream():
    # 4 rows: periodic padding needs at least 2 cells per axis.
    C = torch.zeros(1, 4, 10)
    C[0, :, 4] = 1.0
    u = torch.full((4, 10), 1.0)
    v = torch.zeros(4, 10)
    tend = -advective_divergence(C, u, v, 1.0, 1.0, "periodic")
    assert tend[0, 0, 4] < 0
    assert tend[0, 0, 5] > 0
    assert tend[0, 0, 3] == 0
