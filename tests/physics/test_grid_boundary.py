import pytest
import torch

from dhundh.physics.boundary import pad_last
from dhundh.physics.grid import Grid


def test_centers_are_cell_midpoints():
    g = Grid(nx=4, ny=3, dx=1000.0, dy=500.0)
    X, Y = g.centers()
    assert X.shape == (3, 4) and Y.shape == (3, 4)
    assert X[0].tolist() == [500.0, 1500.0, 2500.0, 3500.0]
    assert Y[:, 0].tolist() == [250.0, 750.0, 1250.0]
    assert g.cell_area == 500_000.0


def test_pad_periodic_wraps():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert pad_last(x, 2, "periodic").tolist() == [[2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]]


def test_pad_zero_gradient_replicates_edges():
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    assert pad_last(x, 1, "zero_gradient").tolist() == [[[1.0, 1.0, 2.0, 3.0, 3.0]]]


def test_pad_rejects_unknown_bc():
    with pytest.raises(ValueError):
        pad_last(torch.zeros(2, 2), 1, "reflect")
