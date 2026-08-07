import numpy as np

from tri_modal_modular_grokking.fourier import grid_fft_energy


def test_grid_fft_energy_identifies_addition_diagonal():
    modulus = 7
    grid = np.zeros((modulus, modulus, modulus))
    for a in range(modulus):
        for b in range(modulus):
            grid[a, b, (a + b) % modulus] = 1.0
    result = grid_fft_energy(grid)
    assert result["addition_diag"] > 0.9
    assert result["addition_diag"] > result["difference_diag"]
    assert result["addition_diag"] > result["a_only"]


def test_grid_fft_energy_identifies_a_only():
    modulus = 7
    grid = np.zeros((modulus, modulus, modulus))
    for a in range(modulus):
        for b in range(modulus):
            grid[a, b, a] = 1.0
    result = grid_fft_energy(grid)
    assert result["a_only"] > 0.9
    assert result["a_only"] > result["addition_diag"]
