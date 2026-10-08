import numpy as np

from pcselsim.band_structure import square_four_wave_band_diagram


def test_square_band_diagram_contains_gamma_and_four_tracked_branches() -> None:
    coupling = np.diag(np.asarray((-300.0, -100.0, 120.0, 400.0), dtype=complex))
    diagram = square_four_wave_band_diagram(
        coupling,
        lattice_constant_nm=300.0,
        bragg_wavelength_nm=1000.0,
        effective_index=10.0 / 3.0,
        q_max=0.01,
        points=21,
    )
    assert diagram.normalized_frequency.shape == (4, 21)
    assert np.isclose(diagram.wave_number_2pi_over_a[10], 0.0)
    assert (diagram.left_endpoint, diagram.right_endpoint) == ("M", "X")
    expected = 0.3 + np.diag(coupling).real * 300e-9 / (2.0 * np.pi * (10.0 / 3.0))
    assert np.allclose(diagram.normalized_frequency[:, 10], expected)
    assert np.all(diagram.radiation_constant_cm >= 0.0)

