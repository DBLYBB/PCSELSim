"""Negative physical controls: invalid optics must fail, not be projected away."""
import numpy as np
import pytest
from pcselsim.geometry import Ellipse, PolygonInclusion, SquareLatticeCell
from pcselsim.triangular_six_wave import TriangularEllipse, TriangularPolygon, TriangularLatticeCell
from pcselsim.three_d_cwt import _project_radiation_passive, _roundoff_hermitian
from pcselsim.triangular_six_wave import triangular_band_diagram
from types import SimpleNamespace


@pytest.mark.parametrize('factory',[
    lambda: Ellipse((0,0),(.1,.1),epsilon=1-.01j),
    lambda: PolygonInclusion(((0,0),(.1,0),(0,.1)),epsilon=1-.01j),
    lambda: SquareLatticeCell(3.4**2-.01j,()),
    lambda: TriangularEllipse(epsilon=1-.01j),
    lambda: TriangularPolygon(((0,0),(.1,0),(0,.1)),epsilon=1-.01j),
    lambda: TriangularLatticeCell(3.4**2-.01j,()),
])
def test_absorption_cannot_be_silently_removed_by_lossless_cwt(factory):
    with pytest.raises(ValueError,match='extended optical model'):
        factory()


def test_substantial_negative_radiation_is_rejected():
    with pytest.raises(ValueError,match='PSD projection cannot repair'):
        _project_radiation_passive(np.diag([1j,-.2j]))


def test_roundoff_radiation_retains_the_audited_cleanup():
    matrix=np.diag([1j,-1e-15j])
    projected,correction=_project_radiation_passive(matrix)
    np.testing.assert_allclose(projected,np.diag([1j,0j]),rtol=0,atol=0)
    assert correction==pytest.approx(1e-15)


def test_large_closed_channel_skew_is_not_hidden_by_hermitianization():
    matrix=np.array([[1,2+.1j],[2,3]],complex)
    with pytest.raises(ValueError,match='refusing to hide'):
        _roundoff_hermitian(matrix,label='physical closed channel')


def test_roundoff_closed_channel_symmetrization_unchanged():
    matrix=np.array([[1,2+1e-15j],[2,3]],complex)
    np.testing.assert_array_equal(_roundoff_hermitian(matrix,label='physical closed channel'),.5*(matrix+matrix.conj().T))


def test_triangular_physical_frequency_origin_uses_its_own_reciprocal_lattice():
    period,wavelength,neff,ng=695.,1030.,1.708,1.85
    beta0=4*np.pi/(np.sqrt(3)*period*1e-9); beta_ref=2*np.pi*neff/(wavelength*1e-9)
    delta_ref=(beta_ref**2-beta0**2)/(2*beta0)
    model=SimpleNamespace(lattice_constant_nm=period,bragg_wavelength_nm=wavelength,
        effective_index=neff,group_index=ng,coupling_m=np.diag(delta_ref+np.arange(6)*100+1j))
    diagram=triangular_band_diagram(model,points=21)
    assert diagram.normalized_frequency[0,10]==pytest.approx(period/wavelength,rel=1e-14)
