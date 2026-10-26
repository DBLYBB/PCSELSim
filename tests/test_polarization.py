import numpy as np
import pytest

from pcselsim.polarization import polarization_metrics, stokes_maps


def test_linear_jones_field_is_invariant_to_common_complex_scale():
    angle=np.linspace(-1,1,21)
    xx,yy=np.meshgrid(angle,angle)
    fx=np.exp(-4*(xx**2+yy**2)).astype(complex)
    fy=.5*fx
    baseline=polarization_metrics(angle,fx,fy,core_radius_deg=.5)
    changed=polarization_metrics(angle,fx*(2e80-3e80j),fy*(2e80-3e80j),core_radius_deg=.5)
    for part in ('integrated','on_axis','core'):
        assert np.allclose(baseline[part]['normalized_stokes_s1_s2_s3'],[.6,.8,0])
        assert np.allclose(changed[part]['normalized_stokes_s1_s2_s3'],[.6,.8,0])
        assert baseline[part]['linear_axis_angle_deg']==pytest.approx(np.rad2deg(np.arctan(.5)))
        assert baseline[part]['degree_of_polarization']==pytest.approx(1)


def test_circular_jones_field_has_explicit_signed_stokes_and_no_axis():
    angle=np.linspace(-1,1,5)
    field=np.ones((5,5),complex)
    for sign in (-1,1):
        metrics=polarization_metrics(angle,field,sign*1j*field)['integrated']
        assert metrics['signed_circular_fraction_s3_over_s0']==pytest.approx(-sign)
        assert metrics['degree_of_linear_polarization']==pytest.approx(0)
        assert metrics['linear_axis_angle_deg'] is None


def test_incoherent_angular_integration_preserves_opposite_polarizations():
    angle=np.array([-1.,-.2,0.,.2,1.])
    fx=np.zeros((5,5),complex);fy=fx.copy()
    fx[2,1]=1;fy[2,3]=1
    result=polarization_metrics(angle,fx,fy)
    # Symmetric, spatially separated x and y emitters cannot coherently cancel.
    assert result['integrated']['degree_of_polarization']==pytest.approx(0)
    assert result['on_axis'] is None
    assert result['integrated']['x_polarized_power_fraction']==pytest.approx(.5)


def test_nonuniform_axis_uses_trapezoid_quadrature_not_sample_count():
    angle=np.array([-1.,-.9,0.,.4,1.])
    fx=np.zeros((5,5),complex);fy=fx.copy()
    fx[:,0]=1;fy[:,2]=1
    out=polarization_metrics(angle,fx,fy)['integrated']
    # x=first endpoint width .05, y-polarized center width .65.
    assert out['x_polarized_power_fraction']==pytest.approx(.05/(.05+.65))


def test_maps_preserve_common_scale_and_mask_dark_or_circular_axes():
    fx=np.zeros((5,5),complex);fy=fx.copy()
    fx[1,1]=2;fy[1,1]=1
    fx[2,2]=1;fy[2,2]=1j
    maps=stokes_maps(fx,fy)
    assert maps['s1_over_s0'][1,1]==pytest.approx(.6)
    assert maps['s2_over_s0'][1,1]==pytest.approx(.8)
    assert not maps['valid_mask'][0,0]
    assert not maps['axis_valid_mask'][2,2]
    for value in maps.values():assert np.isfinite(value).all()


def test_invalid_fields_axes_and_core_rejected():
    angle=np.linspace(-1,1,5);fx=np.ones((5,5),complex)
    with pytest.raises(ValueError):polarization_metrics(angle,fx*0,fx*0)
    with pytest.raises(ValueError):polarization_metrics(angle[::-1],fx,fx)
    with pytest.raises(ValueError):polarization_metrics(angle,fx,fx,core_radius_deg=1.1)
    with pytest.raises(ValueError):polarization_metrics(angle,fx[1:],fx[1:])
