from dataclasses import replace
import numpy as np
from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG,solve_multimode_spatial_ybyag
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold


def system(losses=(100.,200.)):
    medium=YbYAGMediumConfig(dopant_density_cm3=1.38e21,spontaneous_emission_factor=0.,
                            pumped_area_um2=500**2,gain_thickness_um=1.)
    modes=tuple(SpatialYbYAGModel(medium,np.array([.5e-6,.5e-6]),np.array([.4,.4]),loss,
                upward_power_loss_per_m=loss*.3) for loss in losses)
    return SpatialMultimodeYbYAG(modes,tuple(f'm{i}' for i in range(len(losses))))


def test_shared_threshold_is_first_gain_crossing_and_next_is_unlased_crossing():
    modes=system();pump=PumpGeometryConfig('edge_dual',0.,500.,11)
    result=shared_inversion_pump_threshold(modes,pump)
    assert result.first_modes==('m0',)
    assert abs(result.threshold_residual)<1e-7
    assert result.branch_thresholds_internal_W['m0'] < result.branch_thresholds_internal_W['m1']
    assert result.next_small_signal_threshold_internal_W==result.branch_thresholds_internal_W['m1']
    # Equal spatial weights: the second zero-photon crossing does not mean the
    # second mode lases when the lowest-loss mode clamps their shared gain.
    above=solve_multimode_spatial_ybyag(modes,replace(pump,
            internal_incident_power_W=result.next_small_signal_threshold_internal_W*1.1))
    assert above.active_modes==('m0',)


def test_degenerate_thresholds_and_unreachable_branch_are_reported_honestly():
    modes=system((100.,100.,1e8));pump=PumpGeometryConfig('top_single',0.,500.,3,
                                           top_passes=20,return_efficiency=.98)
    result=shared_inversion_pump_threshold(modes,pump,max_power_W=1000.)
    assert result.first_modes==('m0','m1')
    assert abs(result.next_to_first_ratio-1)<1e-8
    assert result.branch_thresholds_internal_W['m2'] is None
    assert result.searched_power_limit_W==1000.
