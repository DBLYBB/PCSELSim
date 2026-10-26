"""Additional pump/loss studies of saved, frequency-closed Yb:YAG fields.

Changing an explicitly uniform phenomenological internal power loss leaves the
lossless cold-cavity spatial eigenfield unchanged in this model. These studies
do not substitute internal loss for measured absorbing-film Green functions.
"""
import argparse,json,sys
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from scipy.constants import c
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from optimize_ybyag_cavity import Design,medium,save
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG,solve_multimode_spatial_ybyag
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig,pump_geometry_metadata
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold
from pcselsim.run_output import prepare_output,write_provenance,default_output


def load_closed(folder,internal_power_loss_per_m=None):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text())
    design=Design(**summary['design'])
    if internal_power_loss_per_m is None:
        internal_power_loss_per_m=100*design.internal_power_loss_cm_inverse
    rows=json.loads((folder/'optical_modes.json').read_text());models=[]
    for r in rows:
        data=np.load(folder/(r['name']+'_fields.npz'))
        # ng is reconstructed from the branch's pump-independent field solver
        # trace at the converged wavelength, rather than assumed bulk n.
        evaluation=min(r['frequency_evaluations'],key=lambda x:abs(x['wavelength_nm']-r['wavelength_nm']))
        models.append(SpatialYbYAGModel(medium(design,r['wavelength_nm']),data['widths'],data['qenergy'],
            2*r['amplitude_loss_per_m']+internal_power_loss_per_m,r['up_loss_per_m'],r['down_loss_per_m'],
            c/evaluation['group_index'],data['qgain'],data['lateral']))
    return design,SpatialMultimodeYbYAG(tuple(models),tuple(r['name'] for r in rows))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--output')
    p.add_argument('--internal-loss-factors',type=float,nargs='+',default=[1,2,5])
    p.add_argument('--pump-powers-W',type=float,nargs='+',default=[.005,.01,.03,.1,1,4,32])
    a=p.parse_args(argv);out=prepare_output(a.output or default_output(ROOT,'optimization_20261021/yag/loss'),ROOT)
    write_provenance(out,ROOT);save(out/'run_plan.json',vars(a));results=[]
    for factor in a.internal_loss_factors:
        design,system=load_closed(a.source,10*factor)
        for architecture,passes,eta in [('edge_single',1,1.),('edge_dual',1,1.),('top_single',1,1.),('top_single',20,.98)]:
            pump=PumpGeometryConfig(architecture,0.,design.device_um,len(system.modes[0].lateral_energy_fractions),
                                   top_passes=passes,return_efficiency=eta)
            threshold=shared_inversion_pump_threshold(system,pump)
            scans=[]
            for power in a.pump_powers_W:
                result=solve_multimode_spatial_ybyag(system,replace(pump,internal_incident_power_W=power))
                scans.append(dict(input_W=power,pump_geometry=pump_geometry_metadata(system.modes[0],replace(pump,internal_incident_power_W=power)),
                    active_modes=result.active_modes,outputs_W=dict(zip(system.names,map(float,result.upward_output_W))),
                    budget=result.energy_budget_W,absorbed_fraction=result.energy_budget_W['absorbed_pump_W']/power if power else 0,
                    ncp_residual=result.complementarity_residual,population_residual_per_s=result.population_residual_per_s,
                    inactive_gain_excess_per_m=result.maximum_inactive_gain_excess_per_m))
            results.append(dict(internal_power_loss_per_m=10*factor,architecture=architecture,top_passes=passes,
                return_efficiency=eta,threshold=asdict(threshold),threshold_pump_geometry=pump_geometry_metadata(system.modes[0],
                    replace(pump,internal_incident_power_W=threshold.first_threshold_internal_W or 0)),cw_scan=scans))
            print(factor,architecture,passes,'threshold',threshold.first_threshold_internal_W,flush=True)
            save(out/'results.json',results)
    save(out/'run_status.json',dict(status='complete'))


if __name__=='__main__':main()
