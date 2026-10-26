"""Frozen-frequency full finite-spectrum screening, replacing four-label ranks."""
import sys,json,argparse
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from optimize_ybyag_cavity import *
from pcselsim.custom_analysis import finite_area_eigenproblem,LinearMode
from pcselsim.frequency_cwt import background_detuning


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('designs_json');p.add_argument('--output')
    p.add_argument('--grid',type=int,default=13);p.add_argument('--truncation',type=int,default=3)
    a=p.parse_args(argv);out=prepare_output(a.output or default_output(ROOT,'optimization_20261021/yag/full_screen'),ROOT)
    write_provenance(out,ROOT);results=[];save(out/'run_status.json',dict(status='running'))
    for value in json.loads(Path(a.designs_json).read_text()):
        design=Design(**value);cell,guide,stack,_=structure(design)
        count=len(solve_guided_te_modes(guide,1030,5,32))
        period=(design.fixed_period_nm if design.fixed_period_nm is not None else
                1030/guide.solve_te0(1030,5).effective_index*design.period_scale)
        if count!=1:
            results.append(dict(design=asdict(design),status='rejected_more_than_one_TE_for_this_screen',TEcount=count));continue
        optical=build_reflector_coupling(cell,guide,stack,'passive YAG photonic crystal',period,1030,
            ThreeDCWTSettings(a.truncation,5),fixed_wavelength_nm=1030.)
        r=optical.result;spec=FourWaveOpticalSpec(1030.,r.effective_index,r.group_index,design.device_um,.1,
            tuple(r.eigenvalues_m.real/100),tuple(r.eigenvalues_m.imag/100),a.grid)
        problem=finite_area_eigenproblem(spec,r.coupling_m)
        problem.near(complex(r.eigenvalues_m[0]),4,np.ones(4*a.grid*a.grid,dtype=complex),dense_limit=2500)
        vals,fields=problem._dense_spectrum
        _,_,qg=active_weights(r,guide,24);rows=[]
        derivative=-2*np.pi*r.group_index/(1030e-9)**2*1e-9
        bg=background_detuning(1030.,r.effective_index,period)
        m=medium(design)
        for i,val in enumerate(vals):
            wave=1030+(val.real-bg)/derivative
            if val.imag<0 or not 1010<=wave<=1060:continue
            loss=2*val.imag+100*design.internal_power_loss_cm_inverse
            fraction=(loss/qg.sum()/(m.dopant_density_cm3*1e6)+m.laser_absorption_cross_section_cm2*1e-4)/((m.laser_emission_cross_section_cm2+m.laser_absorption_cross_section_cm2)*1e-4)
            denom=(m.pump_absorption_cross_section_cm2*(1-fraction)-m.pump_emission_cross_section_cm2*fraction)*1e-4
            if denom<=0:continue
            proxy=fraction*h*c/(m.pump_wavelength_nm*1e-9)/(m.upper_state_lifetime_ms*1e-3*denom)*design.device_um*1e-6*design.gain_nm*1e-9
            rows.append(dict(spectrum_index=i,uniform_intensity_edge_proxy_W=float(proxy),
                amplitude_loss_per_m=float(val.imag),total_power_loss_per_m=float(loss),estimated_wavelength_nm=float(wave)))
        rows.sort(key=lambda x:x['uniform_intensity_edge_proxy_W'])
        folder=out/design.id;folder.mkdir()
        for row in rows[:8]:
            i=row['spectrum_index'];field=fields[:,i].reshape(4,a.grid,a.grid);rx,ry=r.radiation_fields(field)
            mode=LinearMode(f'L{i}',vals[i].real,vals[i].imag,field,0,rx,ry)
            up,down=optical.side_power_losses(field);row['up_fraction_of_total']=up/row['total_power_loss_per_m']
            row['ffp']=beam(spec,mode)
            np.savez_compressed(folder/f'L{i}_seed.npz',field=field,value=vals[i])
        result=dict(design=asdict(design),scope=f'FROZEN_D{a.truncation}_N{a.grid}_FULL_RESIDUAL_FILTERED_SPECTRUM_PROXY',
            period_nm=period,TEcount=count,verified_eigenpairs=len(vals),gain_weight=float(qg.sum()),ranked_modes=rows,
            next_to_first_proxy_ratio=rows[1]['uniform_intensity_edge_proxy_W']/rows[0]['uniform_intensity_edge_proxy_W'] if len(rows)>1 else None)
        results.append(result);save(out/'results.json',results)
        if rows:
            first=rows[0];f=first['ffp'];print(design.id,'Pproxy',first['uniform_intensity_edge_proxy_W'],
                'gap',result['next_to_first_proxy_ratio'],'eta',first['up_fraction_of_total'],
                'ellipse/center/rad',f['ellipticity'],f['center_to_peak'],f['radial_anisotropy_l2'],flush=True)
        else:print(design.id,'unreachable for all scanned eigenmodes',flush=True)
    save(out/'run_status.json',dict(status='complete'))


if __name__=='__main__':main()
