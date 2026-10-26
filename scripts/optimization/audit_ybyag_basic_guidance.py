"""Independent analytic multilayer TE-pole and FD-padding guidance challenge.

Unlike the isolated guide used by CWT, the full-stack pole includes the actual
finite DBR and propagating YAG substrate for the *basic* TE wave. Its complex
in-plane beta quantifies tunnelling loss absent from the old basic-wave guide.
"""
import sys,json,argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import root,brentq
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from optimize_ybyag_cavity import Design,structure,save
from pcselsim.stratified_optics import StratifiedStack,StratifiedLayer,_network
from pcselsim.run_output import prepare_output,write_provenance,default_output


def pole(stack,wavelength_nm,seed_neff,host_name='Yb:YAG gain'):
    k0=2*np.pi/(wavelength_nm*1e-9)
    indices=np.array([stack.bottom_index,*[x.refractive_index for x in stack.layers],stack.top_index],complex)
    raw=np.sqrt((k0*indices)**2-(k0*seed_neff)**2+0j)
    reference=np.where(raw.imag>0,-raw,raw)
    host=next(i+1 for i,x in enumerate(stack.layers) if x.name==host_name)
    d=stack.layers[host-1].thickness_nm*1e-9
    def parameters(neff):
        q=k0*neff
        kz=np.sqrt((k0*indices)**2-q*q+0j)
        kz=np.where(abs(kz-reference)>abs(-kz-reference),-kz,kz)
        lower=_network(stack,kz,kz,0,host)
        upper=_network(stack,kz,kz,host,len(stack.layers)+1)
        return kz,lower[1,1],upper[0,0]
    def residual(values):
        kz,lower,upper=parameters(complex(*values))
        determinant=1-lower*upper*np.exp(-2j*kz[host]*d)
        return np.array([determinant.real,determinant.imag])
    fit=root(residual,[seed_neff,-1e-11],tol=1e-11)
    error=float(np.linalg.norm(residual(fit.x)))
    if error>1e-8: raise RuntimeError(f'Analytic outgoing basic-TE pole failed: {fit.message}, {error}')
    neff=complex(*fit.x);kz,lower,upper=parameters(neff)
    # Compare only the physical gain/PC/cap/cladding part. The outgoing leaky
    # substrate field is not square-integrable at complex beta.
    amplitude_down=upper*np.exp(-2j*kz[host]*d)
    state=np.array([1+amplitude_down,-1j*kz[host]+1j*kz[host]*amplitude_down])
    z=[];fields=[];cursor=0.
    for j in range(host,len(stack.layers)+1):
        layer=stack.layers[j-1];thickness=layer.thickness_nm*1e-9
        samples=np.linspace(0,thickness,max(3,int(np.ceil(layer.thickness_nm/2))+1))
        e=state[0]*np.cos(kz[j]*samples)+state[1]*np.sin(kz[j]*samples)/kz[j]
        z.extend(cursor+samples[:-1]);fields.extend(e[:-1]);cursor+=thickness
        cosine=np.cos(kz[j]*thickness);sine=np.sin(kz[j]*thickness)
        state=np.array([cosine*state[0]+sine/kz[j]*state[1],-kz[j]*sine*state[0]+cosine*state[1]])
    return dict(effective_index_real=neff.real,effective_index_imag=neff.imag,
        basic_amplitude_attenuation_per_m=float(-k0*neff.imag),basic_power_attenuation_per_m=float(-2*k0*neff.imag),
        pole_determinant_residual=error,root_solver_success=bool(fit.success)),np.asarray(z),np.asarray(fields)


def slab_self_test():
    n=1.8166;clad=1.45;t=300e-9;w=1030.;k0=2*np.pi/(w*1e-9)
    even=lambda beta: np.sqrt((k0*n)**2-beta*beta)*np.tan(np.sqrt((k0*n)**2-beta*beta)*t/2)-np.sqrt(beta*beta-(k0*clad)**2)
    beta=brentq(even,k0*clad*(1+1e-10),k0*n*(1-1e-10))
    s=StratifiedStack((StratifiedLayer('Yb:YAG gain',t*1e9,n),),clad,clad)
    solved,_,_=pole(s,w,beta/k0)
    return dict(exact_even_slab_neff=beta/k0,multilayer_pole_neff=solved['effective_index_real'],
        absolute_index_error=abs(beta/k0-solved['effective_index_real']),residual=solved['pole_determinant_residual'])


def audit(design,wavelength_nm=1030.):
    _,guide,full,_=structure(design)
    fd=[]
    for padding in (2.,4.,8.):
        for dz in (10.,5.,2.5):
            mode=replace(guide,padding_um=padding).solve_te0(wavelength_nm,dz)
            z=mode.z_um*1e-6;density=abs(mode.field)**2
            edge=(z-z[0]<.5e-6)|(z[-1]-z<.5e-6)
            trap=getattr(np,'trapezoid',np.trapz)
            fd.append(dict(padding_um=padding,dz_nm=dz,neff=mode.effective_index,
                outer_half_um_energy_fraction=float(trap(np.where(edge,density,0),z)/trap(density,z)),
                lower_boundary_field_relative_to_peak=float(abs(mode.field[1])/abs(mode.field).max()),
                upper_boundary_field_relative_to_peak=float(abs(mode.field[-2])/abs(mode.field).max())))
    isolated=StratifiedStack(tuple(StratifiedLayer(x.name,x.thickness_nm,x.refractive_index) for x in guide.layers),
                            bottom_index=guide.bottom_index,top_index=guide.top_index)
    seed=fd[-1]['neff'];iso,zi,fi=pole(isolated,wavelength_nm,seed)
    actual,za,fa=pole(full,wavelength_nm,seed)
    trap=getattr(np,'trapezoid',np.trapz)
    overlap=float(abs(trap(fi.conj()*fa,zi))**2/(trap(abs(fi)**2,zi)*trap(abs(fa)**2,za)))
    return dict(design=design.__dict__,wavelength_nm=wavelength_nm,
        isolated_claddings=dict(bottom=guide.bottom_index,top=guide.top_index),
        physical_exteriors=dict(bottom_substrate=complex(full.bottom_index).real,top=complex(full.top_index).real),
        finite_difference_padding_sweep=fd,analytic_isolated_guide=iso,analytic_full_DBR_stack=actual,
        full_minus_isolated_neff=actual['effective_index_real']-iso['effective_index_real'],
        normalized_core_field_overlap=overlap,
        interpretation='full-stack basic-wave leakage was absent from original isolated-guide CWT; assess its size rather than assuming zero')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('designs_json');p.add_argument('--output')
    a=p.parse_args(argv);out=prepare_output(a.output or default_output(ROOT,'optimization_20261021/yag/guidance'),ROOT)
    write_provenance(out,ROOT);save(out/'self_test.json',slab_self_test());rows=[]
    for d in json.loads(Path(a.designs_json).read_text()):
        result=audit(Design(**d));rows.append(result);save(out/'results.json',rows)
        print(d['id'],result['analytic_full_DBR_stack'], 'delta_n',result['full_minus_isolated_neff'],
              'overlap',result['normalized_core_field_overlap'],flush=True)
    save(out/'run_status.json',dict(status='complete'))


if __name__=='__main__':main()
