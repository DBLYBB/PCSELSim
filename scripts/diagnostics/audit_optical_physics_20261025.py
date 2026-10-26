"""Independent optical checks and pre-projection diagnostics / 光学物理审计。

Source equations: Liang thesis (3.14), (3.25)-(3.35), (B1)-(B14);
Inoue 2019 Appendix A. The compact TE projectors below are an independent
assembly of those equations, not a second Maxwell solver. All original
results are read-only. The report distinguishes algebraic checks from
experimental/full-vector validation.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from pcselsim.geometry import Ellipse, PolygonInclusion, SquareLatticeCell, polygon_fourier_integral
from pcselsim.vertical import Layer, LayerStack
import pcselsim.three_d_cwt as sq
import pcselsim.triangular_six_wave as tr
import pcselsim.reflector_cwt as ref
import pcselsim.matched300_yag as yr
from pcselsim.band_structure import square_four_wave_band_diagram
from pcselsim.stratified_optics import StratifiedLayer, StratifiedStack, solve_stack, local_green_kernel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instrument(function, tap, overrides=None):
    """Capture matrices before existing projections, without changing arithmetic.

    仅在独立函数副本中插入读回，不修改源文件或生产函数。
    """
    source = inspect.getsource(function)
    name = function.__module__.split(".")[-1]
    needles = {
        "three_d_cwt": ("c1d = 0.5*(c1d+c1d.conj().T)", "c2d = 0.5*(c2d+c2d.conj().T)"),
        "triangular_six_wave": ("cb = 0.5 * (cb + cb.conj().T)", "ch = 0.5 * (ch + ch.conj().T)"),
        "reflector_cwt": ("c2d = (c2d + c2d.conj().T) / 2",),
        "matched300_yag": ("ch = (ch+ch.conj().T)/2",),
    }
    # Accept the audited guard calls after the remediation without changing
    # the captured pre-projection matrices / 修复后仍读回投影前矩阵。
    guarded = {
        "three_d_cwt": ("c1d = _roundoff_hermitian(c1d, label=\"C1D\")", "c2d = _roundoff_hermitian(c2d, label=\"C2D\")"),
        "triangular_six_wave": ("cb = _roundoff_hermitian(cb, label=\"Cb\")", "ch = _roundoff_hermitian(ch, label=\"Ch\")"),
        "reflector_cwt": ("c2d = _roundoff_hermitian(c2d, label=\"reflected C2D\")",),
        "matched300_yag": ("ch = _roundoff_hermitian(ch, label=\"reflected Ch\")",),
    }
    for old, alternative in zip(needles[name], guarded[name]):
        selected = old if old in source else alternative
        if selected not in source:
            raise RuntimeError(f"Audit insertion no longer matches {name}: {old}")
        variable = selected.split(" =")[0]
        source = source.replace(selected, f"_tap['{name}.{variable}'] = {variable}.copy()\n    {selected}")
    namespace = dict(function.__globals__)
    namespace.update(overrides or {})
    namespace["_tap"] = tap
    exec(compile(source, f"<audit copy of {name}>", "exec"), namespace)
    return namespace[function.__name__]


def independent_high(result, orders, polarizations, shell, kernel_for_metric):
    """Assemble the high-order block with 2x2 longitudinal/TE projectors.

    Unlike the production component formulas, the construction uses outer
    products of physical Cartesian G, implementing Liang(3.26)/(B2).
    与分量实现不同，使用笛卡尔投影矩阵独立组装高阶项。
    """
    wavelength = result.bragg_wavelength_nm * 1e-9
    k0 = 2*np.pi/wavelength
    beta0 = (2*np.pi if len(orders) == 4 else 4*np.pi/np.sqrt(3))/(result.lattice_constant_nm*1e-9)
    q = result.pc_confinement/result.average_pc_epsilon
    output = np.zeros((len(orders), len(orders)), complex)
    for m,n in shell:
        g = np.array((m,n),float) if len(orders)==4 else np.array(tr.reciprocal_components((m,n)))
        metric = float(g @ g)
        parallel = np.outer(g,g)/metric
        transverse = np.eye(2)-parallel
        response = k0*k0*kernel_for_metric(m,n,metric)*transverse-q*parallel
        xi_out = np.array([result.fourier(p-m,q0-n) for p,q0 in orders])
        xi_in = np.array([result.fourier(m-r,n-s) for r,s in orders])
        output += -(k0*k0)/(2*beta0)*(xi_out[:,None]*xi_in[None,:])*(polarizations @ response @ polarizations.T)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/'results/current/physics_audit_20261025/optics/independent_checks.json')
    parser.add_argument("--order",type=int,default=4)
    parser.add_argument("--dz",type=float,default=5.)
    args=parser.parse_args()
    started=time.monotonic(); checks=[]; details={}
    def check(name,value,limit):
        checks.append(dict(name=name,value=float(value),limit=float(limit),passed=bool(value<=limit)))
    # Independent area quadrature: translated/rotated ellipse against numerical
    # polar integration; triangle boundary integral against barycentric integral.
    ellipse=Ellipse((.113,-.071),(.12,.061),27.,1.)
    cell=SquareLatticeCell(3.4**2,(ellipse,))
    a=np.deg2rad(ellipse.angle_deg); rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    for m,n in [(0,0),(1,-2),(3,1)]:
        wave=2*np.pi*np.array((m,n)); phase=np.exp(1j*wave@ellipse.center)
        def angle_integral(r):
            real=quad(lambda t: np.cos(wave@(rotation@np.array([ellipse.radii[0]*r*np.cos(t),ellipse.radii[1]*r*np.sin(t)]))),0,2*np.pi,epsabs=2e-12)[0]
            imag=quad(lambda t: np.sin(wave@(rotation@np.array([ellipse.radii[0]*r*np.cos(t),ellipse.radii[1]*r*np.sin(t)]))),0,2*np.pi,epsabs=2e-12)[0]
            return r*(real+1j*imag)
        numerical=ellipse.radii[0]*ellipse.radii[1]*phase*(quad(lambda r:angle_integral(r).real,0,1,epsabs=2e-12)[0]+1j*quad(lambda r:angle_integral(r).imag,0,1,epsabs=2e-12)[0])
        observed=(cell.fourier_epsilon(m,n)-(cell.background_epsilon if (m,n)==(0,0) else 0))/(1-cell.background_epsilon)
        check(f'ellipse Fourier independent quadrature ({m},{n})',abs(observed-numerical),2e-11)
    vertices=np.array([[-.2,-.1],[.23,-.07],[.04,.19]])
    determinant=abs(np.linalg.det(np.column_stack((vertices[1]-vertices[0],vertices[2]-vertices[0]))))
    for m,n in [(0,0),(2,-1),(4,3)]:
        wave=2*np.pi*np.array((m,n))
        def integrate_part(imag):
            def integrand(u,v):
                point=vertices[0]+u*(vertices[1]-vertices[0])+v*(vertices[2]-vertices[0])
                value=np.exp(1j*(wave@point))
                return value.imag if imag else value.real
            return determinant*quad(lambda u:quad(lambda v:integrand(u,v),0,1-u,epsabs=2e-12)[0],0,1,epsabs=2e-12)[0]
        numerical=integrate_part(False)+1j*integrate_part(True)
        check(f'polygon Fourier barycentric quadrature ({m},{n})',abs(polygon_fourier_integral(tuple(map(tuple,vertices)),m,n)-numerical),2e-11)
    # Analytic slab dispersion and Hellmann-Feynman group index.
    slab=LayerStack((Layer('core',500.,3.5),),3.,3.,1.5)
    k0=2*np.pi/1e-6; v=k0*250e-9*np.sqrt(3.5**2-3.**2)
    u=brentq(lambda u:u*np.tan(u)-np.sqrt(v*v-u*u),1e-10,min(v,np.pi/2)-1e-10)
    exact=np.sqrt(3.5**2-(u/(k0*250e-9))**2)
    errors=[]
    for dz in [10.,5.,2.5,1.25]:
        mode=slab.solve_te0(1000.,dz); errors.append(abs(mode.effective_index-exact))
    details['slab_neff_errors']=errors
    check('TE slab finest error',errors[-1],3e-6)
    for i in range(1,len(errors)): check(f'TE slab dz refinement error ratio {i}',errors[i]/errors[i-1],.35)
    mode=slab.solve_te0(1000.,2.5)
    ng_energy=np.trapz(mode.index**2*abs(mode.field)**2,mode.z_um*1e-6)/mode.effective_index
    dn=(slab.solve_te0(1000.1,2.5).effective_index-slab.solve_te0(999.9,2.5).effective_index)/.2
    check('group index from energy versus d beta/d omega',abs(ng_energy-(mode.effective_index-1000*dn)),2e-6)
    # Analytic single-film Airy reflection, both polarizations at oblique q.
    for pol in ['TE','TM']:
        stack=StratifiedStack((StratifiedLayer('film',173.,1.92),),1.46,1.)
        q=2*np.pi/1030e-9*1.46*np.sin(.24)
        kz=np.sqrt((2*np.pi/1030e-9*np.array([1.46,1.92,1.]))**2-q*q+0j)
        adm=kz if pol=='TE' else kz/np.array([1.46,1.92,1.])**2
        r01=(adm[0]-adm[1])/(adm[0]+adm[1]); r12=(adm[1]-adm[2])/(adm[1]+adm[2])
        phase=np.exp(-2j*kz[1]*173e-9); expected=(r01+r12*phase)/(1+r01*r12*phase)
        observed=solve_stack(stack,1030.,polarization=pol,q_parallel_per_m=q)
        check(f'{pol} single-film Airy reflection',abs(observed.r-expected),2e-13)
        check(f'{pol} lossless scattering power',abs(observed.R+observed.T-1),2e-13)
    # Source Green theorem on a stack independent of the device templates.
    stack=StratifiedStack((StratifiedLayer('lower',327.,1.45),StratifiedLayer('host',120.,1.78),StratifiedLayer('upper',91.,2.1)),1.81,1.)
    kernel=local_green_kernel(stack,1,1030.)
    z=np.linspace(327e-9,447e-9,41); weights=sq._trapezoid_weights(z)
    rng=np.random.default_rng(20261025); source=rng.normal(size=z.size)+1j*rng.normal(size=z.size)
    flux=kernel.quadrature_flux(z,source,weights); details['independent_stack_green_flux']=flux
    check('stratified distributed-source optical theorem',abs(flux['relative_residual']),5e-12)
    check('stratified Green reciprocity',np.linalg.norm(kernel.matrix(z)-kernel.matrix(z).T)/np.linalg.norm(kernel.matrix(z)),5e-13)
    # Matched designs: raw matrices, independent Cartesian-projector assembly.
    manifest=json.loads((ROOT/'results/current/matched300_20261021/comparison/selected_cases.json').read_text(encoding='utf-8'))
    tap={}; sq_build=instrument(sq.build_geometry_coupling,tap); tr_build=instrument(tr.build_triangular_coupling,tap)
    ref_build=instrument(ref.build_reflector_coupling,tap,{'build_geometry_coupling':sq_build})
    yr_build=instrument(yr.build_triangular_reflector_coupling,tap,{'build_triangular_coupling':tr_build})
    matrices={}
    for case in manifest['cases']:
        summary=json.loads((ROOT/case['summary_path']).read_text(encoding='utf-8')); tap.clear(); radiation=None
        if case['id']=='semiconductor_square':
            from scripts.optimization.matched300_square_search import Design,build_cell,build_stack,BASE
            d=Design(**summary['design']); cell=build_cell(d,96); guide=build_stack(d,cell); pc=BASE.PC_LAYER_NAME
            model=sq_build(cell,guide,pc,d.lattice_nm,950.,sq.ThreeDCWTSettings(args.order,args.dz),fixed_wavelength_nm=case['wavelength_nm']); period=d.lattice_nm
        elif case['id']=='semiconductor_triangular':
            from scripts.optimization.matched300_triangular_search import ThreeHole,candidate,base_spec
            from scripts.optimization.triangular_semiconductor_search import parameters
            d=ThreeHole(**summary['three_hole_spec']); p=parameters(base_spec(d)); cell=tr.TriangularLatticeCell(p.BACKGROUND_INDEX**2,candidate(d).inclusions)
            nav=np.sqrt(cell.fourier_epsilon(0,0).real)
            guide=LayerStack(tuple(Layer(l.name,l.thickness_nm,nav if l.name==p.PC_LAYER_NAME else l.refractive_index) for l in p.LAYERS),p.TOP_CLADDING_INDEX,p.BOTTOM_CLADDING_INDEX,p.VERTICAL_PADDING_UM); pc=p.PC_LAYER_NAME; period=d.lattice_nm
            model=tr_build(cell,guide,pc,period,950.,tr.TriangularCWTSettings(args.order,args.dz),fixed_wavelength_nm=case['wavelength_nm'])
        else:
            from scripts.optimization.matched300_yag_search import structure,PC
            cell,guide,radiation=structure(case['lattice'],summary['design'],case.get('TE_family',0)); pc=PC; period=summary['period_nm']
            if case['lattice']=='square':
                model=ref_build(cell,guide,radiation,pc,period,1030.,sq.ThreeDCWTSettings(args.order,args.dz),fixed_wavelength_nm=case['wavelength_nm']).result
            else:
                model=yr_build(cell,guide,radiation,pc,period,1030.,tr.TriangularCWTSettings(args.order,args.dz),fixed_wavelength_nm=case['wavelength_nm']).result
        captured={}
        for key,raw in tap.items():
            relative=np.linalg.norm(raw-raw.conj().T)/max(np.linalg.norm(raw),1e-30)
            captured[key]={'raw_antihermitian_relative_norm':float(relative),'raw_norm_per_m':float(np.linalg.norm(raw))}
            check(case['id']+' '+key+' Hermitian before projection',relative,2e-11)
        orders=sq.BASIC_ORDERS if case['lattice']=='square' else tr.TRIANGULAR_BASIC_ORDERS
        pol=np.array([[0,1],[0,1],[1,0],[1,0]]) if case['lattice']=='square' else tr.TRIANGULAR_POLARIZATIONS
        limit=args.order if len(orders)==4 else int(np.ceil(2*args.order/np.sqrt(3)))
        shell=[(m,n) for m in range(-limit,limit+1) for n in range(-limit,limit+1) if (m*m+n*n>1 if len(orders)==4 else 1<tr.reciprocal_metric_squared(m,n)<=args.order**2)]
        zz,theta,ww=sq._pc_layer_coordinates(guide,model.vertical_mode,pc); weighted=theta*ww
        if radiation is not None:
            start=guide.padding_um*1e-6
            for layer in guide.layers:
                if layer.name==pc:break
                start+=layer.thickness_nm*1e-9
            host=next(i for i,l in enumerate(radiation.layers) if l.name==pc)
            zz=zz-start+radiation.boundaries_m[host]
        k0=2*np.pi/(model.bragg_wavelength_nm*1e-9); beta0=(2*np.pi if len(orders)==4 else 4*np.pi/np.sqrt(3))/(period*1e-9); cache={}
        def integral(m,n,metric):
            rounded=int(round(metric))
            if rounded not in cache:
                if radiation is None:
                    decay=np.sqrt(metric*beta0**2-k0*k0*model.average_pc_epsilon)
                    green=np.exp(-decay*abs(zz[:,None]-zz[None,:]))/(2*decay)
                else: green=local_green_kernel(radiation,host,model.bragg_wavelength_nm,q_parallel_per_m=np.sqrt(metric)*beta0).matrix(zz)
                cache[rounded]=weighted.conj()@green@weighted
            return cache[rounded]
        independent=independent_high(model,orders,pol,shell,integral)
        stored=model.c2d_m if len(orders)==4 else model.ch_m
        error=np.linalg.norm(independent-stored)/max(np.linalg.norm(stored),1e-30)
        check(case['id']+' Cartesian projector high-order assembly',error,3e-11)
        check(case['id']+' passive-projection correction relative to C',model.passivity_correction_m/max(np.linalg.norm(model.coupling_m),1e-30),2e-12)
        beta_ref=k0*model.effective_index; delta_ref=(beta_ref**2-beta0**2)/(2*beta0)
        base=period/model.bragg_wavelength_nm
        expected=base+(model.eigenvalues_m.real-delta_ref)*(period*1e-9)/(2*np.pi*model.group_index)*(beta0/beta_ref)
        diagram=(square_four_wave_band_diagram(model.coupling_m,period,model.bragg_wavelength_nm,model.effective_index,group_index=model.group_index,points=21)
                 if len(orders)==4 else tr.triangular_band_diagram(model,points=21))
        check(case['id']+' physical frequency origin',np.max(abs(diagram.normalized_frequency[:,10]-expected)),2e-13)
        matrices[case['id']]={'raw_matrices':captured,'independent_high_order_relative_error':float(error),'passivity_correction_per_m':model.passivity_correction_m,'D':args.order,'dz_nm':args.dz,'neff':model.effective_index,'ng':model.group_index,'background_detuning_per_m':float(delta_ref),'delta_frequency_slope_factor_beta0_over_beta':float(beta0/beta_ref)}
    details['canonical_geometry_matrix_checks']=matrices
    # Document actual guard behavior; failures remain findings, not hidden
    # passes. These negative controls are repeated after remediation.
    negative={}
    for label,construct in [('square absorbing epsilon',lambda:SquareLatticeCell(3.4**2,(Ellipse((0.,0.),(.1,.1),epsilon=1-.2j),))),('triangular absorbing epsilon',lambda:tr.TriangularLatticeCell(3.4**2,(tr.TriangularEllipse(epsilon=1-.2j),))),('material negative radiation projected',lambda:sq._project_radiation_passive(np.diag([1j,-.2j])) )]:
        try: construct();negative[label]={'rejected':False}
        except ValueError as error:negative[label]={'rejected':True,'reason':str(error)}
    details['negative_controls']=negative
    modules=['geometry','vertical','hybrid_vertical','coupling','three_d_cwt','reflector_cwt','triangular_six_wave','matched300_yag','stratified_optics','band_structure','materials']
    output={'status':'passed' if all(c['passed'] for c in checks) and all(v['rejected'] for v in negative.values()) else 'failed','checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),'negative_controls_all_rejected':all(v['rejected'] for v in negative.values()),'checks':checks,'details':details,'audit_script_sha256':sha(__file__),'source_sha256':{f'src/pcselsim/{m}.py':sha(ROOT/f'src/pcselsim/{m}.py') for m in modules},'primary_sources':{str(p):sha(p) for directory in ['CVYT7MQ6','4WEC2TRI'] for p in Path('C:/Users/19254/Zotero/storage',directory).glob('*.pdf')},'scope':'Analytic/quadrature/projector/Green checks plus raw pre-projection inspection. Not independent 3D Maxwell or experimental validation. No frozen numerical result modified.','elapsed_seconds':time.monotonic()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':output['status'],'checks':f"{output['checks_passed']}/{len(checks)}",'negative_controls':negative,'matrix_checks':matrices,'elapsed_seconds':output['elapsed_seconds']},ensure_ascii=False))
    if output['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
