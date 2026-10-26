"""Scientific geometry/actual-field/polarization overview of four selected cases.

中文：仅读取冻结复场。近场为基本波包络功率和，远场为复Jones强度，不生成
拟合光斑或缩放阈值。偏振椭圆直接由轴上Jones向量画出，不暗示圆斑为圆偏振。
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from compare_matched300_results import plot_geometry
from pcselsim.triangular_finite import _cartesian_point_fourier_sum
from pcselsim.design_objectives import beam_metrics

OUT=ROOT/'results/current/strict_rerun_20261025/comparison'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def jones(case,raw):
    key=case['field_bindings'];fxk,fyk=key['far_x_key'],key['far_y_key']
    if fxk in raw.files and fyk in raw.files:
        axis=raw[key['angle_key']]
        return axis,raw[fxk],raw[fyk],'saved_actual_complex_Jones'
    axis=np.linspace(-1.,1.,201)
    rx,ry=raw[key['radiation_x_key']],raw[key['radiation_y_key']]
    if case['lattice']=='square':
        n=rx.shape[0];xy=((np.arange(n)+.5)/n-.5)*300e-6;xx,yy=np.meshgrid(xy,xy)
    else:xx,yy=raw['x_um']*1e-6,raw['y_um']*1e-6
    wave=2*np.pi/(case['wavelength_nm']*1e-9)*np.tan(np.deg2rad(axis))
    fx,fy=_cartesian_point_fourier_sum(xx.ravel(),yy.ravel(),rx.ravel(),ry.ravel(),wave)
    tx,ty=np.meshgrid(np.deg2rad(axis),np.deg2rad(axis));ob=np.cos(tx)+np.cos(ty)-1
    return axis,fx*ob,fy*ob,'exact_saved_radiative_envelope_transform'

def main():
    manifest=read(OUT/'selected_cases.json');cases=manifest['cases']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(4,4,figsize=(16.5,16.2),layout='constrained')
    polar_rows=[];checks=[]
    labels={'semiconductor_square':'Semiconductor square','semiconductor_triangular':'Semiconductor triangular',
        'YbYAG_square':'Yb:YAG square','YbYAG_triangular':'Yb:YAG triangular'}
    for i,case in enumerate(cases):
        key=case['field_bindings'];source=Path(case['fields_path']);initial=sha(source)
        with np.load(source,allow_pickle=False) as raw:
            geometry=read(case['geometry_path'])
            # Schema adapter only; never rewrite the frozen physical boundary.
            if 'lattice_constant_nm' not in geometry:geometry={**geometry,'lattice_constant_nm':geometry['period_nm']}
            plot_geometry(axes[i,0],geometry,case['lattice'])
            axes[i,0].set_title(labels[case['id']]+f"\na={case['period_nm']:.4f}nm")
            field=raw[key['field_key']];nf=np.sum(abs(field)**2,axis=0);nf/=nf.max()
            if case['lattice']=='square':axes[i,1].imshow(nf,origin='lower',extent=[-150,150,-150,150],cmap='inferno',vmin=0,vmax=1)
            else:axes[i,1].tripcolor(raw['x_um'],raw['y_um'],nf,shading='gouraud',cmap='inferno',vmin=0,vmax=1)
            axes[i,1].set(xlim=(-150,150),ylim=(-150,150),xlabel='x (um)',ylabel='y (um)',
                title=f"Near field\ng1={case['g_cm1']:.4f}cm$^{{-1}}$; lambda={case['wavelength_nm']:.5f}nm")
            axes[i,1].set_aspect('equal')
            axis,fx,fy,method=jones(case,raw);ff=abs(fx)**2+abs(fy)**2;ff/=ff.max()
            axes[i,2].pcolormesh(axis,axis,ff,shading='nearest',cmap='inferno',vmin=0,vmax=1)
            axes[i,2].contour(axis,axis,ff,[.5],colors='cyan',linewidths=.7)
            axes[i,2].set(xlim=(-.5,.5),ylim=(-.5,.5),xlabel='theta_x (deg)',ylabel='theta_y (deg)',
                title=f"Vector far field\naxis={case['axis']:.4f}; radial L2={case['radial']:.4f}")
            axes[i,2].set_aspect('equal')
            j=int(np.argmin(abs(axis)));ex,ey=fx[j,j],fy[j,j];norm=np.sqrt(abs(ex)**2+abs(ey)**2)
            phase=np.linspace(0,2*np.pi,721);xx=np.real(ex/norm*np.exp(1j*phase));yy=np.real(ey/norm*np.exp(1j*phase))
            axes[i,3].plot(xx,yy,lw=1.5,color='#206D82');axes[i,3].axhline(0,color='.8',lw=.6);axes[i,3].axvline(0,color='.8',lw=.6)
            pol=case['polarization'];on=pol['on_axis'];s=on['normalized_stokes_s1_s2_s3']
            axes[i,3].set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),xlabel='Re(Ex exp(iwt)) / |E|',ylabel='Re(Ey exp(iwt)) / |E|',
                title=f"On-axis polarization\nazimuth={on['linear_axis_angle_deg']:.2f}deg; S3/S0={s[2]:.5f}")
            axes[i,3].set_aspect('equal')
            calculated=beam_metrics(axis,fx,fy,angle_y_deg=axis,
                requested_view_deg=case['beam'].get('requested_half_window_deg',1.),
                source_nyquist_deg=case['beam'].get('source_nyquist_deg'))
            errors={k:abs(calculated[k]-case['beam'][k]) for k in ('ellipticity','center_to_peak','radial_anisotropy_l2')}
            if max(errors.values())>2e-6:raise RuntimeError(('Far-field metadata mismatch',case['id'],errors))
            s0=abs(ex)**2+abs(ey)**2
            direct_stokes=np.array([abs(ex)**2-abs(ey)**2,2*np.real(ex*ey.conjugate()),2*np.imag(ex*ey.conjugate())])/s0
            stokes_error=float(np.max(abs(direct_stokes-np.asarray(s))))
            if stokes_error>2e-10:raise RuntimeError(('Jones polarization phase mismatch',case['id'],stokes_error))
            polar_rows.append(dict(id=case['id'],on_axis_s1=s[0],on_axis_s2=s[1],on_axis_s3=s[2],
                on_axis_azimuth_deg=on['linear_axis_angle_deg'],core_DOLP=pol['core']['degree_of_linear_polarization'],
                integrated_DOLP=pol['integrated']['degree_of_linear_polarization']))
            checks.append(dict(id=case['id'],field_bindings=key,field_source=str(source),field_sha256=initial,
                farfield_method=method,readback_beam_errors=errors,sampling_qualified_beam=calculated,
                direct_on_axis_stokes=direct_stokes.tolist(),maximum_stokes_error=stokes_error))
        if sha(source)!=initial:raise RuntimeError('Frozen complex field changed during plotting')
    fig.suptitle('Post-audit static CWT results: all nominal300 x300um square boundaries\nEach near/far panel normalized independently; spatial circularity differs from polarization',fontsize=14)
    fig.savefig(OUT/'four_cavity_overview.png',dpi=185);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,group,title in zip(axes,(cases[:2],cases[2:]),('Semiconductor','Yb:YAG')):
        for case in group:
            values=case['gains_cm1'][:6]
            ax.plot(np.arange(1,len(values)+1),values,'o-',label=case['lattice'])
        ax.set(xlabel='Retained closed-mode rank',ylabel='Required material gain (cm^-1)',title=title+'; bounded searched spectrum')
        ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'closed_mode_thresholds.png',dpi=185);plt.close(fig)
    with (OUT/'four_cavity_polarization.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(polar_rows[0]));writer.writeheader();writer.writerows(polar_rows)
    (OUT/'figure_provenance.json').write_text(json.dumps(dict(cases=checks,polarization=polar_rows,
        scope='Own actual-frequency first-mode physical fields; static normalized evidence, not relative emitted powers or full-wave validation.'),indent=2),encoding='utf8')
    print('Saved four-cavity scientific overview, threshold figure and polarization CSV')

if __name__=='__main__':main()
