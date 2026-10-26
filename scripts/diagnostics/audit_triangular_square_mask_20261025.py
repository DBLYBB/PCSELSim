"""Read-only staircase-boundary diagnostics for square masks on axial grids.

中文：检查名义方腔、均匀节点控制面积和特征链midpoint边界之间的差别。
这些是离散几何诊断，不定义一个新的物理腔尺寸，也不用于按面积比例修正阈值。
No production solver, geometry or saved optical result is modified.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import LinearNDInterpolator
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from pcselsim.triangular_finite import build_device_grid


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def continuous_chord(origin,direction,half):
    lower,upper=-np.inf,np.inf
    for coordinate,velocity in zip(origin,direction):
        if abs(velocity)<1e-15:
            if abs(coordinate)>half+1e-10:return 0.
            continue
        roots=sorted([(-half-coordinate)/velocity,(half-coordinate)/velocity])
        lower=max(lower,roots[0]);upper=min(upper,roots[1])
    return max(0.,float(upper-lower))


def inspect_grid(n,length=200.):
    half=length/2;grid=build_device_grid(n,half,'square');h=grid.spacing_m*1e6
    cell_area=np.sqrt(3)/2*h*h;area=grid.points*cell_area
    directions={
        'horizontal':(np.array([1.,0.]),grid.axial_k-grid.axial_j,lambda line:np.array([.5*h*line,np.sqrt(3)/2*h*line])),
        'minus60deg':(np.array([.5,-np.sqrt(3)/2]),grid.axial_k,lambda line:np.array([.5*h*line,np.sqrt(3)/2*h*line])),
        'plus60deg':(np.array([.5,np.sqrt(3)/2]),grid.axial_j,lambda line:np.array([.5*h*line,-np.sqrt(3)/2*h*line]))}
    chords={}
    for name,(direction,labels,origin) in directions.items():
        physical=[];discrete=[]
        for label in np.unique(labels):
            physical.append(continuous_chord(origin(label),direction,half))
            discrete.append(np.count_nonzero(labels==label)*h)
        physical=np.array(physical);discrete=np.array(discrete)
        center=labels==0;center_physical=continuous_chord(np.zeros(2),direction,half)
        chords[name]={'sampled_lines':len(physical),'central_coefficient_nodes':int(np.count_nonzero(center)),
            'central_exact_square_chord_um':center_physical,'central_midpoint_link_chord_um':float(np.count_nonzero(center)*h),
            'central_relative_chord_error':float(np.count_nonzero(center)*h/center_physical-1),
            'chord_weighted_absolute_boundary_error':float(np.sum(abs(discrete-physical))/np.sum(physical)),
            'sum_midpoint_chords_times_interray_spacing_um2':float(discrete.sum()*np.sqrt(3)/2*h),
            'sampled_exact_chords_quadrature_um2':float(physical.sum()*np.sqrt(3)/2*h)}
    return {'N':n,'nominal_square_side_um':length,'nodes':grid.points,'step_um':h,
        'node_x_span_um':float(np.ptp(grid.x_um)),'node_y_span_um':float(np.ptp(grid.y_um)),
        'max_y_row_index':int(np.max(abs(grid.axial_k-grid.axial_j))),
        'node_control_cell_area_um2':float(cell_area),'uniform_node_control_area_um2':float(area),
        'uniform_node_control_area_relative_error':float(area/(length*length)-1),
        'characteristic_chords':chords}


def main():
    out=ROOT/'results/current/physics_audit_20261025/triangular_square_mask_review.json'
    rows=[inspect_grid(n) for n in sorted(set(range(15,66))|{13,91})]
    data=ROOT/'results/current/yag_feasibility_20261025/sizes';pair=[];source={}
    for n in (25,26):
        p=data/f'triangular_L200_D24_dz1_N{n}_K4.json'
        if not p.exists():continue
        x=json.loads(p.read_text(encoding='utf-8'));source[str(p.relative_to(ROOT))]=digest(p)
        if x.get('status')=='completed':
            first=min(x['modes'],key=lambda r:r['material_gain_cm1'])
            pair.append({'N':n,'order':x['order'],'dz_nm':x['dz_nm'],'wavelength_nm':first['wavelength_nm'],'g_material_cm1':first['material_gain_cm1'],'field_name':first['name'],'scope':'minimum returned mode; a common mode identity across grids is not established by this scalar comparison alone'})
    summary={'status':'geometry_diagnostic_completed_not_a_convergence_certificate',
        'scope':'Nominal L and actual axial node masks are separated. Uniform node area and midpoint characteristic chords quantify the staircased discrete boundary; they are not a unique new physical cavity boundary or a correction to optical losses/pump volume.',
        'derivation':'For phi=M u=(u_out+u_in)/2 at each coefficient point, the incoming/outgoing link boundaries lie half a characteristic step from the first/last coefficient point. A central ray with n nodes therefore has midpoint boundary chord n*h. The continuous square chord is 2R/max(|e_x|,|e_y|). Uniform axial node area is sqrt(3)/2*h^2.',
        'findings':[
            'The N25/N26 difference is a floor(2N/sqrt(3)) row-inclusion jump, not a universal odd-even error.',
            'N25→26 changes the maximum y-row from28 to30. Effective characteristic chord errors can oscillate instead of decreasing monotonically.',
            'N13 multiples N26,N39,N52,N65,N91 share the same relative y node extent because floor(2N/sqrt(3))/N=15/13 on this range. A convergence sequence using only this alignment does not sample all boundary phases.',
            'Do not multiply the threshold by nominal_area/node_area or alter the pump volume to hide this boundary error. Changing only weights or individual link lengths can break the discrete flux identity.',
            'The observed gain change is consistent in direction and scale with the boundary change, but geometry diagnostics alone do not prove the whole gain difference is a boundary effect.'],
        'focused_N25_N26':[r for r in rows if r['N'] in (25,26)],'optical_pair':pair,
        'all_grid_diagnostics':rows,'source_sha256':{**source,'src/pcselsim/triangular_finite.py':digest(ROOT/'src/pcselsim/triangular_finite.py'),'src/pcselsim/finite_discretization.py':digest(ROOT/'src/pcselsim/finite_discretization.py'),'scripts/diagnostics/audit_triangular_square_mask_20261025.py':digest(__file__)}}
    if len(pair)==2:summary['N26_gain_relative_to_N25']=pair[1]['g_material_cm1']/pair[0]['g_material_cm1']-1
    if len(pair)==2:
        # Shared interior only: avoids interpreting interpolation across one
        # mask's missing edge rows as a new optical mode.
        axis=np.linspace(-95.,95.,61);xx,yy=np.meshgrid(axis,axis)
        points=np.column_stack((xx.ravel(),yy.ravel()));sampled=[]
        for row in pair:
            p=data/f'triangular_L200_D24_dz1_N{row["N"]}_K4.npz'
            source[str(p.relative_to(ROOT))]=digest(p)
            grid=build_device_grid(row['N'],100.,'square')
            with np.load(p,allow_pickle=False) as stored:
                field=stored[row['field_name']+'_field']
            sampled.append(LinearNDInterpolator(np.column_stack((grid.x_um,grid.y_um)),field.T)(points))
        common=np.isfinite(sampled[0]).all(axis=1)&np.isfinite(sampled[1]).all(axis=1)
        a,b=sampled[0][common],sampled[1][common]
        overlap=abs(np.vdot(a,b))**2/(np.vdot(a,a).real*np.vdot(b,b).real)
        summary['common_interior_field_overlap']={'normalized_complex_field_overlap_squared':float(overlap),'samples':int(np.count_nonzero(common)),'comparison_square_um':[-95.,95.],
            'scope':'Complex six-wave fields interpolated only on a common interior; high overlap supports branch correspondence but is not a cross-grid completeness proof.'}
        summary['source_sha256'].update(source)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':summary['status'],'N25_N26':summary['focused_N25_N26'],'optical_pair':pair,'gain_relative_change':summary.get('N26_gain_relative_to_N25')},ensure_ascii=False))


if __name__=='__main__':main()
