"""English diagnostic panels from the retained actual complex finalist fields."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pcselsim.fabrication import TRIANGULAR_DIRECT,inclusion_boundary
from pcselsim.polarization import polarization_metrics,stokes_maps
from scripts.optimization.matched300_triangular_search import ThreeHole,candidate
from scripts.optimization.triangular_semiconductor_search import json_write


def report(directory):
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    spec=ThreeHole(**summary["three_hole_spec"]);lowest=summary["lowest"];rank=lowest["seed_rank"]
    z=np.load(directory/"frequency_closed/frequency_closed_fields.npz")
    field=z[f"root_{rank}_fields"];fx=z[f"root_{rank}_far_x"];fy=z[f"root_{rank}_far_y"];axis=z["angle_deg"]
    pol=polarization_metrics(axis,fx,fy,core_radius_deg=.2);maps=stokes_maps(fx,fy,intensity_floor=.01)
    json_write(directory/"polarization.json",pol)
    np.savez_compressed(directory/"stokes_maps.npz",angle_deg=axis,**maps)
    power=abs(fx)**2+abs(fy)**2;maximum=power.max();power/=maximum
    holes=candidate(spec).inclusions
    fig,axes=plt.subplots(2,3,figsize=(14,8.5),constrained_layout=True)
    for m in range(-1,2):
        for n in range(-1,2):
            shift=TRIANGULAR_DIRECT@np.array((m,n))
            for h in holes:
                b=(inclusion_boundary(h,TRIANGULAR_DIRECT)+shift)*spec.lattice_nm
                axes[0,0].fill(b[:,0],b[:,1],color="#197b99",ec="#103f50",lw=.5)
    primitive=np.column_stack((np.zeros(2),TRIANGULAR_DIRECT[:,0],TRIANGULAR_DIRECT.sum(1),TRIANGULAR_DIRECT[:,1],np.zeros(2)))*spec.lattice_nm
    axes[0,0].plot(primitive[0],primitive[1],"--",color="darkorange",lw=1.4)
    axes[0,0].set(xlim=(-.3*spec.lattice_nm,1.9*spec.lattice_nm),ylim=(-.8*spec.lattice_nm,.8*spec.lattice_nm),aspect="equal",
        title=f"Actual triangular motif: a = {spec.lattice_nm:.3f} nm",xlabel="x (nm)",ylabel="y (nm)")
    near=np.sum(abs(field)**2,axis=0);near/=near.max()
    im=axes[0,1].tripcolor(z["x_um"],z["y_um"],near,shading="gouraud",cmap="inferno",vmin=0,vmax=1)
    axes[0,1].set(aspect="equal",xlim=(-150,150),ylim=(-150,150),title="Nominal 300 x 300 um square: actual pole",xlabel="x (um)",ylabel="y (um)")
    fig.colorbar(im,ax=axes[0,1],label="Normalized envelope intensity")
    im=axes[0,2].imshow(power,origin="lower",extent=(axis[0],axis[-1],axis[0],axis[-1]),cmap="inferno",vmin=0,vmax=1)
    axes[0,2].set(xlim=(-.35,.35),ylim=(-.35,.35),xlabel="theta x (deg)",ylabel="theta y (deg)",
        title=f"Total far field: ellipse {lowest['beam']['ellipticity']:.3f}")
    fig.colorbar(im,ax=axes[0,2],label="Normalized |Ex|^2 + |Ey|^2")
    ix=int(np.argmin(abs(axis)))
    axes[1,0].plot(axis,power[ix],label="Total, theta y = 0",color="black")
    axes[1,0].plot(axis,abs(fx[ix])**2/maximum,label="x polarization",color="#117da4")
    axes[1,0].plot(axis,abs(fy[ix])**2/maximum,label="y polarization",color="#d75c36")
    axes[1,0].set(xlim=(-.5,.5),ylim=(0,1.05),xlabel="theta x (deg)",ylabel="Common-normalized intensity",title="Jones-component cross section")
    axes[1,0].legend(fontsize=8)
    im=axes[1,1].imshow(np.ma.masked_where(~maps["valid_mask"],maps["s3_over_s0"]),origin="lower",extent=(axis[0],axis[-1],axis[0],axis[-1]),cmap="coolwarm",vmin=-1,vmax=1)
    axes[1,1].set(xlim=(-.35,.35),ylim=(-.35,.35),xlabel="theta x (deg)",ylabel="theta y (deg)",title="S3 / S0; intensity mask >= 1%")
    fig.colorbar(im,ax=axes[1,1],label="Signed circular Stokes fraction")
    rows=json.loads((directory/"frequency_closed/frequency_closed_poles.json").read_text(encoding="utf-8"))
    rows=sorted([r for r in rows if r["status"]=="completed"],key=lambda r:r["g_th_cm"])
    axes[1,2].plot(np.arange(len(rows))+1,[r["g_th_cm"] for r in rows],"o-",color="#117da4")
    axes[1,2].set(xlabel="Frequency-closed finite pole (gain sorted)",ylabel="Material threshold gain (cm^-1)",
        title=f"Nearest tested threshold gap = {summary['g_th_gap_fraction']:.2%}")
    fig.suptitle(f"Triangular PCSEL: actual lambda {lowest['wavelength_nm']:.5f} nm | g1 {lowest['g_th_cm']:.3f} /cm | N={summary['N']}, D={summary['D']}, dz={summary['dz_nm']:g} nm",fontsize=13)
    fig.savefig(directory/"final_overview.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
    for ax,key,label in zip(axes,("s1_over_s0","s2_over_s0","s3_over_s0"),("S1 / S0","S2 / S0","S3 / S0"),strict=True):
        im=ax.imshow(np.ma.masked_where(~maps["valid_mask"],maps[key]),origin="lower",extent=(axis[0],axis[-1],axis[0],axis[-1]),cmap="coolwarm",vmin=-1,vmax=1)
        ax.set(xlim=(-.35,.35),ylim=(-.35,.35),title=label,xlabel="theta x (deg)",ylabel="theta y (deg)")
        fig.colorbar(im,ax=ax)
    fig.savefig(directory/"polarization_stokes.png",dpi=180);plt.close(fig)
    return pol


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True);a=p.parse_args();report(a.input)


if __name__=="__main__":main()
