"""Analytical Q requirement, not a prediction of a fabricated microcavity.

中文：Q=omega*tau_ph、tau_ph=1/(vg*alpha_power)。只画满足名义材料
增益上限的必要条件；Q、模体积与泵耦合并不能在真实设计中任意独立指定。
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/current/yag_feasibility_20261025'


def main():
    wavelength_m=1030e-9;group_index=1.82;gain_weight=.90
    sigma_ap,sigma_ep=.70e-20,.10e-20  # cm^2, explicitly nominal
    sigma_el,sigma_al=2.0e-20,.126e-20
    fmax=sigma_ap/(sigma_ap+sigma_ep)
    quality=np.geomspace(1e3,1e7,300)
    # Intensity loss in /m; required BULK material gain in /cm.
    required=2*np.pi*group_index/(quality*wavelength_m*gain_weight)/100
    records=[]
    fig,ax=plt.subplots(figsize=(7.5,4.7),layout='constrained')
    ax.loglog(quality,required,color='#222222',lw=2,label=r'Required gain: $2\pi n_g/(Q\lambda q_g)$')
    for nt,color in ((.69e21,'#009E73'),(1.38e21,'#0072B2'),(2.76e21,'#D55E00')):
        ceiling=nt*((sigma_el+sigma_al)*fmax-sigma_al)
        qmin=2*np.pi*group_index/(wavelength_m*gain_weight*100*ceiling)
        ax.axhline(ceiling,color=color,ls='--',label=fr'$N_t={nt/1e21:g}\times10^{{21}}$ cm$^{{-3}}$, $Q_{{min}}\approx${qmin:.0f}')
        assert np.isclose(2*np.pi*group_index/(qmin*wavelength_m*gain_weight)/100,ceiling)
        records.append(dict(dopant_density_cm3=nt,pump_limited_gain_ceiling_cm1=ceiling,
                            necessary_loaded_Q=qmin))
    ax.set(xlabel='Assumed cavity-loss Q (Yb reabsorption separate)',ylabel='Material intensity gain (cm$^{-1}$)',
           title='Necessary gain condition for a Yb:YAG microcavity')
    ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
    fig.supxlabel('Illustrative ng=1.82, qg=0.90, 1030 nm; nominal scalar spectroscopy.\nQ and overlap must come from a real optical design. This is not a WGM/PCSEL simulation or sufficiency proof.',fontsize=9)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/'YAG_necessary_Q.png',dpi=210);plt.close(fig)
    (OUT/'necessary_Q.json').write_text(json.dumps(dict(wavelength_nm=1030.,group_index=group_index,
        full_mode_gain_weight=gain_weight,pump_limited_fraction=fmax,records=records,
        quality_definition='Passive cavity-loss Q includes outcoupling and parasitic loss but excludes Yb reabsorption already present in the net-gain term; do not use an unpumped measured Q containing reabsorption without separating it.',
        claim_boundary='Analytic necessary gain condition only; excludes pump coupling, spectral competition, mode volume, spontaneous emission, nonradiative quenching and thermal effects.'),indent=2),encoding='utf-8')
    print(json.dumps(records))


if __name__=='__main__':main()
