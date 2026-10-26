"""Frequency-close larger-aperture Yb:YAG alternatives and lateral challengers."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from optimize_ybyag_cavity import Design, closed_candidate
from audit_ybyag_lateral_spectrum import main as audit_main
from close_ybyag_lateral_challengers import main as close_main


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design',default='tmp/yag_fixed_best_20261021.json')
    p.add_argument('--output',required=True)
    p.add_argument('--sizes',type=float,nargs='+',default=[500.,600.])
    p.add_argument('--grid',type=int,default=17);p.add_argument('--truncation',type=int,default=8)
    a=p.parse_args(argv);base=Design(**json.loads(Path(a.design).read_text())[0])
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    for size in a.sizes:
        design=replace(base,id=f'gain500_air120_L{size:g}',device_um=size)
        source=root/f'L{size:g}_four_closed'
        closed_candidate(design,source,grid=a.grid,truncation=a.truncation,
            pump_powers=(.01,.03,.1,.3,1.),reference_dense_limit=500,vertical_step_nm=2.5)
        challenge=root/f'L{size:g}_full_spectrum'
        audit_main([str(source),'--output',str(challenge),'--branch','TE0_A','--truncation',str(a.truncation),
                    '--vertical-step-nm','2.5'])
        close_main([str(source),str(challenge),'--output',str(root/f'L{size:g}_19poles'),
                    '--count','16','--truncation',str(a.truncation),'--vertical-step-nm','2.5'])


if __name__=='__main__':main()
