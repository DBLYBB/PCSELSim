"""Check TE-family applicability of the same physical process perturbations.

This flags new/near-cutoff isolated TE families; it does not silently add their
uncomputed lateral poles to a twelve-branch continuation sensitivity screen.
"""
from matched300_yag_tolerance import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("case",type=Path)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.case/"summary.json").read_text());lattice,design,period=summary["lattice"],summary["design"],summary["period_nm"]
    cell=geometry(lattice,design);rows=[];a.output.mkdir(parents=True,exist_ok=True)
    for name,holes,perturbation in variants(cell,lattice,period):
        d={**design,"_yag_explicit_holes":encode(holes,lattice)}
        audit=longitudinal_audit(lattice,d,period,1.)
        row=dict(name=name,perturbation=perturbation,audit=audit,
            nominal_family_set_remains_applicable=audit["max_isolated_TE_count"]==summary["isolated_TE_count"]
                and audit["all_tested_families_remote_reflector_small"])
        rows.append(row);save(a.output/"family_cases.json",rows)
        print(name,audit["max_isolated_TE_count"],audit["all_tested_families_remote_reflector_small"],flush=True)
    save(a.output/"summary.json",dict(source_case=str(a.case),dz_nm=1.,analysis_band_nm=[1000,1060],results=rows,
        all_nominal_family_sets_remain_applicable=all(r["nominal_family_set_remains_applicable"] for r in rows),
        scope="Five wavelength TE-family and mirror-loading samples per physical periodic perturbation; new families are flagged, not certified by continuing only the nominal branch set"))


if __name__=="__main__":main()
