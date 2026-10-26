"""Freeze the completed scientific YAG-study index without rerunning models.

中文：只整理已通过的QA/科学图/文档索引，不重跑主仿真，不修改选定几何或复场。
Preserve the earlier partial wrapper-review snapshot separately from final QA.
"""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/current/strict_rerun_20261025'
OUT=BASE/'yag_feasibility'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    qa_path=OUT/'independent_output_qa.json';qa=read(qa_path)
    if qa['status']!='completed' or not qa['all_requested_runs_complete']:
        raise RuntimeError('Both lattices, rates and key size controls must be completed')
    for path,digest in qa['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError(f'QA input changed: {path}')
    for plot in qa['plots']:
        if sha(plot['path'])!=plot['sha256']:raise RuntimeError('A QA scientific figure changed afterwards')
    for lattice in ('square','triangular'):
        audit=read(BASE/f'cross_audit_yag_{lattice}.json')
        if audit['status']!='passed':raise RuntimeError('Independent optical/shared-pump cross audit failed')
    initial_path=OUT/'audit_history/wrapper_initial_partial_snapshot.json'
    initial_path.parent.mkdir(exist_ok=True)
    wrapper_path=BASE/'cross_audit_yag_feasibility_wrapper.json'
    if not initial_path.exists():initial_path.write_bytes(wrapper_path.read_bytes())
    initial=read(initial_path)
    pulse_count=sum(len(r['pulse']['records']) for r in qa['reports'].values())
    cw_count=sum(len(c['checks']) for r in qa['reports'].values() for c in r['cw'])
    size_count=sum(len(s['independent_stored_pencil_checks']) for r in qa['reports'].values() for s in r['sizes'])
    target_failures=sum(len(s['failed_census_targets']) for r in qa['reports'].values() for s in r['sizes'])
    closure_failures=sum(len(s['closure_failures']) for r in qa['reports'].values() for s in r['sizes'])
    size_configurations=sum(len(r['sizes']) for r in qa['reports'].values())
    wrapper=dict(status='passed',initial_static_review_snapshot_path=str(initial_path),
        initial_findings=initial.get('findings',[]),full_output_qa_path=str(qa_path),full_output_qa_sha256=sha(qa_path),
        completed_pulse_cases_checked=pulse_count,completed_CW_points_checked=cw_count,
        completed_size_root_readbacks=size_count,mode_counts={l:r['retained_mode_count'] for l,r in qa['reports'].items()},
        representative_tighter_rtol_cases=[q for r in qa['reports'].values() for q in r['pulse']['accuracy_challenges']],
        checks=[dict(name='mode_specific_merged_inputs_and_lowest_actual_cold_pulse_mode',passed=True),
            dict(name='external_pulse_J_eta0p7_vs_internal_CW_first_entry_W',passed=True),
            dict(name='stored_population_photon_energy_and_pump_boundary_budgets',passed=True),
            dict(name='uniform_one_mode_pulse_separate_from_spatial_multimode_CW',passed=True)],
        scope='Static wrapper/core review plus completed stored-output independent QA. No primary model reruns. Only one representative pulse per lattice received10x tighter rtol; not all96. No full-wave, measured-spectrum, thermal or dynamical-stability certification.')
    save(wrapper_path,wrapper)
    files=[qa_path,OUT/'all_sizes_summary.json',OUT/'triangular_sizes_summary.json',OUT/'triangular_size_summary.csv',
        OUT/'triangular_size_report_zh.md',OUT/'CW_input_output.csv',OUT/'pulse_input_output.csv',OUT/'size_threshold.csv',
        ROOT/'docs/strict_yag_feasibility_20261025_zh.md',BASE/'cross_audit_yag_square.json',
        BASE/'cross_audit_yag_triangular.json',wrapper_path,OUT/'executed_output_review.py',Path(__file__)]
    files += [Path(p['path']) for p in qa['plots']]
    manifest=dict(status='completed',generated_utc=datetime.now(timezone.utc).isoformat(),
        report_path=str(ROOT/'docs/strict_yag_feasibility_20261025_zh.md'),
        final_QA_path=str(qa_path),scientific_plots=[p['path'] for p in qa['plots']],
        pulse_cases=pulse_count,CW_points=cw_count,size_configurations=size_configurations,
        stored_size_root_readbacks=size_count,bounded_spectral_target_failures_retained=target_failures,
        frequency_closure_failures=closure_failures,
        selected_case_sha256={l:sha(BASE/('YbYAG_'+l)/'selected_case.json') for l in ('square','triangular')},
        source_qa_sha256=sha(qa_path),output_sha256={str(p):sha(p) for p in files},
        scientific_scope='Final unit cells/periods/stacks held fixed during size study; nominal constant spectroscopy and ideal pump boundary assumptions. Required material gains and conditional wavelength-window thresholds are not measured gain spectra or physical filters. Finite-pulse results are uniform one-mode reductions; CW is stationary spatial multimode. No minimum feasible size inferred between discrete points.',
        power_scope='CW: total internal first-entry W; pulses: external J before assumed eta=.7 coupling. Dual-edge total power is split between two150um2 entrances at300um. Ideal20-pass relay uses98% return; no fabricated pump optics certification.')
    save(OUT/'final_delivery_manifest.json',manifest)
    text=['# Strict Yb:YAG scientific feasibility results','',
        f'Completed: {pulse_count} uniform one-mode pulse cases, {cw_count} stationary spatial multimode CW points, {size_configurations} fixed-period size configurations, {size_count} stored original-pencil root readbacks. Two representative pulses (one per lattice) received10x tighter rtol.','',
        '- Main report: [Chinese scientific report](../../../../docs/strict_yag_feasibility_20261025_zh.md).',
        '- Final evidence: `final_delivery_manifest.json`, `independent_output_qa.json`, `all_sizes_summary.json`.',
        '- Triangular-only summary: `triangular_sizes_summary.json`, `triangular_size_summary.csv`, `triangular_size_report_zh.md`.',
        '- Original square-only summaries remain unchanged: `sizes_summary.json`, `square_size_summary.csv`, `size_report_zh.md`, `SIZE_README.md`.',
        '- Scientific figures: `science_plots/`; raw rate records: `cw/` and `pulsed/`; raw size fields: `sizes/`.','',
        'CW input is internal first-entry W; pulse input is external J with assumed70% entry coupling. Positive spontaneous-emission energy alone is not a lasing criterion. Null thresholds remain null. Exploratory D12 and checked D32 points/merged300um anchors are identified explicitly.','',
        f'当前支持声明模型内的条件可行性，不证明高外部效率、可制造、完整实测增益谱或热/动态稳定。{target_failures}个未成功有界谱目标原样保留，接受根的频率闭合失败为{closure_failures}；这些是不同的统计。解析/后处理执行失败另留size_execution_failures。','',
        'Regenerate existing-output QA/figures/report, without primary simulations:',
        '`E:/Anaconda/envs/env00/python.exe scripts/diagnostics/review_strict_yag_feasibility_20261025.py`','',
        'Refresh this frozen delivery index afterwards:',
        '`E:/Anaconda/envs/env00/python.exe scripts/diagnostics/finalize_strict_yag_feasibility_delivery_20261025.py`','']
    (OUT/'FEASIBILITY_README.md').write_text('\n'.join(text),encoding='utf8')
    print('completed',pulse_count,cw_count,size_count,'targets_failed',target_failures,flush=True)


if __name__=='__main__':main()
