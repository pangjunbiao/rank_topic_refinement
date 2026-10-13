"""Portable compact Step 8 reports with frozen-source verification and logging."""
from __future__ import annotations
import io
import threading
import traceback
import unittest
from pathlib import Path
from rank_refinement.runtime.artifacts import artifact_manifest,digest,file_hash,read_json,write_json,write_text
from rank_refinement.runtime.state import ProjectLock
from rank_refinement.evaluation.resource_usage import ResourceMonitor
from step08_assembly.integrity import protect,unchanged,verify_tree
from step08_assembly.runner import ReportLog,zip_files

PROTOCOL='STEP08_COMPACT_NATIVE_PROPOSED_ANALYSIS_R1_20261010'


def plan_text():
    return '''# Results plan: compact and defensible

Use six main tables and two figures. Tables need different jobs; more tables do not improve a paper by themselves.

| Main exhibit | Purpose | Status |
|---|---|---|
| T1: two-corpus data/protocol | Counts, groups, splits, vocabulary and coverage | Available |
| T2: primary competitor comparison | Proposed and every locked competitor profile, five seeds | Available |
| T3: two-panel matched ablations | Components and projection controls; retain every observed tradeoff | Available |
| T4: external comparison | Independent-corpus replication of the frozen recipe | Available |
| T5: transportation text examples | Six automatically screened, source-grounded fitting examples | Available; illustrations only |
| T6: native-versus-Proposed divergence | Quantitative affinity adjustment analysis, both corpora | Newly computed from all ten saved readouts |
| F1: before/after/change affinity | Where raw affinity changes, with common before/after scales | Newly computed; primary main, external supplementary |
| F2: graph/solver sensitivity | Twenty completed development cases, connected matched seeds | Available; fixed rank margin, not a hyperparameter sweep |

Keep four short supplementary tables: additional controls, paired prediction uncertainty, metric availability, and reproducibility. Keep full word lists, configuration dumps, per-document results, matrices and numerical certificates as audit/data files rather than long manuscript tables. Exact matrices and vocabulary mappings are in numerical_evidence; figures are PDF/SVG/300-dpi PNG.

## Affinity analysis

Background NTM means the matched native NeuroMax parent. Tuned NTM means Proposed's projected readout of that same parent. Compare the same seed, topic order and vocabulary. Compare each seed separately, then average metric values, never unmatched topic matrices across seeds.

1. Normalize each nonnegative affinity row by its own sum; compute Jensen-Shannon divergence and total variation between before/after affinity shares. These shares are not decoder probabilities.
2. Pass unit-topic inputs through the saved affine decoder: softmax(A[k,:] * gain + offset). Compute divergence between these decoded topic profiles. They are decoder prototypes, not mixture-model beta rows.
3. With identical frozen observed-only document latents, decode native and Proposed predictions, include each frozen backoff, then compare all eligible test distributions. This measures output disturbance, not prediction improvement. Use the existing matched completion NLL for improvement/cost.
4. Verify target top-ten realization, nonnegativity and row/column totals independently. Same-core display only retains native affinity: descriptor gains alone do not demonstrate enforcement in the predictor.

JS(P,Q) = 0.5 KL(P || M) + 0.5 KL(Q || M), M = (P+Q)/2; natural logarithms, units nats. SciPy jensenshannon returns its square-root distance, so square it when checking JS divergence. No smoothing or clipping of saved affinity occurs. Only negative JS roundoff within 2e-14 is rounded to zero.

Official calculation references: https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.spatial.distance.jensenshannon.html and https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.special.rel_entr.html .

## What is still missing, and how to obtain it

The available F2 is sensitivity to graph and solver choices. A full numeric support-threshold/rank-margin robustness study and acquisition perturbation study are exported separately by `--sensitivity-budget`. Do not relabel F2 as those studies. Gamma is not a parameter of the executed final acquisition, so a gamma sweep would require an additional methodological revision.

If required for robustness claims, use the original saved fitting/development parents and association evidence, without neural retraining:

1. Fix the final recipe and all five seeds. Independently vary pair-group support over {3,5,10}, retaining all other settings. Report fitting C_v/NPMI, complete target feasibility, replacements, realized diversity, affinity movement and development completion NLL. Retain failed/unchanged cases.
2. Independently vary the scaled rank margin over {1e-7,1e-6,1e-5}, keeping each seed's selected word sets and fitting contexts fixed; run primary TV and range secondary. Record certification, solver timeouts, affinity movement and full-pipeline development NLL. This yields a genuine rank-margin tradeoff figure.
3. For acquisition stability, take five fixed 80% subsamples of independent fitting groups per seed, rebuild support and association evidence, and report target-set Jaccard/replacement retention. Preserve duplicate-group evidence units and count failed acquisitions.
4. Cap runs and solver time before execution, cache each case, and save results separately. These are post-exposure descriptive diagnostics; do not use them to choose a new final model or alter the exposed test comparisons. Any performance-changing revision requires new untouched evaluation evidence.

No additional files are needed for the supplied compact/divergence extension when the completed Step 6/7 artifacts are retained. The additional numeric sweeps need the original Step 3/5 fitting latents, association evidence and 64 fitting-context IDs. If those caches were removed, supply their inspection ZIPs before implementing the sweeps.

## Paper claims

Primary coherence point means are strong; external winners are mixed. Both native-parent prediction intervals cross zero. Complete five-seed held-out coherence is unavailable, so label coherence as fitting coherence. Removing a component need not lower every score. Divergence measures adjustment, not superiority; constraint realization and controlled predictive disturbance explain the projection's role. The original complete spectral/disjoint methodology was not separately benchmarked under the final protocol. Moment controls do not stand in for it. Do not claim validated passenger-requirement accuracy from unlabeled examples.

This plan supports a reviewable manuscript; no numerical confidence percentage or journal score is assigned.
'''


def run(root,log):
    from step08_assembly.runner import run as assemble
    from .divergence import calculate
    from .report import exhibit_tables
    from .figures import draw
    log.info('COMPACT EXPORT | six main tables; four short supplementary tables; affinity divergence and affinity figures')
    _,assembly=assemble(root,log)
    bundle=protect(root)
    code=list((root/'step08_compact').glob('*.py'))+list((root/'step08_compact/data').glob('*.json'))
    identity={'protocol':PROTOCOL,'assembly_manifest':file_hash(assembly/'artifact_manifest.json'),
              'implementation':{p.relative_to(root).as_posix():file_hash(p)for p in sorted(code)},
              'new_model_fits':0,'new_parameter_selection':False}
    key=digest(identity);folder=root/'outputs/steps/step08_compact'/key[:20]
    if (folder/'artifact_manifest.json').is_file():
        verify_tree(folder);unchanged(root,bundle['snapshot'])
        log.info('VERIFIED REUSE | compact exports and earlier predictions unchanged')
        return read_json(folder/'run_result.json'),folder
    folder.mkdir(parents=True,exist_ok=True)
    write_json(log.folder/'working_path.json',{'relative_path':folder.relative_to(root).as_posix()})
    write_json(folder/'report_identity.json',identity)
    write_json(folder/'input_protection.json',bundle['snapshot'])
    seed_results,details=calculate(bundle,folder,log)
    write_json(folder/'numerical_evidence/primary_fitting_targets.json',
               read_json(root/'step08_compact/data/primary_fitting_targets.json'))
    exhibit_tables(folder,assembly,seed_results)
    draw(folder,details,root/'step08_compact/data/pretest_recipe_cases.json',assembly)
    write_text(folder/'RESULTS_PLAN.md',plan_text())
    write_text(folder/'SUMMARY.md','''# Compact paper exhibits

Open MAIN_RESULTS.md (six tables), SUPPLEMENTARY_RESULTS.md (four short tables), and RESULTS_PLAN.md. Figures are in main/figures and supplementary/figures. Exact matrices and per-seed/topic/document divergence are in numerical_evidence. Numeric support, replacement-limit and rank-margin sweeps are exported separately by --sensitivity-budget; F2 remains the earlier graph/solver comparison.

The native-versus-Proposed affinity comparison is complete for all five seeds on both corpora. Before/after matrices preserve matched topic and word order. The actual-readout target sets are realized in all ten Proposed matrices. Divergence measures movement; paired NLL and matched controls assess predictive cost and mechanism. Neither journal acceptance nor universal superiority is established.

No earlier predictions, configurations or methodological choices are changed. The first fixed seed's heatmap uses 24 fitting-derived changed columns; full matrices are saved for every seed. Before/after share a color scale; signed changes have a separate symmetric scale. Chinese word-ID labels require publication relabeling using the exact lookup.
''')
    unchanged(root,bundle['snapshot']);verify_tree(assembly)
    result={'status':'STEP08_COMPACT_AFFINITY_ANALYSIS_COMPLETE','engineering_passed':True,
        'new_model_fits':0,'new_encoder_calls':0,'new_parameter_selections':0,
        'main_tables':6,'supplementary_tables':4,'native_proposed_cases':10,
        'affinity_divergence_complete':True,'full_numeric_sensitivity_sweep_complete':False,
        'all_metric_superiority_established':False,'old_predictions_unchanged':True,
        'report_id':key,'processed_relative_path':folder.relative_to(root).as_posix()}
    write_json(folder/'run_result.json',result)
    package=zip_files(folder/'compact_paper_package.zip',folder,list(folder.rglob('*')),
        {'scope':'compact publication exhibits and exact affinity analysis; no refits'})
    write_json(folder/'artifact_manifest.json',artifact_manifest(folder))
    write_json(root/'outputs/steps/step08_compact/LATEST.json',{'report_id':key,'relative_path':folder.relative_to(root).as_posix()})
    write_json(root/'state/step08_compact_result.json',result)
    log.info('COMPLETE | affinity analysis=10/10 cases | main tables=6 | supplement tables=4 | numeric sweeps use the separate --sensitivity-budget command')
    log.info('READ FIRST | %s',folder/'MAIN_RESULTS.md');log.info('FIGURES | %s',folder/'main/figures')
    log.info('RESULTS ZIP | %s',package)
    return result,folder


def cli(args,root):
    root=Path(root).resolve();log=ReportLog(root);monitor=ResourceMonitor();result=None;output=None;code=2
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):log.info('RUNNING | compact Step 8 | log=%s',log.log_path)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        log.info('START | compact Step 8 | device=%s | saved-array computation=CPU | new neural fits=0',args.device)
        if args.self_check:
            from . import tests
            from step08_assembly import tests as original_tests
            suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),
                                     unittest.defaultTestLoader.loadTestsFromModule(original_tests)])
            stream=io.StringIO();checked=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            write_text(log.folder/'self_checks.txt',stream.getvalue())
            log.info('SELF CHECK | tests=%d | failures=%d | errors=%d',checked.testsRun,len(checked.failures),len(checked.errors))
            if not checked.wasSuccessful() or checked.skipped:
                log.info('%s',stream.getvalue());raise AssertionError('Compact Step 8 checks failed.')
            result={'status':'SELF_CHECK_PASSED','engineering_passed':True,'tests':checked.testsRun,'synthetic_checks_only':True};code=0
        elif args.doctor_only:
            bundle=protect(root);unchanged(root,bundle['snapshot'])
            result={'status':'STEP08_COMPACT_DOCTOR_COMPLETE','engineering_passed':True};code=0
        else:
            with ProjectLock(root):result,output=run(root,log)
            code=0
    except (Exception,KeyboardInterrupt) as exc:
        result={'status':'FAILED','engineering_passed':False,'error_type':type(exc).__name__,'error':str(exc)}
        write_text(log.folder/'failure_traceback.txt',traceback.format_exc());log.logger.error('FAILED | %s: %s',type(exc).__name__,exc)
        if (log.folder/'working_path.json').is_file():output=root/read_json(log.folder/'working_path.json')['relative_path']
    finally:
        stop.set();thread.join(timeout=1)
        resources=monitor.finish();resources.update(new_model_fits=0,new_encoder_calls=0,GPU_compute=False)
        write_json(log.folder/'resource_usage.json',resources);write_json(log.folder/'run_result.json',result)
        log.history('compact_finish',{'status':result['status'],'exit_code':code})
        log.info('RESOURCES | wall=%.2f sec | peak CPU RSS=%.3f GiB | model/encoder calls=0',resources['wall_seconds'],resources['peak_cpu_rss_gib'])
        try:
            files=list(log.folder.rglob('*'))+[log.log_path,root/'run_history.txt',root/'main.py']
            files+=list((root/'step08_compact').rglob('*.py'))+list((root/'step08_compact/data').glob('*.json'))
            if output:files+=list(output.rglob('*'))
            inspection=zip_files(root/'outputs/inspection'/('compact_step08_'+log.run_id+'_inspection.zip'),root,files)
            log.info('INSPECTION ZIP | %s',inspection)
        except Exception as exc:
            log.logger.error('Inspection packaging failed: %s',exc);code=2
        log.info('REPORT | %s',log.folder/'run_result.json');log.close()
    return code
