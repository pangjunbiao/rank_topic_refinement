"""Same-LP numerical retry: no clipping, changed targets, or wider TV budget."""
from __future__ import annotations
import time
import numpy as np
from rank_refinement.projection.replay_solver import primary_tv,range_secondary
from rank_refinement.projection.minimum_tv import minimum_tv,certify
from rank_refinement.projection.predictive_controls import secondary_lp

def roundoff_nonnegative_failure(report):
    value=report.get('minimum_affinity_scaled')
    return value is not None and -1e-8<=value<0 and report.get('nonnegative')is False

def safe_primary(native,targets,margin,seconds):
    started=time.perf_counter();phi,report=primary_tv(native,targets,margin,seconds)
    if phi is None and roundoff_nonnegative_failure(report):
        remaining=seconds-(time.perf_counter()-started)
        if remaining>0:
            before=report
            # The original affinity-variable LP enforces the identical feasible
            # set/objective, without subtracting almost equal displacement terms.
            phi,report=minimum_tv(native,targets,margin,remaining,1e-7,1e-5)
            report.update(numerical_retry='same_LP_affinity_variables',displacement_attempt=before,
                          same_feasible_set=True,same_primary_objective=True)
    report['seconds_total']=time.perf_counter()-started
    return phi,report

def safe_range(native,targets,primary_value,contexts,gain,primary,margin,slack,seconds,iterations,tolerance,log):
    started=time.perf_counter()
    phi,report=range_secondary(native,targets,primary_value,contexts,gain,primary,
        margin,slack,seconds,iterations,tolerance,log)
    if phi is None and roundoff_nonnegative_failure(report):
        remaining=seconds-(time.perf_counter()-started)
        if remaining>0:
            before=report
            phi,report=secondary_lp(native,targets,primary_value,margin,slack,remaining,
                                   mode='range',contexts=contexts,decoder_gain=gain)
            report.update(numerical_retry='same_LP_affinity_variables_full_range_constraints',
                          displacement_attempt=before,same_feasible_set=True,same_secondary_objective=True)
            if phi is not None:
                measured=float(np.ptp((contexts@(phi-native))*gain,axis=1).mean())
                gap=abs(measured-report['secondary_objective'])
                report.update(full_range_objective=measured,independent_full_objective_gap=gap)
                if gap>max(1e-7,2*tolerance):
                    phi=None;report.update(success=False,certified=False,failure='secondary_full_range_objective_not_certified')
    report['seconds_total']=time.perf_counter()-started
    return phi,report
