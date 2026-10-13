"""Named R2 contract: preserve native diversity, allow shared retained words.

Promoted cores remain exclusive; candidate support, signed complete-set gains,
50-word neighborhoods and four-replacement/native-retention budgets are frozen.
This is a methodological revision of the absolute 100-word union requirement.
"""
from __future__ import annotations
import copy
import numpy as np
from .complete_targets import choose_complete_targets
from .native_targets import native_indices


def recovery_targets(native,vocabulary,metadata,evidence,policy,graph,strict=False):
    if graph not in {'signed_stable','signed_full','positive_only_stable','positive_only_full'}:
        raise ValueError('Unknown frozen recovery graph.')
    changed=copy.deepcopy(evidence)
    if graph in {'signed_full','positive_only_full'}:changed['stable']=changed['full'].copy()
    if graph.startswith('positive_only'):
        changed['stable']=np.maximum(changed['stable'],0.);changed['full']=np.maximum(changed['full'],0.)
    before=len({w for t in native_indices(native) for w in t})
    revised=copy.deepcopy(policy);revised['required_unique_words']=100 if strict else before
    targets,report=choose_complete_targets(native,vocabulary,metadata,changed,revised)
    report.update(diversity_contract='absolute_100_word_control' if strict else 'preserve_native_top10_union',
        graph=graph,native_diversity_floor_words=before,promoted_cores_disjoint=True,
        methodology_revision='STEP05_NATIVE_DIVERSITY_RECOVERY_R2_20261010')
    return targets,report
