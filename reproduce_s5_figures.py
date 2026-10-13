"""Reproduce S5 figures from authenticated archived primary sensitivity CSVs.

Copy this file and data/supplementary_s5/ into the existing project root.
PowerShell: python .\reproduce_s5_figures.py --verify-only
            python .\reproduce_s5_figures.py
No training, encoder calls, test scoring or model selection is performed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

FILES = (
    'sensitivity_mean_sd.csv', 'sensitivity_individual_runs.csv',
    'replacement_budget_mean_sd.csv', 'replacement_budget_individual.csv',
)
EXPECTED_HASHES = {
    'sensitivity_mean_sd.csv': '354eb2234e185c143a10181b9c68a430d3051bdac039d6cf41c3808ac0d09233',
    'sensitivity_individual_runs.csv': 'aca15e182a1992e49ff1483fe53fdfe0ca8110a411148ca695fa3c06ff36b95d',
    'replacement_budget_mean_sd.csv': '30beb9056bb7be2b95e26bc56c57e9637e48b6d8ed54da272cacd5ef562ff875',
    'replacement_budget_individual.csv': '1a669f0465178424d4eb9707ab98d06471c8de504c0f4e67e8174df32657a813',
}

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def verify(data_dir):
    import numpy as np
    records = {}
    for name in FILES:
        path = data_dir / name
        if not path.is_file():
            raise ValueError('Missing data file: ' + str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED_HASHES[name]:
            raise ValueError('Data differs from the archived S5 evidence: ' + name)
        records[name] = read_csv(path)
    replay_checks = 0
    settings = 0
    for mean_name, individual_name, keys in [
        ('sensitivity_mean_sd.csv', 'sensitivity_individual_runs.csv',
         ['study', 'support', 'margin', 'objective']),
        ('replacement_budget_mean_sd.csv', 'replacement_budget_individual.csv',
         ['support', 'margin', 'objective', 'maximum_replacements']),
    ]:
        individuals = records[individual_name]
        for row in records[mean_name]:
            matched = [x for x in individuals if all(x[k] == row[k] for k in keys)]
            if len(matched) != 5 or {int(x['seed']) for x in matched} != set(range(809, 814)):
                raise ValueError('Incomplete or duplicated seed grid: ' + str(row))
            if int(row['certified_seeds']) != 5:
                raise ValueError('An uncertified setting cannot reproduce this archived figure.')
            for x in matched:
                if x['certified'].lower() != 'true':
                    raise ValueError('Uncertified individual result.')
                if x['post_exposure_diagnostic'].lower() != 'true' or x['final_recipe_changed'].lower() != 'false':
                    raise ValueError('Unexpected study scope.')
                delta = float(x['development_nll']) - float(x['native_development_nll'])
                if abs(delta - float(x['delta_native_nll'])) > 1e-10:
                    raise ValueError('Development NLL difference does not replay.')
            for metric in ['cv', 'npmi', 'td', 'delta_native_nll', 'relative_l1_percent', 'fitting_logit_range']:
                values = np.array([float(x[metric]) for x in matched], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite plot evidence: ' + metric)
                if abs(values.mean() - float(row[metric + '_mean'])) > 1e-10:
                    raise ValueError('Mean does not replay: ' + metric)
                if abs(values.std(ddof=1) - float(row[metric + '_sd'])) > 1e-10:
                    raise ValueError('Sample SD does not replay: ' + metric)
                replay_checks += 2
            settings += 1
    return records, {'corpus': 'primary', 'settings': settings,
                     'report_slots': 80, 'mean_sd_checks': replay_checks,
                     'seeds': list(range(809, 814)), 'q': [3, 5, 10], 'R': [2, 4, 6],
                     'frozen_q': 5, 'frozen_R': 4, 'frozen_scaled_margin': 1e-6,
                     'scope': 'post-exposure descriptive fitting/development study',
                     'neural_fits': 0, 'encoder_calls': 0, 'test_scoring': False,
                     'values_changed': False, 'source_sha256': EXPECTED_HASHES}

def plot(records, output_dir, formats):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    def values(rows, key):
        return np.array([float(x[key]) for x in rows])
    def save(fig, name):
        for fmt in formats:
            path = output_dir / (name + '.' + fmt)
            fig.savefig(path, dpi=220 if fmt == 'png' else None, bbox_inches='tight')
            saved.append(str(path.resolve()))
        plt.close(fig)
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    support = sorted([r for r in records['sensitivity_mean_sd.csv'] if r['study'] == 'support'],
                     key=lambda r: float(r['support']))
    budget = sorted(records['replacement_budget_mean_sd.csv'],
                    key=lambda r: float(r['maximum_replacements']))
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.8), layout='constrained')
    for i, (rows, key, chosen, xlabel) in enumerate([
        (support, 'support', 5, r'Pair-support threshold $q$'),
        (budget, 'maximum_replacements', 4, r'Replacement limit $R$'),
    ]):
        x = values(rows, key)
        for j, (metric, label, mult) in enumerate([
            ('cv', r'Fitting-reference $C_v$', 1),
            ('npmi', 'Fitting-reference NPMI', 1),
            ('delta_native_nll', r'Development $\Delta$NLL ($\times 10^{-3}$)', 1000),
        ]):
            ax = axes[i, j]
            y = values(rows, metric + '_mean') * mult
            sd = values(rows, metric + '_sd') * mult
            ax.errorbar(x, y, yerr=sd, color='#2a5c8a', marker='o', capsize=3, lw=1.2)
            k = list(x).index(chosen)
            ax.scatter([x[k]], [y[k]], marker='*', s=100, color='#9b3c26', zorder=5)
            ax.axvline(chosen, color='#9b3c26', ls='--', lw=.8, alpha=.6)
            if metric == 'delta_native_nll':
                ax.axhline(0, color='.5', lw=.7)
            ax.set_xticks(x)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label)
            ax.set_title('(' + chr(97 + i * 3 + j) + ')')
            ax.grid(axis='y', alpha=.18)
    save(fig, 'SF1_parameter_sensitivity')
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2), layout='constrained')
    for method, color, marker in [('primary', '#5e6872', 's'), ('range', '#2a5c8a', 'o')]:
        rows = sorted([r for r in records['sensitivity_mean_sd.csv']
                       if r['study'] == 'margin' and r['objective'] == method],
                      key=lambda r: float(r['margin']))
        x = values(rows, 'margin')
        for j, (metric, label, mult) in enumerate([
            ('relative_l1_percent', 'Affinity movement (%)', 1),
            ('delta_native_nll', r'Development $\Delta$NLL ($\times 10^{-3}$)', 1000),
            ('fitting_logit_range', 'Fitting logit-range objective', 1),
        ]):
            axes[j].errorbar(x, values(rows, metric + '_mean') * mult,
                            yerr=values(rows, metric + '_sd') * mult,
                            color=color, marker=marker, capsize=2, lw=1,
                            label='Minimum TV' if method == 'primary' else 'Prediction-aware secondary')
            axes[j].set_xscale('log')
            axes[j].axvline(1e-6, color='#9b3c26', ls='--', lw=.8)
            axes[j].set_xlabel('Scaled rank margin')
            axes[j].set_ylabel(label)
            axes[j].grid(axis='y', alpha=.18)
            axes[j].set_title('(' + chr(97 + j) + ')')
    axes[1].axhline(0, color='.5', lw=.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False, fontsize=8)
    save(fig, 'SF2_rank_margin_tradeoff')
    return saved

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--formats', nargs='+', choices=['pdf', 'png'], default=['pdf', 'png'])
    args = parser.parse_args()
    root = args.project_root.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else root / 'data' / 'supplementary_s5'
    output_dir = args.output_dir.resolve() if args.output_dir else root / 'outputs' / 'paper' / 'supplementary' / 'figures'
    try:
        records, report = verify(data_dir)
        print('S5 DATA VERIFIED | primary | five seeds | 16 settings | 80 report slots')
        print('MEAN/SD REPLAY | checks=' + str(report['mean_sd_checks']) + ' | failures=0')
        print('SCOPE | fitting/development descriptive study | new model fits=0 | test scoring=0')
        if args.verify_only:
            print('VERIFY-ONLY COMPLETE | figure files written=0')
            return 0
        files = plot(records, output_dir, args.formats)
        report['output_files'] = files
        (output_dir / 'S5_reproduction_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        for path in files:
            print('SAVED | ' + path)
        print('S5 FIGURES COMPLETE | ' + str(output_dir))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print('S5 FAILED | ' + str(exc), file=sys.stderr)
        print('Keep the four supplied CSVs unchanged. This script requires numpy and matplotlib.', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
