"""Replay the frozen statistical functions from the released observation table."""
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import analyze_era_formal as analysis

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New output directory')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'runs/era_analysis/20261019T015255Z_d5b87f8c44c6/observations.csv'
    df = pd.read_csv(source)
    sampling = df[df['mode'].eq('sampling')].copy()
    analysis.count_tables(df, out)
    curves, metrics, _ = analysis.analyze_metrics(sampling, out)
    analysis.missingness_bounds(curves, out)
    models = analysis.fit_models(sampling, out)
    analysis.secondary_scores(df, sampling, out)
    analysis.plots(curves, metrics, models, out)
    analysis.write(out / 'replay.json', {
        'input_sha256': analysis.sha(source), 'input_records': len(df),
        'bootstrap_resamples': analysis.B, 'bootstrap_seed': analysis.BOOT_SEED,
        'raw_response_audit_performed': False, 'new_model_calls': 0,
        'purpose': 'Replay from released extracted observations, not raw-response extraction.'})
    print('Saved statistical replay:', out)

if __name__ == '__main__':
    main()
