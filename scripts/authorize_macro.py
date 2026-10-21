"""Record explicit local opt-in to the frozen, bounded model collection."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approve', action='store_true', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = ROOT / 'configs/sprint_v4_20261019.json'
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as handle:
        json.dump({'approved': True, 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                   'max_new_calls': 1800, 'hard_wall_seconds': 2700}, handle, indent=2)
    print('Saved local opt-in:', target)

if __name__ == '__main__':
    main()
