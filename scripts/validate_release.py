"""Check release integrity and input dependencies without inference or simulations."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def main():
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    for name, record in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record['sha256'], name
    for path in (ROOT / 'src').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        importlib.import_module(path.stem)
    from run_era_formal import load_design
    from run_a20_batch import load_inputs
    design, prompts, requests = load_design()
    core = [r for r in requests if r['module'] == 'core']
    assert len(core) == 42630
    assert len([r for r in requests if r['module'] == 'heldout_items']) == 2400
    load_inputs(ROOT / 'configs/a20_trial_manifest_v2.json')
    for name in ['era_submission_runtime_v3.json', 'era_heldout_runtime_v1.json']:
        runtime = json.loads((ROOT / 'configs' / name).read_text())
        assert hashlib.sha256((ROOT/runtime['readout_policy']).read_bytes()).hexdigest() == runtime['readout_policy_sha256']
        for model in runtime['models'].values():
            assert (ROOT / model['show_file']).is_file()
    config = json.loads((ROOT / 'configs/sprint_v4_20261019.json').read_text())
    assert hashlib.sha256((ROOT / config['core_observations']).read_bytes()).hexdigest() == config['core_observations_sha256']
    count = len(list((ROOT/'runs/sprint_v4/20261019_authorized/calls').glob('*/result.json')))
    assert count == 909
    print(json.dumps({'status': 'PASS', 'release_files': len(manifest['files']),
                      'core_requests': len(core), 'macro_records': count,
                      'model_calls': 0, 'simulations': 0}, indent=2))

if __name__ == '__main__':
    main()
