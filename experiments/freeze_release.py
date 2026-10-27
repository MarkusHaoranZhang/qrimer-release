"""Freeze the code/table artifacts with SHA-256 hashes."""

import hashlib
import json
import os

FILES = [
    'core/block_encoding.py',
    'core/qbra_arithmetic.py',
    'core/qbra_combined.py',
    'core/qer.py',
    'core/structured_qer.py',
    'experiments/run_qbra_benchmark_t4.py',
    'experiments/run_backfill_report.py',
    'experiments/run_table4_backfill.py',
    'experiments/run_section67_backfill.py',
    'experiments/run_final_consistency.py',
    'outputs/tables/table3_structured.json',
    'outputs/tables/table10_generated.json',
    'outputs/manuscript/table2_generated.json',
    'outputs/manuscript/table4_generated.json',
    'outputs/manuscript/table5_generated.json',
    'outputs/manuscript/section67_generated.json',
    'outputs/manuscript/consistency_report.md',
    'outputs/logs/environment_manifest.json',
]


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    entries = {}
    for rel in FILES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            entries[rel] = {'exists': False}
            continue
        with open(path, 'rb') as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        entries[rel] = {
            'exists': True,
            'sha256': digest,
            'bytes': os.path.getsize(path),
        }
    out = os.path.join(root, 'outputs', 'manuscript', 'freeze_manifest.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(entries, fh, indent=2)
    print(f"Frozen {sum(e['exists'] for e in entries.values())} artifacts")
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
