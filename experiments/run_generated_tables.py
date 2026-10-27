"""Print the reproducible tables generated from this repository."""

import json
import os


def _load(name):
    path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'tables', name)
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def main():
    print("=" * 72)
    print("Generated Table 3 (structured one-step QER transpilation)")
    print("=" * 72)
    print(f"  {'N':>3}  {'L':>4}  {'qubits':>7}  {'step depth':>11}  "
          f"{'full depth~':>11}  status")
    for row in _load('table3_structured.json'):
        depth = row.get('step_depth')
        depth = '-' if depth is None else str(depth)
        full = row.get('estimated_full_depth')
        full = '-' if full is None else str(full)
        print(f"  {row['N']:>3}  {row['L']:>4}  {row['qubits']:>7}  "
              f"{depth:>11}  {full:>11}  {row['status']}")

    print("\n" + "=" * 72)
    print("Generated Table 10 (full-L density-matrix operator noise)")
    print("=" * 72)
    print(f"  {'N':>3}  {'noise':>7}  {'error':>11}  {'success product':>16}")
    for row in _load('table10_generated.json'):
        print(f"  {row['N']:>3}  {row['noise']:>7}  {row['error']:>11.4f}  "
              f"{row['success_product']:>16.3e}")
    print("\nSources: table3_structured.json, table10_generated.json")


if __name__ == '__main__':
    main()
