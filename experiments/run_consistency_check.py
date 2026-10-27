"""Cross-table consistency report for the generated paper artifacts."""

import json
import os

from analysis.paper_reference import TABLE3_DEPTH, TABLE10_NOISE


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    table3_path = os.path.join(root, 'outputs', 'tables',
                               'table3_structured.json')
    table10_path = os.path.join(root, 'outputs', 'tables',
                                'table10_generated.json')
    with open(table3_path, encoding='utf-8') as fh:
        table3 = {f"{r['N']}_{r['L']}": r for r in json.load(fh)}
    with open(table10_path, encoding='utf-8') as fh:
        table10 = json.load(fh)

    print("Table 3 consistency")
    for (N, L), legacy_depth in sorted(TABLE3_DEPTH.items()):
        row = table3.get(f'{N}_{L}')
        if row is None:
            print(f"  N={N}, L={L}: missing")
            continue
        depth = row.get('step_depth')
        print(f"  N={N:>2}, L={L:>3}: legacy={legacy_depth:>4} "
              f"generated_step={depth} "
              f"status={row.get('status')}")

    print("\nTable 10 consistency")
    legacy = {f'{N}_{f}': v for N, vals in TABLE10_NOISE.items()
              for f, v in vals.items()}
    for row in table10:
        key = f"N{row['N']}_{row['noise']}"
        ref = legacy.get(key)
        print(f"  N={row['N']:>2}, noise={row['noise']:<5} "
              f"generated={row['error']:.4f} legacy={ref}")


if __name__ == '__main__':
    main()
