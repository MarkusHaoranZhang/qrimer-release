"""Final cross-table consistency report for manuscript backfill."""

import json
import os

from analysis.paper_reference import TABLE3_DEPTH, TABLE3_QUBITS, TABLE10_NOISE

PAPER_TABLE2 = {
    5: (0.0031, 0.0147), 10: (0.0078, 0.0623), 15: (0.0124, 0.1338),
    20: (0.0189, 0.2415), 30: (0.0196, 0.3872),
}
PAPER_SECTION67 = {
    15: (142.0, 18.6),
    20: (3600.0, 420.0),
}


def _load(root, name, sub='manuscript'):
    with open(os.path.join(root, 'outputs', sub, name),
              encoding='utf-8') as fh:
        return json.load(fh)


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    t2 = _load(root, 'table2_generated.json')
    t3 = _load(root, 'table3_structured.json', 'tables')
    t4 = _load(root, 'table4_generated.json')
    t5 = _load(root, 'table5_generated.json')
    t10 = _load(root, 'table10_generated.json', 'tables')
    s67 = _load(root, 'section67_generated.json')

    lines = ['# Final consistency report', '']
    lines.append('## Table 2 vs manuscript')
    for row in t2:
        old = PAPER_TABLE2[row['N']]
        lines.append(
            f"- N={row['N']}: Q-RIMER {row['qrimer_mean']:.4f} "
            f"(old {old[0]:.4f}); BF-QC {row['bfqc_mean']:.4f} "
            f"(old {old[1]:.4f})"
        )
    lines.append('')
    lines.append('## Table 3 vs manuscript')
    for row in t3:
        key = (row['N'], row['L'])
        old_d = TABLE3_DEPTH.get(key)
        old_q = TABLE3_QUBITS.get(key)
        lines.append(
            f"- N={row['N']}, L={row['L']}: qubits {row['qubits']} "
            f"(old {old_q}); step depth {row.get('step_depth')} "
            f"(old {old_d})"
        )
    lines.append('')
    lines.append('## Table 4 vs manuscript')
    for row in t4:
        old = TABLE10_NOISE['N3'].get(row['factor'])
        lines.append(
            f"- N=3, scale={row['factor']}: {row['epsilon_mean']:.4f} "
            f"(old {old})"
        )
    lines.append('')
    lines.append('## Table 5 vs manuscript')
    for row in t5:
        old = TABLE10_NOISE[f"N{row['N']}"].get(row['factor'])
        lines.append(
            f"- N={row['N']}, scale={row['factor']}: "
            f"{row['epsilon_mean']:.4f} (old {old})"
        )
    lines.append('')
    lines.append('## Section 6.7 vs manuscript')
    for row in s67:
        old = PAPER_SECTION67.get(row['N'])
        lines.append(
            f"- N={row['N']}: direct {row['direct_mean']:.4f}s, "
            f"vectorized {row['vectorized_mean']:.4f}s, "
            f"speedup {row['speedup']:.2f}x "
            f"(old direct/vectorized {old})"
        )
    lines.append('')
    lines.append('## Table 10 generated')
    for row in t10:
        lines.append(
            f"- N={row['N']}, p={row['noise']}: "
            f"error {row['error']:.4f}, success {row['success_product']:.3e}"
        )
    lines.append('')
    lines.append('## Freeze status')
    lines.append('- Code artifacts: frozen after this report.')
    lines.append('- Legacy paper references remain in analysis/paper_reference.py.')
    lines.append('- Manuscript must use generated values and the updated noise '
                 'and timing caveats.')
    out = os.path.join(root, 'outputs', 'manuscript', 'consistency_report.md')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f"Saved: {out}")
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    main()
