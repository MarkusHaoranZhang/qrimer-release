"""Export generated tables as LaTeX snippets for manuscript backfill."""

import json
import os


def _load(root, name):
    with open(os.path.join(root, 'outputs', 'tables', name), encoding='utf-8') as fh:
        return json.load(fh)


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    out_dir = os.path.join(root, 'outputs', 'manuscript')
    os.makedirs(out_dir, exist_ok=True)

    t3 = _load(root, 'table3_structured.json')
    t10 = _load(root, 'table10_generated.json')

    lines = [
        r'\begin{table}[h]',
        r'\small',
        r'\centering',
        r'\caption{Structured QER resource measurements (generated)}',
        r'\label{tab:runtime}',
        r'\begin{tabular}{ccccc}',
        r'\hline',
        r'$N$&$L$&Qubits&One-step depth&Estimated full depth\\',
        r'\hline',
    ]
    for r in t3:
        depth = r.get('step_depth')
        full = r.get('estimated_full_depth')
        depth = '--' if depth is None else f"{depth}"
        full = '--' if full is None else f"{full}"
        lines.append(f"{r['N']}&{r['L']}&{r['qubits']}&{depth}&{full}\\\\")
    lines += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    with open(os.path.join(out_dir, 'table3_generated.tex'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    lines = [
        r'\begin{table}[h]',
        r'\small',
        r'\centering',
        r'\caption{Block-encoded operator noise (generated)}',
        r'\label{tab:noise_large}',
        r'\begin{tabular}{cccc}',
        r'\hline',
        r'$N$&Noise $p$&$\varepsilon_\beta$&Success product\\',
        r'\hline',
    ]
    for r in t10:
        lines.append(f"{r['N']}&{r['noise']}&{r['error']:.4f}&"
                     f"{r['success_product']:.3e}\\\\")
    lines += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    with open(os.path.join(out_dir, 'table10_generated.tex'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    note = [
        '# Manuscript backfill bundle',
        '',
        'Generated from:',
        '- `outputs/tables/table3_structured.json`',
        '- `outputs/tables/table10_generated.json`',
        '',
        'These tables replace the legacy reference constants.  Table 3 full',
        'depth is a one-step transpiled depth multiplied by L-1; it is not a',
        'compiled L-step circuit.  Table 10 is block-encoded density-matrix',
        'operator noise, not Falcon gate-level noise.',
        '',
        'Table 2 and Table 5 are generated separately by',
        '`run_backfill_report.py`.  Still requiring backfill: Table 4 and',
        'Section 6.7 runtime/memory values.',
    ]
    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(note))
    print(f"Saved LaTeX bundle to {out_dir}")


if __name__ == '__main__':
    main()
