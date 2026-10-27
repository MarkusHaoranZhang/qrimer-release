"""Figure 3 from generated structured-QER resource data."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, '..')

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
})


def main():
    with open(os.path.join(ROOT, 'outputs', 'tables',
                           'table3_structured.json'), encoding='utf-8') as fh:
        rows = json.load(fh)
    rows = [r for r in rows if r.get('step_depth') is not None]
    N = np.array([r['N'] for r in rows], dtype=float)
    D = np.array([r['step_depth'] for r in rows], dtype=float)
    F = np.array([r['estimated_full_depth'] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ax.plot(N, D, 'o-', color='#0F62FE', markersize=4.5,
            linewidth=1.4, label='One-step structured QER')
    ax.plot(N, F, 's--', color='#FF832B', markersize=4.0,
            linewidth=1.2, label=r'Estimated full depth ($L-1$ steps)')
    ax.set_yscale('log')
    ax.set_xticks([3, 5, 10, 15, 20])
    ax.set_ylim(2.5e3, 4.0e8)
    ax.set_xlabel('Number of evaluation grades $N$')
    ax.set_ylabel('Transpiled depth (log scale)')
    ax.grid(True, which='major', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    offsets = {3: (28, 0), 5: (-8, 10), 10: (-8, 10), 15: None, 20: (-8, 8)}
    shown = set()
    for n, d in zip(N, D):
        if int(d) in shown or offsets.get(int(n)) is None:
            continue
        shown.add(int(d))
        ax.annotate(f'{int(d):,}', (n, d), textcoords='offset points',
                    xytext=offsets.get(int(n), (6, 7)), fontsize=6.5,
                    ha='right' if int(n) in (5, 10, 20) else 'left')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout(pad=0.3)
    out = os.path.join(ROOT, 'outputs', 'figures', 'depth_generated')
    fig.savefig(out + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'Saved: {out}.pdf, {out}.png')


if __name__ == '__main__':
    main()
