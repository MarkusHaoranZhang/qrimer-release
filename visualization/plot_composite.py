"""Composite experimental figure from saved logs (all-new panels)."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, '..')
OUT = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 7.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
})

BLUE = '#0F62FE'
ORANGE = '#FF832B'
LIGHT_BLUE = '#78A9FF'


def _style(ax):
    ax.grid(True, color='0.92', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    large = np.load(os.path.join(ROOT, 'outputs', 'logs',
                                 'large_n_quantum_results.npz'),
                    allow_pickle=True)
    with open(os.path.join(ROOT, 'outputs', 'manuscript',
                           'table5_generated.json'), encoding='utf-8') as fh:
        noise = json.load(fh)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3))

    ax = axes[0, 0]
    for key, color, label in [('N10_L100', BLUE, r'$N=10$, $L=100$'),
                              ('N20_L200', ORANGE, r'$N=20$, $L=200$')]:
        succ = large[key].item()['per_step_success']
        ax.plot(np.arange(1, len(succ) + 1), succ, color=color,
                linewidth=0.9, label=label)
    ax.set_xlabel('Combination step')
    ax.set_ylabel(r'Post-selection success $p_k^{\mathrm{amp}}$')
    ax.set_ylim(0.9675, 0.9695)
    ax.legend(frameon=False, loc='lower right')
    _style(ax)
    ax.set_title('(a) Per-step post-selection success', loc='center', fontsize=7.5)

    ax = axes[0, 1]
    for N, color in [(3, LIGHT_BLUE), (10, BLUE), (20, ORANGE)]:
        rows = [r for r in noise if r['N'] == N]
        rows.sort(key=lambda r: r['factor'])
        ax.plot([r['factor'] for r in rows],
                [r['epsilon_mean'] for r in rows],
                'o-', color=color, markersize=3, linewidth=1.0,
                label=rf'$N={N}$')
    ax.set_xlabel('Noise scale factor')
    ax.set_ylabel(r'$\varepsilon_\beta$')
    ax.set_xlim(-0.4, 10.4)
    ax.legend(frameon=False, loc='upper left')
    _style(ax)
    ax.set_title('(b) Noise response', loc='center', fontsize=7.5)

    ax = axes[1, 0]
    rec = large['N20_L200'].item()
    idx = np.arange(1, len(rec['beta_c']) + 1)
    ax.bar(idx - 0.18, rec['beta_c'], 0.34, color=BLUE,
           label='Classical RIMER', edgecolor='white', linewidth=0.3)
    ax.bar(idx + 0.18, rec['beta_q'], 0.34, color=ORANGE,
           label='Q-RIMER', edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Evaluation grade')
    ax.set_ylabel(r'Belief degree $\beta_j$')
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 0.075)
    _style(ax)
    ax.legend(frameon=False, loc='upper right', ncol=2)
    ax.set_title(r'(c) $N=20$ belief distribution', loc='center', fontsize=7.0)

    ax = axes[1, 1]
    for key, color, label in [('N10_L100', BLUE, r'$N=10$'),
                              ('N20_L200', ORANGE, r'$N=20$')]:
        succ = large[key].item()['per_step_success']
        ax.hist(succ, bins=18, histtype='step', color=color,
                linewidth=1.1, weights=np.ones_like(succ) / len(succ),
                label=label)
    ax.set_xlabel(r'Per-step success $p_k^{\mathrm{amp}}$')
    ax.set_ylabel('Relative frequency')
    ax.set_xlim(0.9679, 0.9697)
    ax.set_xticks([0.9680, 0.9685, 0.9690, 0.9695])
    ax.legend(frameon=False, loc='upper right')
    _style(ax)
    ax.set_title('(d) Per-step success distribution', loc='center', fontsize=7.5)

    fig.tight_layout(pad=0.5)
    base = os.path.join(OUT, 'composite_experiments')
    fig.savefig(base + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(base + '.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'Saved: {base}.pdf, {base}.png')


if __name__ == '__main__':
    main()
