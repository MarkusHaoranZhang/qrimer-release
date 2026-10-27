"""Figure 5 from the executable conflict scan."""

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
    'axes.linewidth': 0.8,
})


def main():
    rows = np.load(os.path.join(ROOT, 'outputs', 'logs', 'conflict_scan.npz'),
                   allow_pickle=True)['rows']
    conflict = rows[:, 0]
    p_success = rows[:, 3]
    kappa_eff = 1.0 / np.sqrt(np.maximum(p_success, 1e-15))

    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    ax.plot(conflict, kappa_eff, 'o-', color='#0F62FE',
            markersize=3.8, linewidth=1.3)
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0, 33)
    ax.set_xlabel('Conflict coefficient $K_{\\mathrm{cnf}}$')
    ax.set_ylabel('Amplification $1/\\sqrt{p}$')
    ax.grid(True, color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.3)
    out = os.path.join(ROOT, 'outputs', 'figures', 'kappa_generated')
    fig.savefig(out + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(out + '.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'Saved: {out}.pdf, {out}.png')
    print('kappa range:', kappa_eff.min(), kappa_eff.max())


if __name__ == '__main__':
    main()
