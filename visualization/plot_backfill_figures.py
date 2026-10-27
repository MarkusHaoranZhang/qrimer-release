"""Regenerate correctness and ablation figures from saved runs."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, '..')
OUT = os.path.join(ROOT, 'outputs', 'figures')
DATA = os.path.join(ROOT, 'outputs', 'manuscript')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), dpi=600,
                bbox_inches='tight', pad_inches=0.03)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=600,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def correctness():
    with open(os.path.join(DATA, 'figure2_data.json'), encoding='utf-8') as fh:
        d = json.load(fh)
    labels = ['High', 'Medium', 'Low', 'Ignorance']
    classical = np.asarray(d['classical'])
    quantum = np.asarray(d['quantum'])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    ax.bar(x - 0.18, classical, 0.34, color='#0F62FE',
           label='Classical RIMER', edgecolor='white', linewidth=0.4)
    ax.bar(x + 0.18, quantum, 0.34, color='#FF832B',
           label='Q-RIMER', edgecolor='white', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel('Belief degree')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.25)
    _save(fig, 'correctness_generated')


def ablation():
    with open(os.path.join(DATA, 'figure4_data.json'), encoding='utf-8') as fh:
        d = json.load(fh)
    labels = ['Full\nQ-RIMER', 'Without\nQBRA', 'Without\nER matrix',
              'Full\npipeline', 'Hybrid\nscheme']
    vals = np.asarray(d['values'])
    colors = ['#0F62FE', '#A6C8FF', '#DA1E28', '#0F62FE', '#DA1E28']
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    bars = ax.bar(np.arange(len(vals)), vals, color=colors,
                  width=0.62, edgecolor='white', linewidth=0.4)
    ax.set_yscale('log')
    ax.set_ylim(3e-3, 1.8)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'Belief distribution error $\varepsilon_\beta$ (log scale)')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.18,
                f'{v:.4f}', ha='center', va='bottom', fontsize=6.5)
    ax.grid(axis='y', which='major', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.3)
    _save(fig, 'ablation_generated')


if __name__ == '__main__':
    correctness()
    ablation()
    print('Saved correctness_generated and ablation_generated')
