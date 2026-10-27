"""
Dataset generation and loading for Quantum RIMER experiments.
Corresponds to Section 6.1 (Preparation) of the paper.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark: graphite identification (Yang et al. 2006)
# ---------------------------------------------------------------------------

# Published classical RIMER outputs from Yang et al. 2006, Table III.
# Used as ground truth for correctness validation (RQ1).
# Test 3 inputs: X1=0.8, X3=0.98, X5=0.2, X7=0.8 (stated in paper).
CLASSICAL_GROUND_TRUTH = [
    (np.array([1.000000, 0.000000, 0.000000]), 0.0),  # Test 1
    (np.array([0.875093, 0.124907, 0.000000]), 0.0),  # Test 2
    (np.array([0.268564, 0.677169, 0.054266]), 0.0),  # Test 3  <- primary validation
    (np.array([0.014338, 0.530059, 0.455603]), 0.0),  # Test 4
    (np.array([0.031612, 0.632823, 0.335565]), 0.0),  # Test 5
    (np.array([0.017795, 0.097347, 0.884858]), 0.0),  # Test 6
    (np.array([0.000000, 0.000000, 1.000000]), 0.0),  # Test 7
    (np.array([0.313389, 0.345264, 0.341347]), 0.0),  # Test 8
    (np.array([0.010207, 0.211271, 0.778521]), 0.0),  # Test 9
    (np.array([0.007952, 0.269545, 0.722503]), 0.0),  # Test 10
    (np.array([0.043067, 0.383513, 0.573419]), 0.0),  # Test 11
    (np.array([0.186961, 0.380182, 0.432857]), 0.0),  # Test 12
]

# Confidence scores reported for the primary validation case (test run 3).
# The paper reports confidence score 0.6071 for test run 3.
CONFIDENCE_TEST3 = 0.6071


class BenchmarkLoader:
    """
    Loader for the graphite identification benchmark.
    Provides the 12 test inputs for [X1, X3, X5, X7] in [0,1], together
    with the ground-truth outputs used for validation, taken from
    CLASSICAL_GROUND_TRUTH (published Table III values).
    """

    @staticmethod
    def load() -> tuple[np.ndarray, np.ndarray, list]:
        """
        Load the 12 benchmark test inputs.

        Returns
        -------
        test_inputs : np.ndarray of shape (12, 4)
            Normalized input values for [X1, X3, X5, X7].
        labels : np.ndarray of shape (12,)
            Ground-truth class labels (0=Good, 1=Average, 2=Poor).
        descriptions : list of str
        """
        # 12 test cases: [X1, X3, X5, X7] from Yang et al. 2006, Table III.
        # Test 3 is the paper's primary validation case:
        #   X1=0.8, X3=0.98, X5=0.2, X7=0.8  -> beta_H=0.2685, beta_M=0.6772, beta_L=0.0543
        # Columns in the published Table III are (X3, X1, X5, X7);
        # they are reordered here to [X1, X3, X5, X7].
        test_inputs = np.array([
            [1.00, 0.98, 1.00, 1.00],  # Test 1
            [0.80, 0.98, 0.80, 0.80],  # Test 2
            [0.80, 0.98, 0.20, 0.80],  # Test 3  (primary validation case)
            [0.40, 0.98, 0.40, 0.80],  # Test 4
            [0.40, 0.98, 0.60, 0.80],  # Test 5
            [1.00, 0.98, 0.00, 0.00],  # Test 6
            [0.00, 0.98, 0.00, 0.00],  # Test 7
            [1.00, 0.98, 1.00, 0.20],  # Test 8
            [0.40, 0.98, 0.60, 0.20],  # Test 9
            [0.60, 0.98, 0.40, 0.20],  # Test 10
            [0.80, 0.98, 0.20, 0.20],  # Test 11
            [0.80, 0.98, 0.80, 0.20],  # Test 12
        ])

        labels = np.array([0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 0])

        descriptions = [
            f"Test {i+1}" for i in range(12)
        ]

        return test_inputs, labels, descriptions

    @staticmethod
    def ground_truth(test_id: int) -> tuple[np.ndarray, float]:
        """Return (beta, beta_D) ground truth for a 1-indexed test id."""
        beta_c, beta_D_c = CLASSICAL_GROUND_TRUTH[test_id - 1]
        return beta_c.copy(), beta_D_c


# ---------------------------------------------------------------------------
# Synthetic expanded cases (Section 6.1, Table 1)
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generator for synthetic BRB instances used in scaling experiments.

    Rule generation follows the procedure of Section 6.1: consequent belief
    distributions are drawn from Dirichlet distributions with concentration
    gamma = 0.5 (sparse but non-degenerate), attribute weights and rule
    weights are drawn from U(0.5, 1).

    Parameters
    ----------
    N : int
        Number of evaluation grades.
    L : int
        Number of belief rules.
    T : int
        Number of antecedent attributes.
    seed : int
        Random seed.
    gamma : float
        Dirichlet concentration parameter (default 0.5 per Section 6.1).
    incomplete_scale : float or None
        If not None, belief degrees are scaled by this factor to introduce
        partial ignorance (sum beta < 1).  Used by the ER-matrix ablation
        study where incomplete rules are required.
    """

    def __init__(self, N: int, L: int, T: int, seed: int = 42, gamma: float = 0.5,
                 incomplete_scale: float | None = None):
        self.N = N
        self.L = L
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.incomplete_scale = incomplete_scale

    def generate_rules(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate random belief rules using Dirichlet distributions.

        Returns
        -------
        betas : np.ndarray of shape (L, N)
            Belief degrees (rows sum to <= 1).
        rule_weights : np.ndarray of shape (L,)
            Rule weights in [0.5, 1] (Section 6.1).
        attr_weights : np.ndarray of shape (L, T)
            Attribute weights in [0.5, 1] (Section 6.1).
        """
        # Sample from Dirichlet with concentration gamma (Section 6.1)
        betas = self.rng.dirichlet(np.full(self.N, self.gamma), size=self.L)
        if self.incomplete_scale is not None:
            betas = betas * self.incomplete_scale

        rule_weights = self.rng.uniform(0.5, 1.0, size=self.L)
        attr_weights = self.rng.uniform(0.5, 1.0, size=(self.L, self.T))

        return betas, rule_weights, attr_weights

    def generate_inputs(self, n_samples: int = 100) -> np.ndarray:
        """
        Generate random input samples.

        Returns
        -------
        inputs : np.ndarray of shape (n_samples, T)
        """
        return self.rng.uniform(0.0, 1.0, size=(n_samples, self.T))


# ---------------------------------------------------------------------------
# Hierarchical cases (Section 6.1, Table 1)
# ---------------------------------------------------------------------------

# Two-layer hierarchical rule base: N1=5 intermediate conclusions feed
# N2=3 final conclusions, with L1=20 and L2=15 rules.
TWO_LAYER_CONFIG = [
    {'N': 5, 'L': 20, 'T': 4, 'Tk_max': 2},
    {'N': 3, 'L': 15, 'T': 5, 'Tk_max': 2},
]

# Three-layer hierarchical rule base: adds a diagnostic layer with
# N3=2 and L3=12.
THREE_LAYER_CONFIG = [
    {'N': 5, 'L': 20, 'T': 4, 'Tk_max': 2},
    {'N': 3, 'L': 15, 'T': 5, 'Tk_max': 2},
    {'N': 2, 'L': 12, 'T': 3, 'Tk_max': 2},
]
