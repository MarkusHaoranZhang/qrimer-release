"""Quantum arithmetic primitives for QBRA.

This module implements the first executable arithmetic stage of rule
activation: weighted matching degrees and their sum in fixed-point binary.
It is deliberately explicit about scope: it computes

    v_k = theta_k * alpha_k
    Z   = sum_k v_k

for T=1 activation.  The nonlinear T>1 product and the reciprocal are built
on top of these primitives in subsequent stages.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CDKMRippleCarryAdder,
    WeightedAdder,
)
from qiskit.quantum_info import Statevector


class QBRAArithmetic:
    def __init__(self, bits: int = 6, scale: int = 64):
        self.bits = int(bits)
        self.scale = int(scale)

    def quantize(self, value: float) -> int:
        value = float(np.clip(value, 0.0, 1.0))
        return int(round(value * (2 ** self.bits - 1)))

    def dequantize(self, value: int) -> float:
        return value / (2 ** self.bits - 1)

    def _weighted_register(self, alpha: float, theta: float):
        """Prepare |alpha_bits> and return the weighted-sum register size."""
        a = self.quantize(alpha)
        t = max(1, int(round(theta * self.scale)))
        weights = [t * (2 ** j) for j in range(self.bits)]
        adder = WeightedAdder(self.bits, weights)
        n_sum = adder.num_sum_qubits
        return a, adder, n_sum

    def weighted_value_circuit(self, alpha: float, theta: float) -> QuantumCircuit:
        a, adder, n_sum = self._weighted_register(alpha, theta)
        qc = QuantumCircuit(adder.num_qubits)
        for j in range(self.bits):
            if (a >> j) & 1:
                qc.x(j)
        qc.append(adder, range(adder.num_qubits))
        return qc

    def weighted_value(self, alpha: float, theta: float) -> float:
        qc = self.weighted_value_circuit(alpha, theta)
        sv = Statevector.from_instruction(qc)
        probs = np.abs(sv.data) ** 2
        idx = int(np.argmax(probs))
        n_sum = qc.num_qubits - self.bits
        value = (idx >> self.bits) & ((1 << n_sum) - 1)
        return value / (self.scale * (2 ** self.bits - 1))

    def weighted_sum_circuit(self, alphas, thetas) -> QuantumCircuit:
        if len(alphas) != len(thetas):
            raise ValueError("alphas and thetas must have equal length")
        values = [self.quantize(a) for a in alphas]
        tvals = [max(1, int(round(t * self.scale))) for t in thetas]
        weights = []
        for t in tvals:
            weights.extend([t * (2 ** j) for j in range(self.bits)])
        adder = WeightedAdder(self.bits * len(values), weights)
        qc = QuantumCircuit(adder.num_qubits)
        for k, a in enumerate(values):
            for j in range(self.bits):
                if (a >> j) & 1:
                    qc.x(k * self.bits + j)
        qc.append(adder, range(adder.num_qubits))
        return qc

    def weighted_sum(self, alphas, thetas) -> float:
        qc = self.weighted_sum_circuit(alphas, thetas)
        sv = Statevector.from_instruction(qc)
        probs = np.abs(sv.data) ** 2
        idx = int(np.argmax(probs))
        n_sum = qc.num_qubits - self.bits * len(alphas)
        value = (idx >> (self.bits * len(alphas))) & ((1 << n_sum) - 1)
        return value / (self.scale * (2 ** self.bits - 1))

    def activation_weights(self, alphas, thetas) -> np.ndarray:
        """T=1 activation path: weighted values, Z, and quantum division."""
        vals = [self.weighted_value(a, t) for a, t in zip(alphas, thetas)]
        Z = sum(vals)
        divider = DivisionLookup(bits=3)
        qmax = 2 ** divider.bits - 1
        weights = []
        for v in vals:
            vq = int(round(v * self.scale))
            zq = max(1, int(round(Z * self.scale)))
            q = divider.divide(min(vq, qmax), min(zq, qmax))
            weights.append(q / qmax)
        weights = np.asarray(weights, dtype=float)
        return weights / max(weights.sum(), 1e-15)

    def activation_weights_t4(self, alphas, thetas) -> np.ndarray:
        """T=4 activation path using the shift-add product tree."""
        if len(alphas) != len(thetas):
            raise ValueError("alphas and thetas must have equal length")
        from .qbra_combined import CombinedT4ProductShared
        qprod = CombinedT4ProductShared(bits=4)
        qvals = [[int(round(float(a) * 15)) for a in values]
                 for values in alphas]
        products = qprod.batch_evaluate(qvals)
        vals = [t * p for t, p in zip(thetas, products)]
        Z = sum(vals)
        divider = DivisionLookup(bits=4)
        weights = []
        for v in vals:
            weights.append(divider.divide_adaptive(int(round(v)), int(round(Z))))
        weights = np.asarray(weights, dtype=float)
        return weights / max(weights.sum(), 1e-15)

    def activation_weights_t2(self, alphas, thetas) -> np.ndarray:
        """T=2 activation path using the product and division oracles."""
        if len(alphas) != len(thetas):
            raise ValueError("alphas and thetas must have equal length")
        products = []
        for a0, a1 in alphas:
            prod = ProductLookup(T=2, bits=2)
            x0 = int(round(float(a0) * 3))
            x1 = int(round(float(a1) * 3))
            products.append(prod.evaluate([x0, x1]))
        vals = [t * p for t, p in zip(thetas, products)]
        Z = sum(vals)
        divider = DivisionLookup(bits=4)
        weights = []
        for v in vals:
            weights.append(divider.divide_adaptive(int(round(v)), int(round(Z))))
        weights = np.asarray(weights, dtype=float)
        return weights / max(weights.sum(), 1e-15)


class DivisionLookup:
    """Exact small-bit division oracle.

    Maps |v,z,0> -> |v,z,floor(v*(2^b-1)/z)> for z>0.  The input is retained,
    so the mapping is a permutation and can be executed as a unitary.  This is
    a finite-bit building block for the QBRA reciprocal; it is not a
    polynomial-depth arithmetic circuit.
    """

    def __init__(self, bits: int = 2):
        self.bits = int(bits)
        self.n_in = 2 * self.bits
        self.n_out = self.bits
        self.n_total = self.n_in + self.n_out
        self._matrix_cache = None
        self._circuit_cache = None

    def quotient(self, v: int, z: int) -> int:
        if z <= 0:
            return 0
        return min(int(v * (2 ** self.bits - 1) // z), 2 ** self.bits - 1)

    def matrix(self) -> np.ndarray:
        if self._matrix_cache is not None:
            return self._matrix_cache
        dim = 1 << self.n_total
        u = np.zeros((dim, dim), dtype=complex)
        for v in range(2 ** self.bits):
            for z in range(2 ** self.bits):
                q = self.quotient(v, z)
                for q_in in range(2 ** self.n_out):
                    src = ((v << (self.bits + self.n_out))
                           | (z << self.n_out) | q_in)
                    u[src ^ q, src] = 1.0
        self._matrix_cache = u
        return u

    def circuit(self):
        if self._circuit_cache is not None:
            return self._circuit_cache
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        qc = QuantumCircuit(self.n_total)
        qc.append(UnitaryGate(self.matrix(), label='DIV'), range(self.n_total))
        self._circuit_cache = qc
        return qc

    def divide(self, v: int, z: int) -> int:
        from qiskit.quantum_info import Statevector
        qc = self.circuit()
        src = (v << (self.bits + self.n_out)) | (z << self.n_out)
        psi = np.zeros(1 << self.n_total, dtype=complex)
        psi[src] = 1.0
        out = Statevector(psi).evolve(qc)
        idx = int(np.argmax(np.abs(out.data) ** 2))
        return idx & ((1 << self.n_out) - 1)

    def divide_fixed(self, v: int, z: int, stages: int = 2) -> float:
        """Multi-stage fixed-point division using the lookup oracle."""
        if z <= 0:
            return 0.0
        qmax = 2 ** self.bits - 1
        q1 = self.divide(min(int(v), qmax), min(int(z), qmax))
        rem = int(v) * qmax - q1 * int(z)
        q2 = self.divide(min(max(rem, 0), qmax), min(int(z), qmax))
        return (q1 * qmax + q2) / (qmax * qmax)

    def divide_adaptive(self, v: int, z: int) -> float:
        """Multi-scale division with two quotient digits per scale."""
        if z <= 0:
            return 0.0
        qmax = 2 ** self.bits - 1
        for scale in (qmax ** 3, qmax ** 2, qmax):
            num = int(int(v) * scale / int(z))
            if num <= qmax:
                q1 = self.divide(max(num, 0), qmax)
                rem = int(v) * scale - q1 * int(z)
                q2 = int(rem * qmax / int(z))
                q2 = self.divide(min(max(q2, 0), qmax), qmax)
                return (q1 * qmax + q2) / (scale * qmax)
        q = self.divide(qmax, qmax)
        return q / qmax


class ProductLookup:
    """Finite-bit product oracle for a small number of antecedents.

    Maps |x_1,...,x_T,0> to |x_1,...,x_T, prod_i x_i> using an XOR
    permutation.  It is an exact small-bit primitive for the T>1 QBRA
    product stage, not a polynomial-depth multiplier.
    """

    def __init__(self, T: int = 2, bits: int = 2):
        self.T = int(T)
        self.bits = int(bits)
        self.n_in = self.T * self.bits
        self.n_out = self.T * self.bits
        self.n_total = self.n_in + self.n_out
        self._matrix_cache = None
        self._circuit_cache = None

    def product(self, values) -> int:
        p = 1
        for v in values:
            p *= int(v)
        return min(p, 2 ** self.n_out - 1)

    def matrix(self) -> np.ndarray:
        if self._matrix_cache is not None:
            return self._matrix_cache
        dim = 1 << self.n_total
        u = np.zeros((dim, dim), dtype=complex)
        for x in range(1 << self.n_in):
            vals = [(x >> (k * self.bits)) & ((1 << self.bits) - 1)
                    for k in range(self.T)]
            prod = self.product(vals)
            for q_in in range(1 << self.n_out):
                src = (x << self.n_out) | q_in
                u[src ^ prod, src] = 1.0
        self._matrix_cache = u
        return u

    def circuit(self):
        if self._circuit_cache is not None:
            return self._circuit_cache
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        qc = QuantumCircuit(self.n_total)
        qc.append(UnitaryGate(self.matrix(), label='PROD'), range(self.n_total))
        self._circuit_cache = qc
        return qc

    def evaluate(self, values) -> int:
        from qiskit.quantum_info import Statevector
        x = 0
        for k, v in enumerate(values):
            x |= int(v) << (k * self.bits)
        src = x << self.n_out
        psi = np.zeros(1 << self.n_total, dtype=complex)
        psi[src] = 1.0
        out = Statevector(psi).evolve(self.circuit())
        idx = int(np.argmax(np.abs(out.data) ** 2))
        return idx & ((1 << self.n_out) - 1)


class ShiftAddMultiplier:
    """n-bit x b-bit reversible multiplier via controlled shift-add."""

    def __init__(self, n: int, b_bits: int = 2):
        self.n = int(n)
        self.b_bits = int(b_bits)
        self.m = self.n + self.b_bits
        self.total = 5 * self.n + 5 * self.b_bits + 2
        self._adder = None

    def circuit(self, a: int = 0, b: int = 0):
        from qiskit import QuantumCircuit
        n, m, nb = self.n, self.m, self.b_bits
        qc = QuantumCircuit(self.total)
        aq = list(range(n))
        bq = list(range(n, n + nb))
        out = list(range(n + nb, n + nb + m))
        add = list(range(n + nb + m, n + nb + 2 * m))
        anc = list(range(n + nb + 2 * m, n + nb + 2 * m + 2 * m + 2))
        for j in range(n):
            if (a >> j) & 1:
                qc.x(aq[j])
        for j in range(nb):
            if (b >> j) & 1:
                qc.x(bq[j])
        if self._adder is None:
            self._adder = CDKMRippleCarryAdder(m, kind='half')
        adder = self._adder
        for j in range(nb):
            for i in range(n):
                if i + j < m:
                    qc.cx(aq[i], add[i + j])
            qc.append(adder.control(1),
                      [bq[j]] + add + out + anc[0:2])
            for i in range(n):
                if i + j < m:
                    qc.cx(aq[i], add[i + j])
        return qc

    def evaluate(self, a: int, b: int) -> int:
        return self.batch_evaluate([(a, b)])[0]

    def batch_evaluate(self, pairs) -> list:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        circuits = []
        for a, b in pairs:
            qc = self.circuit(a, b)
            qc.measure_all()
            circuits.append(qc)
        trans = transpile(circuits, basis_gates=['u', 'cx'],
                          optimization_level=0)
        sim = AerSimulator(
            method='matrix_product_state',
            matrix_product_state_max_bond_dimension=128,
        )
        result = sim.run(trans, shots=1, seed_simulator=42).result()
        counts_list = result.get_counts()
        if isinstance(counts_list, dict):
            counts_list = [counts_list]
        out = []
        for counts in counts_list:
            bits = int(next(iter(counts)), 2)
            out.append((bits >> (self.n + self.b_bits)) & ((1 << self.m) - 1))
        return out


class T4ProductShiftAdd:
    """T=4 fixed-point product from a shift-add multiplier tree."""

    def __init__(self, bits: int = 2):
        self.bits = int(bits)
        n = self.bits
        self._m1 = ShiftAddMultiplier(n, self.bits)
        self._m2 = ShiftAddMultiplier(2 * n, self.bits)
        self._m3 = ShiftAddMultiplier(3 * n, self.bits)

    def evaluate(self, values) -> int:
        return self.batch_evaluate([values])[0]

    def batch_evaluate(self, rules) -> list:
        rules = [list(v) for v in rules]
        for values in rules:
            if len(values) != 4:
                raise ValueError("T4ProductShiftAdd expects four values")
        p1 = self._m1.batch_evaluate([(v[0], v[1]) for v in rules])
        p2 = self._m2.batch_evaluate([(p, v[2]) for p, v in zip(p1, rules)])
        p3 = self._m3.batch_evaluate([(p, v[3]) for p, v in zip(p2, rules)])
        return p3


class CombinedT4Product:
    """Single-circuit T=4 product tree for one or more 4-tuples."""

    def __init__(self, bits: int = 2):
        self.bits = int(bits)
        b = self.bits
        self._m1 = ShiftAddMultiplier(b, b)
        self._m2 = ShiftAddMultiplier(2 * b, b)
        self._m3 = ShiftAddMultiplier(3 * b, b)
        self._p3_start = 9 * b

    def circuit(self, values):
        b = self.bits
        m1, m2, m3 = self._m1, self._m2, self._m3
        a0 = list(range(0, b))
        a1 = list(range(b, 2 * b))
        p1 = list(range(2 * b, 2 * b + m1.m))
        a2 = list(range(4 * b, 5 * b))
        p2 = list(range(5 * b, 5 * b + m2.m))
        a3 = list(range(8 * b, 9 * b))
        p3 = list(range(9 * b, 9 * b + m3.m))
        add1 = list(range(13 * b, 13 * b + m1.m))
        add2 = list(range(15 * b, 15 * b + m2.m))
        add3 = list(range(18 * b, 18 * b + m3.m))
        anc1 = list(range(22 * b, 22 * b + 2 * m1.m + 2))
        anc2 = list(range(22 * b + 2 * m1.m + 2,
                           22 * b + 2 * m1.m + 2 + 2 * m2.m + 2))
        anc3 = list(range(22 * b + 2 * m1.m + 2 + 2 * m2.m + 2,
                           22 * b + 2 * m1.m + 2 + 2 * m2.m + 2
                           + 2 * m3.m + 2))
        total = max(anc3) + 1
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(total)
        for j in range(b):
            if (int(values[0]) >> j) & 1:
                qc.x(a0[j])
            if (int(values[1]) >> j) & 1:
                qc.x(a1[j])
            if (int(values[2]) >> j) & 1:
                qc.x(a2[j])
            if (int(values[3]) >> j) & 1:
                qc.x(a3[j])
        qc.compose(m1.circuit(0, 0),
                   qubits=a0 + a1 + p1 + add1 + anc1, inplace=True)
        qc.compose(m2.circuit(0, 0),
                   qubits=p1 + a2 + p2 + add2 + anc2, inplace=True)
        qc.compose(m3.circuit(0, 0),
                   qubits=p2 + a3 + p3 + add3 + anc3, inplace=True)
        qc.measure_all()
        return qc

    def batch_evaluate(self, rules) -> list:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        circuits = [self.circuit(v) for v in rules]
        trans = transpile(circuits, basis_gates=['u', 'cx'],
                          optimization_level=0)
        sim = AerSimulator(
            method='matrix_product_state',
            matrix_product_state_max_bond_dimension=128,
        )
        result = sim.run(trans, shots=1, seed_simulator=42).result()
        counts_list = result.get_counts()
        if isinstance(counts_list, dict):
            counts_list = [counts_list]
        out = []
        for counts in counts_list:
            bits = int(next(iter(counts)), 2)
            out.append((bits >> self._p3_start) & ((1 << self._m3.m) - 1))
        return out

    def evaluate(self, values) -> int:
        return self.batch_evaluate([values])[0]
