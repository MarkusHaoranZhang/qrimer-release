"""Single-circuit T=4 product tree with shared intermediate registers."""

from .qbra_arithmetic import ShiftAddMultiplier


class CombinedT4ProductShared:
    """T=4 product tree reusing add/ancilla registers across multipliers."""

    def __init__(self, bits: int = 4):
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
        a2 = list(range(2 * b, 3 * b))
        a3 = list(range(3 * b, 4 * b))
        p1 = list(range(4 * b, 4 * b + m1.m))
        p2 = list(range(6 * b, 6 * b + m2.m))
        p3 = list(range(9 * b, 9 * b + m3.m))
        add = list(range(13 * b, 13 * b + m3.m))
        anc = list(range(17 * b, 17 * b + 2 * m3.m + 2))
        total = max(anc) + 1
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
                   qubits=a0 + a1 + p1 + add[:m1.m] + anc[:2 * m1.m + 2],
                   inplace=True)
        qc.compose(m2.circuit(0, 0),
                   qubits=p1 + a2 + p2 + add[:m2.m] + anc[:2 * m2.m + 2],
                   inplace=True)
        qc.compose(m3.circuit(0, 0),
                   qubits=p2 + a3 + p3 + add + anc, inplace=True)
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
