"""High-precision fixed-point division oracle for QBRA."""


class DivisionOracle:
    """6-bit (default) division oracle.

    This is an oracle-level component: it supplies the exact fixed-point
    quotient used by the activation path without constructing a dense
    lookup unitary.
    """

    def __init__(self, bits: int = 6):
        self.bits = int(bits)
        self.qmax = 2 ** self.bits - 1

    def divide(self, v, z):
        if z <= 0:
            return 0
        return min(int(v * self.qmax // z), self.qmax)

    def divide_adaptive(self, v, z):
        if z <= 0:
            return 0.0
        qmax = self.qmax
        for scale in (qmax ** 3, qmax ** 2, qmax):
            num = int(int(v) * scale / int(z))
            if num <= qmax:
                rem = int(v) * scale - num * int(z)
                q2 = int(rem * qmax / int(z))
                return (num * qmax + min(q2, qmax)) / (scale * qmax)
        return 1.0
