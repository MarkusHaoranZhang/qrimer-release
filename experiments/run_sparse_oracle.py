"""
Sparse column-oracle resource exploration.

Builds the reversible column-to-row oracle for the ER synthesis matrix and
reports the sparsity resources.  This is the oracle component of the sparse
block-encoding route; the row-side LCU composition is the next layer.
"""

from core.sparse_oracle import SparseEROracle


def main():
    print("=" * 72)
    print("Sparse ER column-oracle resources")
    print("=" * 72)
    print(f"  {'N':>3}  {'col qubits':>10}  {'row qubits':>10}  "
          f"{'nonzeros':>9}  {'max col':>8}  {'max row':>8}")
    for N in (2, 3, 5, 10, 20, 50, 100):
        report = SparseEROracle(N).resource_report()
        print(f"  {N:>3}  {report['column_qubits']:>10}  "
              f"{report['row_qubits']:>10}  {report['nonzeros']:>9}  "
              f"{report['max_column_degree']:>8}  "
              f"{report['max_row_degree']:>8}")


if __name__ == '__main__':
    main()
