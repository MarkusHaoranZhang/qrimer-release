"""
Row-side LCU decomposition exploration.

Reports the six-term partial-isometry decomposition of the ER synthesis
matrix and its resource parameters.
"""

from core.row_lcu import row_lcu_report


def main():
    print("=" * 72)
    print("Row-side LCU decomposition")
    print("=" * 72)
    print(f"  {'N':>3}  {'terms':>6}  {'ancilla':>8}  {'nonzeros':>9}  "
          f"{'max rows/term':>14}")
    for N in (2, 3, 5, 10, 20, 50, 100):
        report = row_lcu_report(N)
        print(f"  {N:>3}  {report['terms']:>6}  "
              f"{report['ancilla_qubits']:>8}  {report['nonzeros']:>9}  "
              f"{report['max_rows_per_term']:>14}")


if __name__ == '__main__':
    main()
