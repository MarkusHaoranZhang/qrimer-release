"""Print the paper-reference closure tables.

Values are the final manuscript's Table 3 and Table 10.  They are exposed
through ``analysis.paper_closure`` and are explicitly marked as paper
references, not local dense/operator-level measurements.
"""

from analysis.paper_closure import table3_rows, table10_rows


def main():
    print("=" * 72)
    print("Paper Table 3 (structured QER resource reference)")
    print("=" * 72)
    print(f"  {'N':>3}  {'L':>4}  {'qubits':>7}  {'depth':>7}  source")
    for row in table3_rows():
        print(f"  {row['N']:>3}  {row['L']:>4}  {row['qubits']:>7}  "
              f"{row['depth']:>7}  {row['source']}")

    print("\n" + "=" * 72)
    print("Paper Table 10 (Falcon noise reference)")
    print("=" * 72)
    print(f"  {'N':>3}  {'factor':>7}  {'epsilon_beta':>13}  source")
    for row in table10_rows():
        print(f"  {row['N']:>3}  {row['factor']:>7.1f}  "
              f"{row['epsilon_beta']:>13.4f}  {row['source']}")


if __name__ == '__main__':
    main()
