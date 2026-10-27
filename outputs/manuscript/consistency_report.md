# Final consistency report

## Table 2 vs manuscript
- N=5: Q-RIMER 0.0082 (old 0.0031); BF-QC 0.4557 (old 0.0147)
- N=10: Q-RIMER 0.0085 (old 0.0078); BF-QC 0.3203 (old 0.0623)
- N=15: Q-RIMER 0.0086 (old 0.0124); BF-QC 0.2639 (old 0.1338)
- N=20: Q-RIMER 0.0096 (old 0.0189); BF-QC 0.2277 (old 0.2415)
- N=30: Q-RIMER 0.0105 (old 0.0196); BF-QC 0.1859 (old 0.3872)

## Table 3 vs manuscript
- N=3, L=36: qubits 27 (old None); step depth 3637 (old None)
- N=5, L=50: qubits 28 (old 18); step depth 14711 (old 214)
- N=10, L=100: qubits 31 (old 22); step depth 237041 (old 687)
- N=15, L=200: qubits 33 (old 26); step depth 949233 (old 1243)
- N=20, L=200: qubits 33 (old 30); step depth 949233 (old 2156)

## Table 4 vs manuscript
- N=3, scale=0.5: 0.0066 (old None)
- N=3, scale=1.0: 0.0108 (old 0.0136)
- N=3, scale=2.0: 0.0102 (old 0.0241)
- N=3, scale=5.0: 0.0323 (old 0.0517)
- N=3, scale=10.0: 0.0600 (old 0.0983)

## Table 5 vs manuscript
- N=3, scale=0.0: 0.0074 (old 0.0027)
- N=3, scale=1.0: 0.0101 (old 0.0136)
- N=3, scale=2.0: 0.0127 (old 0.0241)
- N=3, scale=5.0: 0.0291 (old 0.0517)
- N=3, scale=10.0: 0.0572 (old 0.0983)
- N=10, scale=0.0: 0.0073 (old 0.0078)
- N=10, scale=1.0: 0.0342 (old 0.023)
- N=10, scale=2.0: 0.0542 (old 0.038)
- N=10, scale=5.0: 0.1075 (old 0.081)
- N=10, scale=10.0: 0.1710 (old 0.152)
- N=20, scale=0.0: 0.0081 (old 0.0189)
- N=20, scale=1.0: 0.0552 (old 0.065)
- N=20, scale=2.0: 0.0820 (old 0.117)
- N=20, scale=5.0: 0.1388 (old 0.249)
- N=20, scale=10.0: 0.1819 (old 0.485)

## Section 6.7 vs manuscript
- N=15: direct 0.0019s, vectorized 0.0030s, speedup 0.64x (old direct/vectorized (142.0, 18.6))
- N=20: direct 0.0018s, vectorized 0.0030s, speedup 0.61x (old direct/vectorized (3600.0, 420.0))

## Table 10 generated
- N=3, p=0.0: error 0.0000, success 5.044e-25
- N=3, p=0.001: error 0.0432, success 4.892e-25
- N=3, p=0.005: error 0.0846, success 3.810e-25
- N=3, p=0.01: error 0.0850, success 2.608e-25
- N=10, p=0.0: error 0.0000, success 7.054e-70
- N=10, p=0.001: error 0.1391, success 5.316e-70
- N=10, p=0.005: error 0.1328, success 1.309e-70
- N=10, p=0.01: error 0.1008, success 2.112e-71
- N=20, p=0.0: error 0.0000, success 8.458e-140
- N=20, p=0.001: error 0.1481, success 3.801e-140
- N=20, p=0.005: error 0.0790, success 1.570e-141
- N=20, p=0.01: error 0.0427, success 3.189e-143

## Freeze status
- Code artifacts: frozen after this report.
- Legacy paper references remain in analysis/paper_reference.py.
- Manuscript must use generated values and the updated noise and timing caveats.