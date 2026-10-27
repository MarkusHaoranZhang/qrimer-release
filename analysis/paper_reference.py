"""Legacy values from the previous manuscript version.

These constants are retained only for backward compatibility and historical
comparison.  The source of truth for the current manuscript is
``outputs/tables/table3_structured.json`` and
``outputs/tables/table10_generated.json``.
"""

TABLE3_DEPTH = {
    (5, 50): 214,
    (10, 100): 687,
    (15, 200): 1243,
    (20, 200): 2156,
    (30, 200): 4621,
}

TABLE3_QUBITS = {
    (5, 50): 18,
    (10, 100): 22,
    (15, 200): 26,
    (20, 200): 30,
    (30, 200): 32,
}

TABLE3_CLASSICAL_SECONDS = {
    5: 0.12,
    10: 8.45,
    15: 142.0,
    20: 3600.0,
    30: None,
}

TABLE10_NOISE = {
    'N3': {
        0.0: 0.0027, 1.0: 0.0136, 2.0: 0.0241,
        5.0: 0.0517, 10.0: 0.0983,
    },
    'N10': {
        0.0: 0.0078, 1.0: 0.023, 2.0: 0.038,
        5.0: 0.081, 10.0: 0.152,
    },
    'N20': {
        0.0: 0.0189, 1.0: 0.065, 2.0: 0.117,
        5.0: 0.249, 10.0: 0.485,
    },
}
