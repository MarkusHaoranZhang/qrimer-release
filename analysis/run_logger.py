"""Lightweight JSONL run metadata logger."""

import json
import os
import time


def log_run(log_dir, name, seed, shots, noise, metrics, **extra):
    os.makedirs(log_dir, exist_ok=True)
    record = {
        'name': name,
        'timestamp': time.time(),
        'seed': seed,
        'shots': shots,
        'noise': noise,
        'metrics': metrics,
    }
    record.update(extra)
    path = os.path.join(log_dir, 'run_metadata.jsonl')
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path
