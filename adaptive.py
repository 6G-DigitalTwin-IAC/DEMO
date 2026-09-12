"""
adaptive.py -- the kNN-driven adaptive gate.

This is the piece that makes the gate self-tuning. Instead of a fixed
threshold, the gate asks a kNN "given the conditions I can see RIGHT NOW,
what threshold should I use?" -- and the conditions it sees come only from
the twin's stale belief, never from reality. It is exactly as blind as the
rest of the system.

Two parts:
  1. AdaptiveThresholdKNN -- loads the training table (the CSV the team
     generated) and, given (telemetry_delay, sigma), returns the best
     threshold, using distance-weighted k nearest neighbours. Same method
     the team's notebook uses; reimplemented here with no sklearn so the
     simulator has no extra dependency.

  2. measure_sigma_from_twin -- works out "how uneven are the link loads"
     from the twin's CURRENT belief. This is the one genuinely new bit of
     logic: in a live system nobody hands you sigma, you infer it from
     what you can see (the stale twin).
"""

import csv
import numpy as np


class AdaptiveThresholdKNN:
    def __init__(self, table_csv, k=3):
        self.k = k
        rows = []
        with open(table_csv) as f:
            for r in csv.DictReader(f):
                rows.append((float(r["telemetry_delay_s"]),
                             float(r["load_sigma"]),
                             float(r["best_threshold"])))
        arr = np.array(rows, dtype=float)
        self.X = arr[:, :2]          # (delay, sigma)
        self.Y = arr[:, 2]           # best_threshold
        # min-max scale each input to 0-1 so neither dominates distance.
        # Store the scale so live queries use the SAME scaling.
        self.xmin = self.X.min(axis=0)
        self.xmax = self.X.max(axis=0)
        self.span = np.where(self.xmax - self.xmin > 0,
                             self.xmax - self.xmin, 1.0)
        self.Xs = (self.X - self.xmin) / self.span

    def predict(self, delay, sigma):
        q = (np.array([delay, sigma]) - self.xmin) / self.span
        d = np.linalg.norm(self.Xs - q, axis=1)
        nk = np.argsort(d)[:self.k]
        w = 1.0 / (d[nk] + 1e-8)
        w = w / w.sum()
        th = float(np.average(self.Y[nk], weights=w))
        # safety: keep inside the sane threshold band
        return min(1.0, max(0.5, th))


def measure_sigma_from_twin(twin):
    """
    Estimate load unevenness (sigma) from the twin's CURRENT belief.

    In training, sigma was a knob we set. Live, the system must infer it
    from what it sees -- and all it sees is the twin's stale snapshot.
    We take the standard deviation of the believed link loads. That's the
    honest, measurable stand-in for 'how uneven is the network right now'.

    Before the twin has any belief, fall back to a neutral mid value.
    """
    if not getattr(twin, "_belief", None):
        return 0.15
    vals = np.array(list(twin._belief.values()), dtype=float)
    if len(vals) < 2:
        return 0.15
    return float(np.std(vals))
