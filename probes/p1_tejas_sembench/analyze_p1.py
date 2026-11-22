"""P1 (policies): does the energy-optimal safe model differ per tuple and per deployment?

Reads tuples_<tier>.csv + energy_<tier>.json for tiers given smallest -> largest and
evaluates, with 5-fold cross-validation grouped by movie (thresholds fitted on the
training folds, reported on the held-out fold):

  static_<tier>        one model for every tuple
  uniform_downgrade    largest tier -> next smaller tier for every tuple (Tejas' "fixed" baseline)
  learned_router       Tejas-style: predict per tier whether it answers correctly from
                       pre-inference features, pick the cheapest tier predicted safe
  cascade              smallest tier first, escalate while confidence < tau
  oracle_correct       cheapest tier that is correct (non-deployable bound)

Quality requirement: aggregate accuracy >= accuracy(reference) - 2 pp, where the reference is
  ref=large  the largest tier (as in Tejas), or
  ref=best   the most accurate static tier on the training folds (accuracy is not monotone in
             model size on every task, so "largest" can be a weak reference).
Router and cascade only use tiers up to the reference (cheapest first) and fall back to it.
Reported: energy vs static-largest, accuracy, delta pp, violation rate (largest right,
policy wrong), regret vs oracle, tier mix.

  python -m probes.p1_tejas_sembench.analyze_p1 probes/results/p1/<run> small,medium,large [large|best]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REQ_PP = 2.0


def load(run: Path, tiers):
    cols = {}
    for t in tiers:
        with open(run / f"tuples_{t}.csv") as f:
            rows = list(csv.DictReader(f))
        cols[t] = rows
    ids = [r["reviewId"] for r in cols[tiers[0]]]
    for t in tiers:
        assert [r["reviewId"] for r in cols[t]] == ids, "tier files must cover the same tuples in order"
    base = cols[tiers[0]]
    label = np.array([int(r["label"]) for r in base])
    movie = np.array([r["movie"] for r in base])
    feats = np.array([[np.log1p(float(r["len_chars"])), float(r["isTopCritic"] == "True")] for r in base])
    ans = np.stack([[int(r["answer"]) for r in cols[t]] for t in tiers], 1)
    pyes = np.stack([[float(r["p_yes"]) for r in cols[t]] for t in tiers], 1)
    tok = np.stack([[float(r["prompt_tokens"] or 1) for r in cols[t]] for t in tiers], 1)
    # Energy per tuple per tier: tier pass energy (mean over reps) split by prompt-token share.
    e = np.zeros_like(tok)
    metric = "gpu_j"
    for j, t in enumerate(tiers):
        meta = json.loads((run / f"energy_{t}.json").read_text())
        vals = [p["window"]["gpu_j"] for p in meta["passes"] if p["window"]["gpu_j"] is not None]
        if not vals:
            metric = "seconds"
            vals = [p["window"]["seconds"] for p in meta["passes"]]
        e[:, j] = np.mean(vals) * tok[:, j] / tok[:, j].sum()
    return label, movie, feats, ans, pyes, e, metric


def folds(movie, k=5, seed=0):
    ms = np.unique(movie)
    rng = np.random.default_rng(seed)
    rng.shuffle(ms)
    return [np.isin(movie, ms[i::k]) for i in range(k)]


def logreg(X, y, iters=500, lr=0.1):
    X = np.c_[np.ones(len(X)), (X - X.mean(0)) / (X.std(0) + 1e-9)]
    mu, sd = None, None
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        w -= lr * X.T @ (p - y) / len(y)
    return w


def logreg_predict(w, Xtr, X):
    Z = np.c_[np.ones(len(X)), (X - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)]
    return 1 / (1 + np.exp(-Z @ w))


def evaluate(choice, ans, label, e):
    n = len(label)
    pred = ans[np.arange(n), choice]
    acc = (pred == label).mean()
    return pred, acc, e[np.arange(n), choice].sum()


def cascade_choice(pyes, tau):
    conf = np.maximum(pyes, 1 - pyes)
    n, T = pyes.shape
    choice = np.full(n, T - 1)
    invoked = np.zeros((n, T), bool)
    for i in range(n):
        for j in range(T):
            invoked[i, j] = True
            if j == T - 1 or conf[i, j] >= tau:
                choice[i] = j
                break
    return choice, invoked


def main():
    run, tiers = Path(sys.argv[1]), sys.argv[2].split(",")
    ref_mode = sys.argv[3] if len(sys.argv) > 3 else "large"
    label, movie, feats, ans, pyes, e, metric = load(run, tiers)
    n, T = ans.shape
    L = T - 1
    correct = ans == label[:, None]
    R = L if ref_mode == "large" else int(np.argmax(correct.mean(0)))
    res = {}

    def record(name, choice, energy, mask_all=None):
        pred = ans[np.arange(n), choice]
        acc = (pred == label).mean()
        res[name] = {
            "energy_vs_large": float(energy / e[:, L].sum()),
            "accuracy": float(acc),
            "delta_pp": float(100 * (acc - correct[:, R].mean())),
            "violation_rate": float(((correct[:, R]) & (pred != label)).mean()),
            "tier_mix": {t: float((choice == j).mean()) for j, t in enumerate(tiers)},
        }

    for j, t in enumerate(tiers):
        record(f"static_{t}", np.full(n, j), e[:, j].sum())
    record("uniform_downgrade", np.full(n, L - 1), e[:, L - 1].sum())

    # oracle: cheapest correct tier (else largest)
    oc = np.array([next((j for j in range(T) if correct[i, j]), R) for i in range(n)])
    record("oracle_correct", oc, e[np.arange(n), oc].sum())
    e_oracle = e[np.arange(n), oc].sum()

    # cross-validated learned router and cascade
    rc, cc = np.full(n, L), np.full(n, L)
    cas_energy = 0.0
    for test in folds(movie):
        tr = ~test
        Rf = L if ref_mode == "large" else int(np.argmax(correct[tr].mean(0)))
        acc_req = correct[tr, Rf].mean() - REQ_PP / 100
        if Rf == 0:
            rc[test] = 0; cc[test] = 0; cas_energy += e[test, 0].sum(); continue
        # learned router: per cheaper tier, P(correct | features); fallback = reference tier
        ws = [logreg(feats[tr], correct[tr, j].astype(float)) for j in range(Rf)]
        p_tr = np.stack([logreg_predict(ws[j], feats[tr], feats[tr]) for j in range(Rf)], 1)
        p_te = np.stack([logreg_predict(ws[j], feats[tr], feats[test]) for j in range(Rf)], 1)
        best_theta = 1.01
        for theta in np.linspace(0.5, 1.0, 51)[::-1]:
            ch = np.array([next((j for j in range(Rf) if p_tr[i, j] >= theta), Rf) for i in range(tr.sum())])
            if (ans[tr][np.arange(tr.sum()), ch] == label[tr]).mean() >= acc_req:
                best_theta = theta
            else:
                break
        rc[test] = [next((j for j in range(Rf) if p_te[i, j] >= best_theta), Rf) for i in range(test.sum())]
        # cascade: smallest tau meeting the requirement on the training folds
        best_tau = 1.01
        for tau in np.linspace(0.5, 1.0, 51)[::-1]:
            ch, _ = cascade_choice(pyes[tr][:, :Rf + 1], tau)
            if (ans[tr][np.arange(tr.sum()), ch] == label[tr]).mean() >= acc_req:
                best_tau = tau
            else:
                break
        ch, inv = cascade_choice(pyes[test][:, :Rf + 1], best_tau)
        cc[test] = ch
        cas_energy += (e[test][:, :Rf + 1] * inv).sum()
    record("learned_router", rc, e[np.arange(n), rc].sum())
    record("cascade", cc, cas_energy)
    for k in res:
        en = res[k]["energy_vs_large"] * e[:, L].sum()
        res[k]["regret_vs_oracle"] = float((en - e_oracle) / e[:, L].sum())
        res[k]["meets_requirement"] = bool(res[k]["delta_pp"] >= -REQ_PP - 1e-9)

    (run / f"policies_ref-{ref_mode}.json").write_text(json.dumps(
        {"metric": metric, "tiers": tiers, "n": n, "ref": ref_mode, "ref_tier": tiers[R], "policies": res}, indent=2))
    print(f"metric={metric} n={n} tiers={tiers} ref={ref_mode}->{tiers[R]} acc_ref={correct[:, R].mean():.3f}")
    for k, v in res.items():
        print(f"{k:18s} E/E_large={v['energy_vs_large']:.3f} acc={v['accuracy']:.3f} dpp={v['delta_pp']:+.1f} "
              f"viol={v['violation_rate']:.3f} regret={v['regret_vs_oracle']:.3f} ok={v['meets_requirement']} "
              f"mix={ {t: round(x, 2) for t, x in v['tier_mix'].items()} }")


if __name__ == "__main__":
    main()
