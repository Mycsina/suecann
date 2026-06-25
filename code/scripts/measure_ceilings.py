"""
Stage B bottleneck decomposition (2026-06-24).

The Phase-0 card-match accuracy plateaued at ~0.45. That plateau is the WANN
ceiling (Ceiling 3). To know WHAT to fix we must localize the bottleneck into:

  Ceiling φ  — φ-reachability: in how many states is SOME teacher-best card
               reachable as argmax of dot(w, phi) for some knob vector w?
               (Hard limit of the 6-feature vocabulary. If low, enrich φ,
               NOT belief — no belief→knobs function can pick an unreachable card.)
  Ceiling 2  — belief ceiling: how well can an UNCONSTRAINED learner (GBM) pick
               the best card from belief features? (Limit of the 35-feature belief.)
  Ceiling 3  — WANN: ~0.45 (measured in training).

If Ceiling-φ ≈ 0.45 → φ features are the limiter (enrich φ / drop φ-utility).
If Ceiling-φ high but Ceiling-2 ≈ 0.45 → belief is the limiter (enrich belief).
If Ceiling-2 high but Ceiling-3 ≈ 0.45 → WANN substrate is the limiter.

φ is replicated exactly from the engine tables (CARD_RANK=c%10 is the Sueca
strength order; CARD_POINTS=[0,0,0,0,0,2,3,4,10,11] per suit).
"""
import sys
import numpy as np

POINTS = np.array([0, 0, 0, 0, 0, 2, 3, 4, 10, 11], dtype=np.float64)  # per rank index 0..9

def card_beats(c, best, trump, led):
    cs, cr = divmod(c, 10); bs, br = divmod(best, 10)
    ct, bt = (cs == trump), (bs == trump)
    if ct and not bt: return True
    if (not ct) and bt: return False
    if ct and bt: return cr > br
    if cs == led and bs == led: return cr > br
    if cs == led: return True
    return False

def would_beat(c, trump, trick, tlen):
    if tlen == 0: return True
    led = trick[0] // 10
    best = trick[0]
    for i in range(1, tlen):
        if card_beats(trick[i], best, trump, led): best = trick[i]
    return card_beats(c, best, trump, led)

def compute_phi(c, trump, hand, trick, tlen):
    suit, rank = divmod(c, 10)
    win = would_beat(c, trump, trick, tlen)
    suit_count = bin(int(hand) & (0x3FF << (suit * 10))).count("1")
    tp = float(sum(POINTS[trick[i] % 10] for i in range(tlen)))
    return np.array([
        rank / 9.0,
        POINTS[rank] / 11.0,
        1.0 if suit == trump else 0.0,
        1.0 if win else 0.0,
        (tp / 30.0) if win else 0.0,
        1.0 if suit_count == 1 else 0.0,
    ])

def legal_hand(hand, trick, tlen):
    if tlen == 0: return hand
    led = trick[0] // 10
    suited = hand & (0x3FF << (led * 10))
    return suited if suited else hand

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "code/expert_states_v7.npz"
    n_sub = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    z = np.load(path)
    st = z["states"]; bc = z["best_cards"]; trump = z["ctx_trump"]
    hand = z["ctx_hand"]; trick = z["ctx_trick"]; tlen = z["ctx_trick_len"]
    N = len(bc)
    rng = np.random.default_rng(20260624)
    idx = rng.choice(N, size=min(n_sub, N), replace=False)

    # --- Build per-state legal cards, phi vectors, best mask ---
    states_phi = []      # list of (n_legal,6) arrays
    states_legal = []    # list of legal card arrays
    states_best = []     # list of bool arrays (card in mask)
    beliefs = []
    for i in idx:
        h = int(hand[i]); tr = int(trump[i]); tl = int(tlen[i])
        tk = [int(trick[i, j]) for j in range(4)]
        legal = legal_hand(h, tk, tl)
        cards = []
        phis = []
        mask = int(bc[i])
        bbest = []
        hh = legal
        while hh:
            c = (hh & -hh).bit_length() - 1
            hh &= hh - 1
            cards.append(c)
            phis.append(compute_phi(c, tr, h, tk, tl))
            bbest.append((mask >> c) & 1 == 1)
        states_legal.append(np.array(cards, dtype=np.int64))
        states_phi.append(np.array(phis, dtype=np.float64))
        states_best.append(np.array(bbest, dtype=bool))
        beliefs.append(st[i])

    # --- Ceiling φ: reachability via random knob sampling ---
    W = 4000
    wsamp = rng.uniform(-1, 1, size=(W, 6))
    reach_hit = 0
    reach_total = 0
    for phis, bbest in zip(states_phi, states_best):
        if len(phis) == 0: continue
        scores = wsamp @ phis.T          # (W, n_legal)
        argmax = scores.argmax(axis=1)   # lowest-index tiebreak, matches resolver
        reachable = np.zeros(len(phis), dtype=bool)
        reachable[argmax] = True
        if bbest.any():
            reach_hit += int((reachable & bbest).any())
            reach_total += 1
    ceiling_phi = reach_hit / max(reach_total, 1)

    # --- Baseline: best single global knob vector (one policy for all states) ---
    # Coarse search over a grid + refinement on the best.
    def cardmatch(w):
        hit = tot = 0
        for phis, bbest in zip(states_phi, states_best):
            if len(phis) == 0 or not bbest.any(): continue
            choice = int((phis @ w).argmax())
            tot += 1; hit += int(bbest[choice])
        return hit / max(tot, 1)
    grid = np.array(np.meshgrid(*[[-1, 0, 1]] * 6)).reshape(6, -1).T
    gscore = np.array([cardmatch(g) for g in grid])
    best_w = grid[gscore.argmax()]
    # local refine
    for _ in range(40):
        cand = best_w + rng.normal(0, 0.2, size=(12, 6))
        cand = np.clip(cand, -1, 1)
        for c in cand:
            if cardmatch(c) > cardmatch(best_w): best_w = c
    best_global = cardmatch(best_w)

    # --- Ceiling 2: belief ceiling via GBM (belief ⊕ phi -> is_best) ---
    from sklearn.ensemble import GradientBoostingClassifier
    # Split by STATE (keep each state's cards together).
    n_st = len(states_phi)
    perm = rng.permutation(n_st)
    n_tr = int(n_st * 0.8)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

    def build_rows(state_ids):
        X, y = [], []
        for s in state_ids:
            bel = beliefs[s]
            for p, b in zip(states_phi[s], states_best[s]):
                X.append(np.concatenate([bel, p])); y.append(int(b))
        return np.array(X), np.array(y)
    Xtr, ytr = build_rows(tr_idx)

    def eval_gbm(model, feat="both"):
        # feat: which columns to use (both | belief | phi)
        sl = {"both": slice(0, 41), "belief": slice(0, 35), "phi": slice(35, 41)}[feat]
        hit = tot = 0
        for s in te_idx:
            phis = states_phi[s]; bbest = states_best[s]
            if len(phis) == 0 or not bbest.any(): continue
            bel = np.tile(beliefs[s], (len(phis), 1))
            X = np.concatenate([bel, phis], axis=1)[:, sl]
            proba = model.predict_proba(X)[:, 1]
            choice = int(proba.argmax())
            tot += 1; hit += int(bbest[choice])
        return hit / max(tot, 1)

    full = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(Xtr, ytr)
    cm_full = eval_gbm(full, "both")
    phi_only = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(Xtr[:, 35:], ytr)
    cm_phi = eval_gbm(phi_only, "phi")
    bel_only = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(Xtr[:, :35], ytr)
    cm_bel = eval_gbm(bel_only, "belief")

    # --- Report ---
    print("\n========== STAGE B BOTTLENECK DECOMPOSITION ==========")
    print(f"dataset: {path}   (evaluated on {n_st} sampled states)")
    print(f"  mean legal cards / state : {np.mean([len(p) for p in states_phi]):.2f}")
    print(f"  mean best-cards / state  : {np.mean([b.sum() for b in states_best]):.2f}")
    print()
    print(f"  random baseline          : ~{1.0/np.mean([len(p) for p in states_phi]):.3f}  (1 / mean legal)")
    print(f"  best single global knob  : {best_global:.3f}   (one fixed policy for all states)")
    print(f"  WANN (Phase-0 plateau)   : ~0.450   (Ceiling 3, from training)")
    print()
    print(f"  Ceiling-φ  (φ-reachability, hard limit of the 6 features) : {ceiling_phi:.3f}")
    print(f"  Ceiling-2  GBM belief⊕φ  → best-card                    : {cm_full:.3f}")
    print(f"             GBM φ-only    (no belief)                     : {cm_phi:.3f}")
    print(f"             GBM belief-only (no φ)                        : {cm_bel:.3f}")
    print()
    print("Diagnosis:")
    if ceiling_phi < 0.60:
        print(f"  ⚠ Ceiling-φ ({ceiling_phi:.2f}) is LOW → the 6 φ features CANNOT distinguish the")
        print("    teacher-best card in many states. Enriching BELIEF will NOT help; the φ")
        print("    vocabulary itself is the limiter. Fix: richer φ (more card features) or")
        print("    abandon φ-utility for direct belief→card scoring.")
    else:
        print(f"  Ceiling-φ ({ceiling_phi:.2f}) is high → φ features can express the best card;")
        gap = cm_full - 0.45
        print(f"  belief⊕φ GBM reaches {cm_full:.2f} (vs WANN 0.45, gap {gap:+.2f}).")
        if cm_bel < cm_full - 0.05:
            print(f"  belief-only ({cm_bel:.2f}) ≪ belief⊕φ ({cm_full:.2f}) → φ carries most signal;")
            print("    the belief features are the weak link. Fix: ENRICH BELIEF.")
        else:
            print("  belief and φ both informative; WANN substrate likely the limiter.")
    print("======================================================")

if __name__ == "__main__":
    main()
