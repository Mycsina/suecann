"""
Probe A — belief/context enrichment ceiling.

Does richer per-state context raise the realizable card-match ceiling above the
0.65 that (belief35 ⊕ φ(candidate)) reaches? If a cheap, richer feature set
pushes the GBM toward the 0.99 φ-reachability ceiling, belief/context enrichment
is worth building in Rust. If it saturates near 0.65, the current belief already
carries (almost) all the learnable signal.

Feature sets compared (all per (state, candidate-card) row, per-state argmax):
  B    : belief(35) ⊕ φ(c)(6)                         [baseline = 0.65]
  +REL : B + relative-standing features (candidate vs the other legal cards)
  +FULL: B + the φ-profile of ALL legal cards (padded to 10×6)
  REL  : relative features only (no belief)           [ablation: pure context]
"""
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

POINTS = np.array([0, 0, 0, 0, 0, 2, 3, 4, 10, 11], dtype=np.float64)

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
    led = trick[0] // 10; best = trick[0]
    for i in range(1, tlen):
        if card_beats(trick[i], best, trump, led): best = trick[i]
    return card_beats(c, best, trump, led)

def phi_of(c, trump, hand, trick, tlen):
    suit, rank = divmod(c, 10)
    win = would_beat(c, trump, trick, tlen)
    sc = bin(int(hand) & (0x3FF << (suit * 10))).count("1")
    tp = float(sum(POINTS[trick[i] % 10] for i in range(tlen)))
    return np.array([rank/9.0, POINTS[rank]/11.0, 1.0 if suit==trump else 0.0,
                     1.0 if win else 0.0, (tp/30.0) if win else 0.0,
                     1.0 if sc==1 else 0.0])

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "code/expert_states_v7.npz"
    n_sub = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    z = np.load(path)
    st, bc = z["states"], z["best_cards"]
    trump, hand, trick, tlen = z["ctx_trump"], z["ctx_hand"], z["ctx_trick"], z["ctx_trick_len"]
    N = len(bc)
    rng = np.random.default_rng(20260624)
    idx = rng.choice(N, size=min(n_sub, N), replace=False)

    rows = []  # each: dict of feature arrays per state
    beliefs = []
    for i in idx:
        h, tr, tl = int(hand[i]), int(trump[i]), int(tlen[i])
        tk = [int(trick[i, j]) for j in range(4)]
        legal = h if tl == 0 else (h & (0x3FF << (tk[0]//10 * 10)) or h)
        cards, phis = [], []
        hh = legal
        while hh:
            c = (hh & -hh).bit_length() - 1; hh &= hh - 1
            cards.append(c); phis.append(phi_of(c, tr, h, tk, tl))
        cards = np.array(cards, dtype=np.int64); phis = np.array(phis)
        mask = int(bc[i])
        tp_exact = float(sum(POINTS[trick[i, j] % 10] for j in range(tl)))
        # relative features per candidate
        ranks = phis[:, 0]; pts = phis[:, 1]; trumps = phis[:, 2]; wins = phis[:, 3]; caps = phis[:, 4]
        n = len(cards)
        rank_pct = (ranks[:, None] > ranks[None, :]).sum(1) / max(n - 1, 1)
        pt_pct = (pts[:, None] > pts[None, :]).sum(1) / max(n - 1, 1)
        wins_count = int(wins.sum()); trump_count = int(trumps.sum())
        caps_max = caps.max() if n else 0.0
        rel = np.stack([
            rank_pct, pt_pct,
            (ranks == ranks.max()).astype(float), (ranks == ranks.min()).astype(float),
            (pts == pts.max()).astype(float),
            trumps * float(trump_count == 1),                   # is_only_trump
            np.full(n, trump_count / n),                         # trump_density
            wins * float(wins_count == 1),                       # is_only_winner
            np.full(n, wins_count / n),                          # win_density
            wins * (caps == caps_max).astype(float),             # is_best_capture_winner
            np.full(n, tl),                                      # trick_len
            np.full(n, tp_exact / 30.0),                         # exact trick points
            np.full(n, tr),                                      # trump suit
            np.full(n, tk[0]//10 if tl > 0 else 4),              # led suit
            np.full(n, n),                                       # legal_count
        ], axis=1)  # (n, 15)
        rows.append((beliefs_i := st[i], phis, rel, (mask, cards)))
        beliefs.append(st[i])

    # Persist a view for splitting
    states = list(range(len(rows)))
    perm = rng.permutation(states)
    n_tr = int(len(states) * 0.8)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

    def build(state_ids, mode):
        X, y = [], []
        for s in state_ids:
            bel, phis, rel, (mask, cards) = rows[s]
            bel_t = np.tile(bel, (len(phis), 1))
            is_best = np.array([(mask >> int(c)) & 1 == 1 for c in cards])
            if mode == "B":
                feat = np.concatenate([bel_t, phis], axis=1)
            elif mode == "+REL":
                feat = np.concatenate([bel_t, phis, rel], axis=1)
            elif mode == "+FULL":
                # φ-profile of all legal cards, padded to 10×6, tiled per candidate
                prof = np.zeros(60); prof[:len(phis)*6] = phis.flatten()
                prof_t = np.tile(prof, (len(phis), 1))
                feat = np.concatenate([bel_t, phis, prof_t], axis=1)
            elif mode == "REL":
                feat = rel
            X.append(feat); y.append(is_best)
        return np.concatenate(X), np.concatenate(y)

    def eval_model(model, state_ids, mode):
        hit = tot = 0
        for s in state_ids:
            bel, phis, rel, (mask, cards) = rows[s]
            if len(phis) == 0: continue
            bel_t = np.tile(bel, (len(phis), 1))
            if mode == "B": X = np.concatenate([bel_t, phis], axis=1)
            elif mode == "+REL": X = np.concatenate([bel_t, phis, rel], axis=1)
            elif mode == "+FULL":
                prof = np.zeros(60); prof[:len(phis)*6] = phis.flatten()
                X = np.concatenate([bel_t, phis, np.tile(prof,(len(phis),1))], axis=1)
            else: X = rel
            p = model.predict_proba(X)[:, 1]
            choice = int(p.argmax())
            if (mask >> int(cards[choice])) & 1 == 1: hit += 1
            tot += 1
        return hit / max(tot, 1)

    print("\n========== PROBE A — belief/context enrichment ceiling ==========")
    print(f"dataset: {path}  ({len(rows)} states, 80/20 state-split)")
    for mode in ["B", "+REL", "+FULL", "REL"]:
        Xtr, ytr = build(tr_idx, mode)
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(Xtr, ytr)
        cm = eval_model(m, te_idx, mode)
        label = {"B":"belief⊕φ (baseline)", "+REL":"+ relative standing", "+FULL":"+ full legal φ-profile", "REL":"relative only (no belief)"}[mode]
        print(f"  {mode:6s} {label:32s}: {cm:.3f}   ({Xtr.shape[1]} features)")
    print("================================================================")

if __name__ == "__main__":
    main()
