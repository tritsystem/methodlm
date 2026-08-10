#!/usr/bin/env python3
"""MethodLM -- the method, in a language model, kept honest by a ledger.

Point it at data; it computes on it (ternary two-timescale readout) and reasons about
it (gbranaa-hue method), pre-registering every test.

  python methodlm.py --demo                      interventional demo world (hidden
                                                 confound + answer key at the end)
  python methodlm.py --diabetes                  real clinical data (442 patients,
                                                 sklearn load_diabetes, raw units)
  python methodlm.py --csv F --target COL        any recorded CSV (observational)

Pipeline (identical in all modes):
  COMPUTE  a tritkit TwoTimescaleLinear ternary readout learns target from the other
           columns -> NMSE + structural evidence share + gate selection.
  REASON   the method copilot (Qwen-3B + gbranaa-hue method) investigates with tools,
           pre-registers every test, writes an honest ledger.

Tools by mode:  CORR (all) | ATTR (all) | RUN true intervention (demo world only)
                STRAT observational conditioning (recorded data): corr(x,target)
                inside quartile bands of z -- the honest substitute for clamping
                when you cannot rerun the world.
"""
import argparse, os, re, sys, subprocess, textwrap, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_tk = os.environ.get("METHODLM_TRITKIT")     # optional: path to the tritkit pkg (ternary 2nd witness)
if _tk:
    sys.path.insert(0, _tk)
import methodlm_models
rng = np.random.default_rng(11)

BACKEND = None      # the model driving the reasoning; set by main() or lazily to local
def backend():
    global BACKEND
    if BACKEND is None:
        BACKEND = methodlm_models.get_model(os.environ.get("METHODLM_MODEL", "local"), HERE)
    return BACKEND

# ---------------- data sources ----------------
def demo_world(n=2000, clamp=None):
    clamp = clamp or {}
    season = np.cumsum(rng.normal(0, 0.15, n)); season -= season.mean()
    d = {"temperature": 25 + 6 * np.tanh(season) + rng.normal(0, 0.8, n),
         "humidity":    50 + 15 * np.tanh(season) + rng.normal(0, 2.5, n),
         "vibration":   rng.normal(0, 1, n)}
    for k, v in clamp.items():
        if k in d: d[k] = np.full(n, float(v))
    d["error"] = np.clip(0.5 + 0.08 * (d["humidity"] - 50) + rng.normal(0, 0.35, n), 0, None)
    return d

def load_diabetes():
    from sklearn.datasets import load_diabetes as ld
    raw = ld(scaled=False)
    names = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]
    d = {n: raw.data[:, i].astype(float) for i, n in enumerate(names)}
    d["progression"] = raw.target.astype(float)
    return d

def load_csv(path, target):
    import csv as _csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(_csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        try:
            cols[k.strip()] = np.array([float(r[k]) for r in rows])
        except ValueError:
            pass                                   # skip non-numeric columns
    assert target in cols, f"target '{target}' not among numeric columns {list(cols)}"
    return cols

# ---------------- COMPUTE: ternary two-timescale readout ----------------
def train_readout(data, target):
    import torch
    import torch.nn.functional as F
    from tritkit.twotimescale import TwoTimescaleLinear
    torch.manual_seed(0)
    names = [k for k in data if k != target]
    X = np.column_stack([data[k] for k in names])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Yv = data[target]; Yn = (Yv - Yv.mean()) / Yv.std()
    Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Yn, dtype=torch.float32)
    n = len(Yn)
    lyr = TwoTimescaleLinear(len(names), 1, density=min(0.5, 3 / len(names) + 0.15),
                             bias=True, evidence_beta=0.97)
    opt = torch.optim.SGD(lyr.parameters(), lr=0.02)
    e2 = []
    for ep in range(10):
        perm = torch.randperm(n)
        for t in range(0, n - 16, 16):
            idx = perm[t:t + 16]
            loss = F.mse_loss(lyr(Xt[idx]), Yt[idx, None])
            opt.zero_grad(); loss.backward(); opt.step()
            if t % 320 == 0: lyr.step_gate()
            if ep == 9: e2.append(loss.item())
    ev = lyr.evidence[0].detach().numpy(); share = ev / ev.sum()
    order = np.argsort(-share)
    return (float(np.mean(e2)),
            {names[i]: float(share[i]) for i in order},
            [names[i] for i in range(len(names)) if lyr.G[0, i] > 0])

# ---------------- tools ----------------
_DOWHY_OK = None
def _dowhy_available():
    """Same lazy-singleton optional-dependency pattern as RECALL's
    _vault_engine() below: tried-and-failed is cached (True/False), not
    retried every call."""
    global _DOWHY_OK
    if _DOWHY_OK is None:
        try:
            import dowhy  # noqa: F401
            _DOWHY_OK = True
        except ImportError:
            _DOWHY_OK = False
    return _DOWHY_OK

def make_tools(data, target, interventional):
    def corr(a, b):
        if a not in data or b not in data:
            return f"unknown column(s); columns are {list(data)}"
        r = float(np.corrcoef(data[a], data[b])[0, 1])
        return f"corr({a},{b}) = {r:+.2f} over {len(data[a])} samples."

    def run(vary, clamp):
        if not interventional:
            return "RUN unavailable: this is recorded data, not a rerunnable system. Use STRAT."
        d = demo_world(400, clamp=clamp)
        r = float(np.corrcoef(d[vary], d[target])[0, 1]) if vary in d else float("nan")
        cl = ", ".join(f"{k}={v}" for k, v in clamp.items()) or "nothing"
        return (f"Controlled run: 400 fresh trials varying {vary}, clamping {cl}. "
                f"corr({vary},{target}) = {r:+.2f}; mean {target} {d[target].mean():.2f}.")

    def strat(x, z):
        if x not in data or z not in data:
            return f"unknown column(s); columns are {list(data)}"
        raw = float(np.corrcoef(data[x], data[target])[0, 1])
        qs = np.quantile(data[z], [0, .25, .5, .75, 1.0])
        rs = []
        for i in range(4):
            m = (data[z] >= qs[i]) & (data[z] <= qs[i + 1])
            if m.sum() > 20 and np.std(data[x][m]) > 0:
                rs.append(float(np.corrcoef(data[x][m], data[target][m])[0, 1]))
        within = float(np.mean(rs)) if rs else float("nan")
        return (f"Stratified: raw corr({x},{target}) = {raw:+.2f}; inside quartile bands of {z} "
                f"it is {['%+.2f' % r for r in rs]} (mean {within:+.2f}). "
                f"If the within-band mean collapses, {x}'s link runs through {z}.")

    def adjust(x, zs):
        """Backdoor adjustment (multiple regression) + Cinelli-Hazlett sensitivity, WITH a
        collider/mediator bias audit. Adjusting for a MEDIATOR (on the X->target path) or a
        COLLIDER (a common effect of X and target) INTRODUCES bias -- the 'Table 2 fallacy' --
        so 'control for everything' is wrong for observational/causal data. Data alone cannot
        prove a variable's role (a confounder and a mediator are observationally identical);
        this flags the one danger that IS detectable (a collider: conditioning opens a path
        and RAISES the X-target association) and defers the rest to the DAG / time-order."""
        if x not in data:
            return f"unknown column '{x}'; columns are {list(data)}"
        others = [c for c in data if c not in (x, target)]
        zs = [z for z in zs if z in data and z not in (x, target)]
        n = len(data[target])

        def zsc(a):
            a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-9)
        y = zsc(data[target])

        def fit(cond):                                              # partial corr, t, RV of x | cond
            X = np.column_stack([zsc(data[c]) for c in [x] + cond] + [np.ones(n)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta; dof = n - X.shape[1]
            se = np.sqrt(((resid ** 2).sum() / max(dof, 1)) * np.diag(np.linalg.pinv(X.T @ X)))
            t = float(beta[0] / (se[0] + 1e-12))
            partial = t / np.sqrt(t * t + dof) if dof > 0 else float("nan")
            f = abs(t) / np.sqrt(max(dof, 1))
            return partial, t, 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)   # RV (q=1)

        def pcorr_xy(cond):                                         # partial corr of x & target | cond
            if not cond:
                return float(np.corrcoef(data[x], data[target])[0, 1])
            Z = np.column_stack([zsc(data[c]) for c in cond] + [np.ones(n)])
            rx = zsc(data[x]) - Z @ np.linalg.lstsq(Z, zsc(data[x]), rcond=None)[0]
            ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            return float(np.corrcoef(rx, ry)[0, 1])

        partial, t, rv = fit(zs)
        raw = float(np.corrcoef(data[x], data[target])[0, 1])
        zst = ", ".join(zs) if zs else "nothing"
        msg = (f"ADJUST: effect of {x} on {target} controlling for [{zst}] (n={n}). "
               f"raw corr {raw:+.2f} -> adjusted partial corr {partial:+.2f} (t={t:+.1f}). "
               f"Robustness value RV={rv:.2f}: an unmeasured confounder would need to explain "
               f">={rv*100:.0f}% of the residual variance of BOTH {x} and {target} to null it. "
               f"RV<0.10 = fragile; higher = more robust to hidden confounding.")

        r0 = abs(raw); collide, explain = [], []                    # per-variable bias audit
        for z in zs:
            rz = pcorr_xy([z])
            opened = abs(rz) - r0 > 0.10                            # conditioning grows the association
            flipped = rz * raw < 0 and abs(rz) > 0.15              # ...or reverses its sign
            if opened or flipped:
                collide.append((z, rz))
            elif r0 - abs(rz) > 0.15:                               # z soaks up much of x<->target
                explain.append((z, rz))
        if collide:
            lst = ", ".join(f"{z} (corr {raw:+.2f}->{rz:+.2f})" for z, rz in collide)
            msg += (f"\n[BIAS-AUDIT] conditioning on {lst} sharply changes the {x}-{target} link "
                    f"(opens or reverses it) -- a COLLIDER signature (a common effect of both). If "
                    f"{x} and {target} both cause it, do NOT adjust; that manufactures a spurious "
                    f"effect. (A strong confounder can also flip the sign -- confirm via the DAG.)")
        if explain:
            lst = ", ".join(f"{z} (corr {raw:+.2f}->{rz:+.2f})" for z, rz in explain)
            msg += (f"\n[BIAS-AUDIT] {lst} soak(s) up much of the link. Correct to adjust ONLY if a "
                    f"CONFOUNDER (a prior common cause); if a MEDIATOR (on the {x}->{target} path) "
                    f"or measured AFTER {x}, adjusting ERASES the real effect. Data can't tell "
                    f"them apart -- decide by the DAG / measurement time-order.")
        if zs and not collide and not explain:
            msg += ("\n[BIAS-AUDIT] no collider signature in the set (still confirm none are "
                    "mediators / post-exposure via the DAG).")

        omitted = [c for c in others if c not in zs]
        if omitted:                                                # show the full-set result as a reference
            fp, ft, frv = fit(others)
            flip = (abs(fp) < 0.10) != (abs(partial) < 0.10)
            msg += (f"\n[FULL SET] controlling for ALL others {omitted}: {x}'s partial is {fp:+.2f} "
                    f"(RV={frv:.2f})" + (" -- FLIPS vs your subset." if flip else ", consistent.") +
                    " Trust this ONLY if none of those are mediators/colliders (see BIAS-AUDIT); for "
                    "observational data adjust for confounders, not 'everything'.")
        return msg

    def interact(x, z):
        """Tests the explicit PRODUCT term x*z as a candidate driver -- something CORR/ADJUST
        structurally cannot see, by construction: a linear regression's fitted surface is
        additive in its inputs, so a pure interaction (outcome driven by x*z, not x or z alone)
        is invisible to it no matter how strong the real effect is. Verified directly: for
        independent mean-zero x,z, corr(x,target) and corr(z,target) can both be ~0 while
        corr(x*z,target) is ~1 -- the same odd/even symmetry argument as the point-group
        selection rule (see github.com/tritsystem/symmetry-selection-rule): a purely
        additive/linear method has no way to represent an even-order term until something
        breaks that structure. Report only, not a full backdoor adjustment -- confirm x,z
        aren't downstream of target before trusting this as causal, same caveat ADJUST's
        bias-audit already carries."""
        if x not in data or z not in data:
            return f"unknown column(s); columns are {list(data)}"
        rx = float(np.corrcoef(data[x], data[target])[0, 1])
        rz = float(np.corrcoef(data[z], data[target])[0, 1])
        product = np.asarray(data[x], float) * np.asarray(data[z], float)
        rxz = float(np.corrcoef(product, data[target])[0, 1])
        msg = (f"INTERACT: {x}*{z} vs {target} (n={len(data[target])}). "
               f"Individually: corr({x},{target})={rx:+.2f}, corr({z},{target})={rz:+.2f}. "
               f"Product term: corr({x}*{z},{target})={rxz:+.2f}.")
        if abs(rxz) - max(abs(rx), abs(rz)) > 0.15:
            msg += (f"\n[FOUND] the product explains far more than either variable alone -- "
                    f"a real candidate INTERACTION driver, invisible to CORR/ADJUST's linear-"
                    f"only view. Not yet a full causal claim: confirm neither {x} nor {z} is "
                    f"downstream of {target} before trusting this.")
        else:
            msg += "\nNo meaningful interaction signal beyond what the individual variables already show."
        return msg

    def refute(x, zs):
        """DoWhy-backed refutation testing -- a SECOND, independently-derived
        robustness check on the SAME x|zs hypothesis ADJUST already tested,
        using an established causal-inference library's estimator plus three
        real perturbation tests, instead of this file's own hand-rolled
        Cinelli-Hazlett RV. Does NOT resolve confounder-vs-mediator ambiguity
        (see ADJUST's bias-audit for that, still a DAG/time-order question
        data alone can't answer) -- it only asks whether the NUMERICAL
        estimate survives real perturbation:
          placebo treatment   -- effect should COLLAPSE toward 0 (confirms
                                  the estimate isn't a fitting-procedure
                                  artifact; treatment is randomly permuted)
          random common cause -- effect should barely CHANGE (a real,
                                  already-adjusted effect shouldn't move much
                                  from one more irrelevant confounder)
          data subset (80%)   -- effect should barely CHANGE (not driven by
                                  a handful of influential points)
        Stochastic (permutation/resampling-based) -- re-running can shift the
        exact numbers slightly; the qualitative collapsed/stable read is what
        matters, not the third decimal place."""
        if not _dowhy_available():
            return "REFUTE unavailable: pip install dowhy to enable (not a core dependency)."
        if x not in data:
            return f"unknown column '{x}'; columns are {list(data)}"
        zs = [z for z in zs if z in data and z not in (x, target)]
        import pandas as pd
        from dowhy import CausalModel
        df = pd.DataFrame({c: data[c] for c in data})
        try:
            cm = CausalModel(data=df, treatment=x, outcome=target, common_causes=zs or None)
            identified = cm.identify_effect(proceed_when_unidentifiable=True)
            est = cm.estimate_effect(identified, method_name="backdoor.linear_regression")
            orig = float(est.value)
            placebo = cm.refute_estimate(identified, est, method_name="placebo_treatment_refuter", placebo_type="permute")
            rcc = cm.refute_estimate(identified, est, method_name="random_common_cause")
            subset = cm.refute_estimate(identified, est, method_name="data_subset_refuter", subset_fraction=0.8)
        except Exception as e:
            return f"REFUTE error (DoWhy could not fit/refute this model): {e}"

        zst = ", ".join(zs) if zs else "nothing"
        p_new, rcc_new, subset_new = float(placebo.new_effect), float(rcc.new_effect), float(subset.new_effect)
        scale = abs(orig) if abs(orig) > 1e-9 else 1e-9
        collapsed = abs(p_new) < 0.1 * scale
        stable_rcc = abs(rcc_new - orig) < 0.2 * scale
        stable_subset = abs(subset_new - orig) < 0.2 * scale
        n_pass = sum([collapsed, stable_rcc, stable_subset])

        msg = (f"REFUTE (DoWhy): effect of {x} on {target} controlling for [{zst}] -- "
               f"original estimate {orig:+.3f} (backdoor.linear_regression, independent of ADJUST's own fit).\n"
               f"  placebo treatment: new effect {p_new:+.3f} -- "
               + ("collapsed toward 0, as expected for a real effect." if collapsed
                  else "did NOT collapse -- suspicious; the estimate may reflect the fitting procedure, not a real relationship.") + "\n"
               f"  random common cause: new effect {rcc_new:+.3f} -- "
               + ("stable." if stable_rcc else "changed notably -- sensitive to an irrelevant confounder, a fragility signal.") + "\n"
               f"  data subset (80%): new effect {subset_new:+.3f} -- "
               + ("stable." if stable_subset else "changed notably -- may be driven by a subset of influential points.") + "\n"
               f"[{n_pass}/3 refutation checks consistent with a real, stable effect]")
        return msg

    return corr, run, strat, adjust, interact, refute

# ---------------- RECALL: optional semantic search over an external vault (OBSERVE) ----------------
_OBSERVE_PATH = os.environ.get("METHODLM_OBSERVE_PATH")     # optional: path to an OBSERVE checkout
_VAULT_INDEX_DIR = os.environ.get("METHODLM_VAULT_INDEX")   # optional: path to a pre-built OBSERVE index dir
_VAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VAULT_ENGINE = None   # lazy singleton: loading the embedding model takes real seconds, do it once

def _vault_engine():
    """RECALL is fully optional, same pattern as the tritkit second witness above: unset
    either env var and RECALL just reports itself unavailable rather than erroring. False
    (not None) once tried-and-failed, so a missing index doesn't retry every call."""
    global _VAULT_ENGINE
    if _VAULT_ENGINE is not None:
        return _VAULT_ENGINE
    if not _OBSERVE_PATH or not _VAULT_INDEX_DIR:
        _VAULT_ENGINE = False
        return _VAULT_ENGINE
    if _OBSERVE_PATH not in sys.path:
        sys.path.insert(0, _OBSERVE_PATH)
    try:
        from search_engine import SearchEngine
        eng = SearchEngine()
        eng.load_blocking(_VAULT_INDEX_DIR, _VAULT_MODEL)
        _VAULT_ENGINE = eng if eng.ready else False
    except Exception as e:
        print(f"[warn] RECALL vault search unavailable: {e}")
        _VAULT_ENGINE = False
    return _VAULT_ENGINE

def recall(query, k=5):
    """Real semantic search (MiniLM embeddings, not a keyword grep) over an external OBSERVE-
    indexed corpus (e.g. a notes vault) -- 'did we already investigate/decide this before'
    memory. NOT a causal test: does not count toward nrun and cannot satisfy the pre-FINAL
    test requirement, same as ATTR/CORR. Fully optional -- see METHODLM_OBSERVE_PATH /
    METHODLM_VAULT_INDEX above."""
    eng = _vault_engine()
    if not eng:
        return ("RECALL unavailable: set METHODLM_OBSERVE_PATH (a checkout of OBSERVE, "
                "https://github.com/tritsystem/observe-api) and METHODLM_VAULT_INDEX (a "
                "pre-built semantic index directory) to enable it.")
    hits = eng.search(query, k=k)
    if not hits:
        return f"RECALL: no vault matches for '{query}'."
    lines = [f"RECALL: top {len(hits)} vault match(es) for '{query}':"]
    for h in hits:
        lines.append(f"  [{h['score']:.2f}] {os.path.basename(h['path'])}: {h['preview'][:160].strip()}")
    return "\n".join(lines)

# ---------------- REASON: the copilot loop ----------------
def build_system(cols, target, interventional):
    names = ", ".join(cols)
    a, b = [c for c in cols if c != target][:2]
    runline = (f"  RUN: vary={a}, clamp={{{b}:50}}    (true controlled experiment; clamp value is a NUMBER)"
               if interventional else
               f"  STRAT: {a},{b}          (quick check: corr({a},{target}) inside bands of {b})")
    return textwrap.dedent(f"""\
        You are a research-methodology copilot running a real instrument by the gbranaa-hue
        method. A correlation is never a cause; suspect the boring explanation (artifact,
        confound, the instrument itself) first; trust only tests that could have failed.

        Data columns (EXACT names): {names}. Target: {target}.
        Tools -- end each reply with exactly one tool line:
          CORR: {a},{target}
        {runline}
          ADJUST: {a} | <other confounder columns>   (backdoor adjustment + sensitivity: effect
                                     of {a} on {target} controlling for the listed confounders,
                                     with a robustness value + a collider/mediator bias audit)
          INTERACT: {a},<other column>   (tests the PRODUCT of two columns as a driver --
                                     CORR/ADJUST are linear and CANNOT see a pure interaction
                                     effect, even a perfect one, no matter how strong; if every
                                     candidate looks fragile alone under ADJUST, try this before
                                     concluding "no driver")
          REFUTE: {a} | <same confounder columns as your ADJUST call>   (SECOND, independent
                                     robustness check on a candidate ADJUST already found
                                     promising -- DoWhy's own estimator + 3 real perturbation
                                     tests: placebo treatment should COLLAPSE the effect,
                                     random common cause and data-subset should barely change
                                     it. Use AFTER ADJUST on the same candidate, not instead of
                                     it -- ADJUST's bias-audit and REFUTE's perturbation checks
                                     catch different failure modes.)
          ATTR:                    (the learned ternary model's evidence per column)
          RECALL: <free-text query>   (semantic search over past vault notes/decisions for
                                     relevant prior work -- memory, NOT a causal test; use it
                                     ONCE, if at all, to check "did we already find this" before
                                     re-deriving -- it cannot be used a second time; unavailable
                                     unless METHODLM_OBSERVE_PATH/METHODLM_VAULT_INDEX are set)
        HOW TO FIND THE DRIVER: for a candidate X, run 'ADJUST: X | <the other candidate
        columns>'. The tool shows the full-set result AND a [BIAS-AUDIT]. HEED IT: drop any
        variable flagged as a COLLIDER (conditioning on it manufactures a fake effect), and do
        NOT adjust for a MEDIATOR (a variable on the X->{target} path or measured after X) --
        'adjust for everything' is the Table 2 fallacy. Condition on prior common causes only.
        If X's adjusted partial corr stays large with a high robustness value, X drives
        {target}; if it COLLAPSES toward 0 (or RV < 0.10), X is a confounded bystander.
        CRITICAL: the driver is the candidate that SURVIVES full adjustment -- it is NEVER
        the variable you controlled for. Do not name a conditioning/control variable as the
        cause; that is backwards. Strategy: ADJUST the tempting/decoy variable
        first; if it collapses, ADJUST the other strong candidate to confirm the real
        driver; once one survives, REFUTE it on the same X | <same confounders> as a second,
        independently-derived check before concluding (if REFUTE is unavailable it will say so
        -- don't let that block FINAL, ADJUST's own robustness value already counts as a test).
        Before any ADJUST/STRAT/RUN/INTERACT/REFUTE, include a 'PREREGISTER:' line in the SAME
        reply naming the same X you test and what result confirms vs disconfirms. Comparing two
        correlations is NOT a test. You MUST run at least one ADJUST/STRAT/RUN/INTERACT (never
        only CORR) before any FINAL. Your FINAL must name the variable (or product of two
        variables) whose effect SURVIVED as the driver (or say none did). Be brief.""")

def ask(system, messages, n=230):
    return backend().generate(system, messages, n)

def vanilla_answer(question, n=200):
    """The SAME model, no method and no tools -- a plain analyst. The control racer."""
    sysp = ("You are a helpful data analyst. Answer the question directly and concisely: "
            "give your conclusion and a recommendation.")
    return ask(sysp, [{"role": "user", "content": question}], n)

def investigate(name, data, target, question, interventional, answer_key=None, ingest_report=None):
    ledger = os.path.join(HERE, f"ledger_{name}.txt")
    led = open(ledger, "w", encoding="utf-8")
    def out(s):
        led.write(s + "\n"); led.flush()
        try: print(s, flush=True)
        except UnicodeEncodeError: print(s.encode("ascii", "replace").decode(), flush=True)

    if ingest_report:
        out(f"[ingest]\n{ingest_report}\n")
    t0 = time.time()
    try:
        nmse, share, gate = train_readout(data, target)
        top = ", ".join(f"{k} {v*100:.0f}%" for k, v in list(share.items())[:4])
        out(f"[compute] ternary readout NMSE {nmse:.2f} | evidence: {top} | gate: {gate}")
    except ImportError:
        nmse, share, gate = None, {}, []
        out("[compute] ternary second witness skipped (set METHODLM_TRITKIT + install torch to enable)")
    corr, run, strat, adjust, interact, refute = make_tools(data, target, interventional)
    system = build_system(list(data), target, interventional)

    msgs = [{"role": "user", "content": question}]
    pre = nrun = n_recall = 0
    verdict = "(no verdict — ran out of turns)"
    seen = {}          # loop guard: signatures of tests already run
    stuck = 0          # consecutive turns producing no usable tool call
    for turn in range(1, 10):
        raw = ask(system, msgs)
        # ENFORCE ONE ACTION PER TURN: keep text up to and including the first tool
        # directive (or FINAL); drop anything the model dumped after it.
        cut = None
        for m in re.finditer(r"^\s*(CORR|RUN|STRAT|ADJUST|INTERACT|REFUTE|ATTR|RECALL|FINAL)\b.*$", raw, re.MULTILINE | re.IGNORECASE):
            cut = m.end(); break
        reply = raw[:cut] if cut else raw
        out(f"\n--- copilot turn {turn} ---\n{reply}")
        msgs.append({"role": "assistant", "content": reply})
        if "PREREGISTER:" in reply: pre += 1
        m_run = re.search(r"RUN:\s*vary=(\w+),\s*clamp=\{([^}]*)\}", reply)
        m_adj = re.search(r"ADJUST:\s*(\w+)\s*\|\s*([\w,\s]*)", reply)
        m_str = re.search(r"STRAT:\s*(\w+)\s*,\s*(\w+)", reply)
        m_cor = re.search(r"CORR:\s*(\w+)\s*,\s*(\w+)", reply)
        m_attr = re.search(r"\bATTR:", reply)
        m_rec = re.search(r"RECALL:\s*(.+)$", reply, re.MULTILINE)
        m_int = re.search(r"INTERACT:\s*(\w+)\s*,\s*(\w+)", reply)
        m_ref = re.search(r"REFUTE:\s*(\w+)\s*\|\s*([\w,\s]*)", reply)
        # FINAL honored ONLY when it's not a hedge AND a real test has run
        if re.search(r"\bfinal\s*:", reply, re.IGNORECASE) and not (m_run or m_adj or m_str or m_cor or m_attr or m_int or m_ref):
            if nrun < 1:
                out("[TOOL] REFUSED: comparing correlations is not a test. Run one STRAT or "
                    "RUN before concluding.")
                msgs.append({"role": "user", "content": "REFUSED: run at least one STRAT or "
                             "RUN test before FINAL."})
                continue
            verdict = re.sub(r"^.*?final\s*:", "", reply, flags=re.IGNORECASE | re.DOTALL).strip()
            out("\n[done] verdict reached."); break
        if m_run:
            clamp = {k: float(v) for k, v in re.findall(r"(\w+)\s*:\s*([-\d.]+)", m_run.group(2))}
            res = ("REFUSED: PREREGISTER in the same reply first." if "PREREGISTER:" not in reply
                   else "Clamp needs numbers, e.g. clamp={humidity:50}" if not clamp
                   else run(m_run.group(1), clamp)); nrun += m_run and bool(clamp) and "PREREGISTER:" in reply
        elif m_adj:
            zs = [z.strip() for z in m_adj.group(2).split(",") if z.strip()]
            res = ("REFUSED: PREREGISTER in the same reply first." if "PREREGISTER:" not in reply
                   else adjust(m_adj.group(1), zs)); nrun += "PREREGISTER:" in reply
        elif m_str:
            res = ("REFUSED: PREREGISTER in the same reply first." if "PREREGISTER:" not in reply
                   else strat(m_str.group(1), m_str.group(2))); nrun += bool(m_str) and "PREREGISTER:" in reply
        elif m_cor:
            res = corr(m_cor.group(1), m_cor.group(2))
        elif m_int:
            res = ("REFUSED: PREREGISTER in the same reply first." if "PREREGISTER:" not in reply
                   else interact(m_int.group(1), m_int.group(2))); nrun += "PREREGISTER:" in reply
        elif m_ref:
            zs = [z.strip() for z in m_ref.group(2).split(",") if z.strip()]
            res = ("REFUSED: PREREGISTER in the same reply first." if "PREREGISTER:" not in reply
                   else refute(m_ref.group(1), zs)); nrun += "PREREGISTER:" in reply
        elif re.search(r"\bATTR:", reply):
            if nmse is None:
                res = "Ternary second witness unavailable (install torch + set METHODLM_TRITKIT)."
            else:
                res = (f"Ternary readout (NMSE {nmse:.2f}) evidence share: "
                       + ", ".join(f"{k}: {v*100:.0f}%" for k, v in share.items())
                       + f". Gate connects: {', '.join(gate)}.")
        elif m_rec:
            # SINGLE-USE GUARD: a weak model echoing the [TOOL] result text back as its own
            # next RECALL: line was a real, observed failure mode (spiraling into nested
            # self-quoting, e.g. "RECALL: top 5 vault match(es) for 'top 5 vault match(es)...'"
            # until it lost the ability to emit any tool line at all). RECALL's job is a
            # one-time "did we already do this" memory check, not a repeatable action, so cap
            # it at one real call per investigation regardless of query content.
            n_recall += 1
            if n_recall > 1:
                res = ("REFUSED: RECALL already used once this investigation -- it is a one-time "
                       "memory check, not repeatable. PREREGISTER and run a real causal test "
                       "(ADJUST/STRAT/RUN) now.")
            else:
                query = m_rec.group(1).strip()[:200]   # cap length: reject echoed-tool-output blowup
                res = recall(query)
        else:
            res = "No tool recognized. Use CORR:, STRAT:/RUN:, ADJUST:, INTERACT:, REFUTE:, ATTR:, RECALL:, or FINAL:."
        # LOOP GUARD: a weak model can re-run the same test forever. If a real test
        # repeats, don't re-run it -- nudge to conclude; force a stop on a 2nd repeat.
        # Tagged with the tool name: REFUTE is DESIGNED to be re-run on the exact same
        # (x, zs) an ADJUST call already used (a second, independent robustness check
        # on the same candidate) -- an untagged sig would wrongly flag that as a repeat.
        _sig_src = [("RUN", m_run), ("ADJUST", m_adj), ("STRAT", m_str), ("CORR", m_cor),
                    ("INTERACT", m_int), ("REFUTE", m_ref)]
        sig = next(((tag, g.groups()) for tag, g in _sig_src if g), None)
        if sig and not str(res).startswith(("REFUSED", "Clamp")):
            if sig in seen:
                seen[sig] += 1
                # nudge toward the NEXT untested candidate (not FINAL) -- a collapsed test
                # means that variable is a bystander, not that the job is done.
                tested = ", ".join(sorted({s[1][0] for s in seen})) or "none"
                untested = [c for c in data if c not in (target,) and c not in {s[1][0] for s in seen}]
                res = (f"{res}\n[REPEAT: you already ran this. A collapsed effect (RV<0.10) means "
                       f"that variable is a BYSTANDER, not the answer. You have tested: {tested}. "
                       f"Now run ADJUST on a DIFFERENT untested candidate ({', '.join(untested) or 'none left'}) "
                       "to find the real driver. Reply FINAL only once a variable SURVIVES (high RV).]")
                if seen[sig] >= 3:
                    out(f"[TOOL] {res}")
                    verdict = "(loop-guard: model repeated the same test without concluding)"
                    out("\n[done] loop-guard stopped a repeat loop."); break
            else:
                seen[sig] = 1
        out(f"[TOOL] {res}")
        # NO-PROGRESS guard: a model that can't emit tool syntax loops uselessly.
        if str(res).startswith(("No tool recognized", "REFUSED")):
            stuck += 1
            if stuck >= 3:
                verdict = "(model could not drive the tool protocol)"
                out("\n[done] no-progress guard: model never produced a usable test."); break
        else:
            stuck = 0
        msgs.append({"role": "user", "content": res})
    if answer_key:
        out(f"\n[ANSWER KEY -- the instrument was never told] {answer_key}")
    out(f"\n[ledger] preregistrations {pre}, registered tests {nrun}, "
        f"{time.time()-t0:.0f}s -> {ledger}")
    led.close()
    return {"verdict": verdict, "pre": pre, "nrun": nrun,
            "nmse": nmse, "gate": gate, "share": share}


def finish_race(question, res, answer_key=None):
    """Same question, two racers: MethodLM (method + tools + tests) vs plain LLM."""
    print("\n" + "=" * 62)
    print("HEAD-TO-HEAD -- same question, two racers")
    print("=" * 62)
    v = vanilla_answer(question)
    tested = f"{res['nrun']} test(s), {res['pre']} pre-reg" if res["nrun"] else "NO TEST RUN"
    print(f"\n[ vanilla 3B | no method, no tools ]\n  {v}\n")
    print(f"[ MethodLM | method + tools | {tested} ]\n  {res['verdict']}\n")
    if answer_key:
        print(f"[ answer key ] {answer_key}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--diabetes", action="store_true")
    ap.add_argument("--data", help="any file: csv/tsv/json/jsonl/parquet/sqlite/npz/xlsx")
    ap.add_argument("--csv", help="alias for --data")
    ap.add_argument("--folder", help="labelled folder of text/image files")
    ap.add_argument("--target")
    ap.add_argument("--table", help="sqlite table name (optional)")
    ap.add_argument("--query", help="sqlite SQL query (optional)")
    ap.add_argument("--race", action="store_true", help="also run a plain-LLM racer on the same question")
    ap.add_argument("--model", default="local", help="reasoning backend: local (Qwen-3B) | claude")
    args = ap.parse_args()
    global BACKEND
    BACKEND = methodlm_models.get_model(args.model, HERE)
    print(f"[methodlm] reasoning backend: {BACKEND.label}")
    if args.demo:
        data = demo_world()
        r = float(np.corrcoef(data['temperature'], data['error'])[0, 1])
        q = (f"Our sensor's error correlates with temperature (r={r:+.2f}) in 2000 logged "
             "samples; the team wants to install cooling. Find what actually drives the "
             "error before we spend the money.")
        key = "error = f(humidity); temperature only co-rises with humidity via season."
        res = investigate("demo", data, "error", q, True, key)
        if args.race: finish_race(q, res, key)
    elif args.diabetes:
        data = load_diabetes()
        r = float(np.corrcoef(data['bmi'], data['progression'])[0, 1])
        q = (f"In 442 real diabetes patients, bmi correlates with one-year disease "
             f"progression (r={r:+.2f}). The clinic wants to fund a weight-loss-only "
             "program. Before they do: is bmi's link robust, or does it run through blood "
             "serum markers like ltg? You cannot rerun patients; use STRAT.")
        res = investigate("diabetes", data, "progression", q, False)
        if args.race: finish_race(q, res)
    elif args.folder and args.target:
        from methodlm_io import load_folder, featurize, format_report
        raw, notes = load_folder(args.folder)
        data, rep = featurize(raw, args.target)
        report = format_report(notes, rep, args.target)
        cols = [c for c in data if c != args.target]
        q = (f"Investigate what actually drives {args.target} across these files "
             f"(features: {', '.join(cols[:12])}). Do not trust raw correlations.")
        res = investigate(os.path.basename(args.folder.rstrip('/\\')) or "folder",
                          data, args.target, q, False, ingest_report=report)
        if args.race: finish_race(q, res)
    elif args.data or args.csv:
        from methodlm_io import validate, load_any, featurize, format_report
        path = args.data or args.csv
        v = validate(path, args.target, table=args.table, query=args.query)
        if not v["ok"]:
            print("MethodLM cannot run yet:")
            for e in v["errors"]:
                print(f"  x {e}")
            if v["suggestions"]:
                print(f"  -> try: {', '.join(map(str, v['suggestions']))}")
            if v["info"]:
                print("  columns available: "
                      + ", ".join(f"{c['name']} [{c['kind']}]" for c in v["info"]["columns"]))
            return
        raw, notes = load_any(path, table=args.table, query=args.query)
        data, rep = featurize(raw, args.target)
        report = format_report(notes, rep, args.target)
        cols = [c for c in data if c != args.target]
        q = (f"Investigate what actually drives {args.target} in this recorded dataset "
             f"(columns: {', '.join(cols[:12])}{'...' if len(cols) > 12 else ''}). "
             "Do not trust raw correlations.")
        name = os.path.splitext(os.path.basename(path))[0]
        res = investigate(name, data, args.target, q, False, ingest_report=report)
        if args.race: finish_race(q, res)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
