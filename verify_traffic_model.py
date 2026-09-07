"""
verify_traffic_model.py -- prove the two traffic modes behave correctly.

Checks:
  1. SMOOTH mode is unchanged (small, gentle steps -- no big jumps).
  2. BURSTY mode produces occasional LARGE sudden jumps that smooth
     never does.
  3. BURSTY does not just make everything busier -- the mean load stays
     close to smooth. Bursts are occasional events, not a flood.
  4. The twin, reading late, is MORE WRONG under bursty traffic -- which
     is the whole point: bursty is a harder, fairer test of the gate.
"""
import numpy as np
import importlib
import config, traffic, isl, ground

PASS=0; FAIL=0
def check(name, cond, detail=""):
    global PASS,FAIL
    if cond: PASS+=1
    else: FAIL+=1
    print(f"  [{'PASS' if cond else '**FAIL**'}] {name}")
    if detail: print(f"          {detail}")

def build(model, seed=0):
    config.TRAFFIC_MODEL = model
    importlib.reload(traffic)
    config.TRAFFIC_MODEL = model
    plan=isl.build_isl_plan(); gs=ground.build_ground_stations()
    rng=np.random.default_rng(seed)
    return traffic.BackgroundTraffic(plan,[g['name'] for g in gs],rng)

def link_series(bg, ticks=2000):
    key=list(bg.load.keys())[0]
    kind='gsl' if isinstance(key[0],str) else 'isl'
    s=[]
    for _ in range(ticks):
        bg.step(1.0)
        s.append(bg.utilisation(key[0],key[1],kind))
    return np.array(s)

def main():
    print("="*66)
    print("VERIFICATION: traffic model (smooth vs bursty)")
    print("="*66)

    print("\n=== 1 & 2: smooth is gentle, bursty has big jumps ===")
    s_smooth = link_series(build('smooth'))
    s_bursty = link_series(build('bursty'))
    jump_smooth = np.abs(np.diff(s_smooth)).max()
    jump_bursty = np.abs(np.diff(s_bursty)).max()
    print(f"  biggest single-tick jump -- smooth: {jump_smooth:.3f}, bursty: {jump_bursty:.3f}")
    check("smooth never makes large sudden jumps", jump_smooth < 0.25,
          "gentle drift only")
    check("bursty produces jumps smooth never does", jump_bursty > 2*jump_smooth,
          "occasional real spikes, as intended")

    print("\n=== 3: bursty doesn't just flood the network ===")
    print(f"  mean load -- smooth: {s_smooth.mean():.3f}, bursty: {s_bursty.mean():.3f}")
    check("bursty mean load stays close to smooth",
          abs(s_bursty.mean()-s_smooth.mean()) < 0.1,
          "bursts are occasional events, not a permanent load increase")

    print("\n=== 4: the twin is more wrong under bursty traffic ===")
    print("  (this is the POINT: bursty is a harder, fairer test)")
    from twin import DigitalTwin
    def twin_error(model, seed=1, ticks=1500):
        bg=build(model, seed)
        config.TRAFFIC_MODEL=model
        import twin as twmod; importlib.reload(twmod)
        tw=twmod.DigitalTwin(delay_s=5.0, interval_s=5.0)
        errs=[]
        t=0.0
        for _ in range(ticks):
            bg.step(1.0); tw.observe(bg,t)
            if t>60:  # after warmup
                for k,true in list(bg.snapshot().items())[:30]:
                    bel = tw._belief.get(k, 0.45) if tw._belief else 0.45
                    errs.append(abs(true-bel))
            t+=1.0
        return np.mean(errs)
    e_smooth=twin_error('smooth')
    e_bursty=twin_error('bursty')
    print(f"  mean twin error -- smooth: {e_smooth:.4f}, bursty: {e_bursty:.4f}")
    check("twin is more wrong under bursty traffic", e_bursty > e_smooth,
          "bursts catch the stale twin out -- exactly what stresses the gate")

    print("\n"+"="*66)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("="*66)
    if FAIL==0:
        print("\nBursty mode is a genuine, harder test: same average load, but")
        print("occasional real spikes that catch the stale twin out. Smooth")
        print("mode is unchanged. Compare the two to show robustness.")
    return FAIL

if __name__=="__main__":
    import sys; sys.exit(main())
