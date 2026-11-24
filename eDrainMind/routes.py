"""Smart Emergency Route Recommendation for eDrainMind AI.

Everything here uses REAL data - nothing is invented:
  * Roads / alternative routes ... OpenStreetMap road network through the free OSRM routing service
  * Ground height ................ Open-Meteo elevation (SRTM)        -> low-lying road stretches
  * Waterways, underpasses, bridges, road surface, OSM closures ... OpenStreetMap (Overpass)
  * Flood risk of nearby drains .. this app's own models (rain, water level, blockage, forecast)
  * Road closures / flooded roads  .. incidents that YOU report on the map (saved in data/incidents.json)
Drain readings are simulated until real sensors are connected - the screen says so.

Pipeline:  candidates -> sample every ~100 m -> flood risk per sample -> Safe/Moderate/High/Critical
           -> route risk score -> weighted route score (distance 30 / flood 40 / condition 20 / time 10, configurable) -> recommendation.
"""
import math, time, datetime as dt, concurrent.futures as cf
import numpy as np
import geo

OSRM = ["https://routing.openstreetmap.de/routed-car/route/v1/driving/", "https://router.project-osrm.org/route/v1/driving/"]
K = 111320.0
CLASSES = ["Safe", "Moderate", "High", "Critical"]; CUTS = [25, 50, 75]                    # risk 0-100 -> class
DEFAULT_W = dict(distance=30, flood=40, condition=20, time=10)                              # starting weights (configurable, not "optimal")
EMERGENCY_W = dict(distance=10, flood=40, condition=10, time=40)                            # ambulance / rescue: time matters more
SLOW = dict(normal=[1.0, 1.15, 1.6, 2.5], emergency=[1.0, 1.05, 1.3, 2.0])                  # assumed speed penalty per class (estimate)
HOSP_BBOX = (10.70, 76.55, 10.86, 76.76)

# ---------------------------------------------------------------- geometry helpers
def _xy(coords, lat0, lon0):
    a = np.asarray(coords, float).reshape(-1, 2); return np.c_[(a[:, 1] - lon0) * K * math.cos(math.radians(lat0)), (a[:, 0] - lat0) * K]

def _segdist(P, A, B):
    """Distance from each point in P (n,2) to the nearest of the segments A->B (m,2). Returns (distance, segment index)."""
    d = B - A; l2 = (d ** 2).sum(1) + 1e-9
    t = np.clip(((P[:, None, :] - A[None]) * d[None]).sum(2) / l2[None], 0, 1); pr = A[None] + t[..., None] * d[None]
    dist = np.hypot(P[:, None, 0] - pr[..., 0], P[:, None, 1] - pr[..., 1]); return dist.min(1), dist.argmin(1)

def _resample(coords, lat0, lon0, step):
    xy = _xy(coords, lat0, lon0); cum = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]; total = max(float(cum[-1]), 1.0)
    n = max(3, int(round(total / step)) + 1); s = np.linspace(0, total, n); arr = np.asarray(coords, float)
    return dict(lat=np.interp(s, cum, arr[:, 0]), lon=np.interp(s, cum, arr[:, 1]), s=s, total=total, step=total / (n - 1),
                xy=np.c_[np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])])

# ---------------------------------------------------------------- real routing (OSRM)
def osrm(points, alternatives=False):
    path = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points); last = None
    for base in OSRM:
        try:
            j = geo._json(f"{base}{path}?overview=full&geometries=geojson&steps=false&alternatives={'true' if alternatives else 'false'}", timeout=25)
            if j.get("code") == "Ok" and j.get("routes"):
                return [dict(coords=[[c[1], c[0]] for c in r["geometry"]["coordinates"]], distance=float(r["distance"]), duration=float(r["duration"])) for r in j["routes"]]
            last = RuntimeError(j.get("message") or j.get("code") or "no route")
        except Exception as e: last = e
    raise RuntimeError("Road routing service not reachable or no road route found (" + str(last)[:90] + ")")

def _overlap(r1, r2, lat0, lon0):
    P = _resample(r1["coords"], lat0, lon0, 100)["xy"]; q = _xy(r2["coords"], lat0, lon0); d, _ = _segdist(P, q[:-1], q[1:]); return float((d < 40).mean())

def _loops(r, lat0, lon0):
    """True if the route doubles back on itself (a detour spur made by a via-point)."""
    P = _resample(r["coords"], lat0, lon0, 100)["xy"]; n = len(P); D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    far = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > 8; return float(((D < 25) & far).any(1).mean()) > .12

def candidates(a, b):
    """The routes OSRM offers (up to 3) + routes forced through side waypoints so there is a real detour when the main road is flooded."""
    lat0, lon0 = a; base = osrm([a, b], True); L = geo.metres(a, b); cs = math.cos(math.radians(a[0]))
    dx, dy = (b[1] - a[1]) * K * cs, (b[0] - a[0]) * K; n = math.hypot(dx, dy) or 1; pe, pn = -dy / n, dx / n; mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    vias = [(mid[0] + pn * o / K, mid[1] + pe * o / (K * cs)) for o in [s * float(np.clip(f * L, 500, 5000)) for f in (.3, .6) for s in (1, -1)]]
    def one(w):
        try: return osrm([a, w, b])[0]
        except Exception: return None
    with cf.ThreadPoolExecutor(4) as ex: extra = [r for r in ex.map(one, vias) if r]
    short = min(r["distance"] for r in base); keep = []
    for r in sorted(base + extra, key=lambda r: r["duration"]):
        if r["distance"] > 2.0 * short + 500 or r["distance"] > short + 15000: continue
        if any(_overlap(r, k, lat0, lon0) > .9 and _overlap(k, r, lat0, lon0) > .9 for k in keep): continue
        if r not in base and _loops(r, lat0, lon0): continue
        keep.append(r)
        if len(keep) == 5: break
    return keep

# ---------------------------------------------------------------- real road / water context (OpenStreetMap)
SURF_BAD = {"unpaved", "gravel", "dirt", "sand", "ground", "mud", "grass", "compacted", "fine_gravel", "earth", "pebblestone", "woodchips"}
SURF_OK = {"asphalt", "paved", "concrete", "concrete:plates", "concrete:lanes", "paving_stones", "sett", "bricks", "cobblestone"}
HWY = {"motorway": 0, "trunk": 0, "primary": 0, "secondary": .05, "tertiary": .1, "residential": .15, "living_street": .2, "unclassified": .2, "service": .3, "track": .7, "path": .8}
SMOOTH = {"excellent": 0, "good": 0, "intermediate": .25, "bad": .55, "very_bad": .8, "horrible": .9, "very_horrible": 1, "impassable": 1}

def condition(tags):
    """Road condition 0 (good) - 1 (poor) from real OSM tags. Returns (score, surface_known)."""
    v = []; s = tags.get("surface"); sm = tags.get("smoothness")
    if s: v.append(.1 if s in SURF_OK else .8 if s in SURF_BAD else .35)
    if sm in SMOOTH: v.append(SMOOTH[sm])
    base = HWY.get(tags.get("highway"), .25)
    return (.7 * max(v) + .3 * base if v else .5 * base + .5 * .3), bool(v)

def _pts(smp, k=3, cap=50):
    idx = list(range(0, len(smp["lat"]), k)); idx = idx[:cap] if len(idx) <= cap else list(np.linspace(0, len(smp["lat"]) - 1, cap).astype(int))
    return ",".join(f"{smp['lat'][i]:.5f},{smp['lon'][i]:.5f}" for i in idx)

def _fetch_context(routes):
    q = "[out:json][timeout:45];("
    for r in routes:
        p = _pts(r["smp"]); q += f'way["highway"](around:20,{p});way["waterway"~"^(river|canal|stream|drain|ditch)$"](around:80,{p});way["natural"="water"](around:60,{p});'
    return geo.overpass(q + ");out body geom 3000;")["elements"]

def _static(r, elements, lat0, lon0, elev):
    """Fixed (not rain-dependent) properties of every sample along a route."""
    smp = r["smp"]; P = smp["xy"]; n = len(P)
    roads = [e for e in elements if "highway" in e.get("tags", {}) and e.get("geometry")]; waters = [e for e in elements if e.get("geometry") and (e.get("tags", {}).get("waterway") or e.get("tags", {}).get("natural") == "water")]
    def segs(ws):
        A, B, I = [], [], []
        for i, w in enumerate(ws):
            xy = _xy([[g["lat"], g["lon"]] for g in w["geometry"]], lat0, lon0)
            if len(xy) > 1: A.append(xy[:-1]); B.append(xy[1:]); I.append(np.full(len(xy) - 1, i))
        return (np.vstack(A), np.vstack(B), np.concatenate(I)) if A else None
    cond = np.full(n, .3); known = np.zeros(n, bool); under = np.zeros(n); bridge = np.zeros(n); closed = np.zeros(n, bool); road = [""] * n
    S = segs(roads)
    if S is not None:
        d, ix = _segdist(P, S[0], S[1])
        for i in range(n):
            if d[i] > 25: continue
            t = roads[int(S[2][ix[i]])]["tags"]; cond[i], known[i] = condition(t); road[i] = t.get("name") or t.get("ref") or ""
            under[i] = float(t.get("tunnel") in ("yes", "culvert", "building_passage") or t.get("layer") in ("-1", "-2")); bridge[i] = float(t.get("bridge", "no") not in ("no", ""))
            closed[i] = bool(d[i] <= 12 and (t.get("highway") in ("construction", "proposed") or t.get("access") == "no" or t.get("construction")))
    wp = np.zeros(n); wname = [""] * n; W = segs(waters)
    if W is not None:
        d, ix = _segdist(P, W[0], W[1]); wp = np.where(d < 25, 1.0, np.where(d < 70, .5, np.where(d < 120, .15, 0.0)))
        for i in range(n):
            if d[i] < 120: t = waters[int(W[2][ix[i]])]["tags"]; wname[i] = t.get("name") or t.get("waterway") or "water body"
    wp = wp * np.where(bridge > 0, .4, 1.0)
    low = np.zeros(n)
    if elev is not None:
        e = np.asarray(elev, float); low = np.array([np.clip((e[max(0, i - 4):i + 5].mean() - e[i]) / 3, 0, 1) for i in range(n)])
    r.update(cond=cond, known=known, under=under, bridge=bridge, closed_osm=closed, road=road, wp=wp, wname=wname, low=low, elev=(None if elev is None else [round(float(x), 1) for x in elev]))

def build_context(a, b):
    """Network step (several seconds): real candidate routes + real context. The result is cached and re-scored cheaply as conditions change."""
    lat0, lon0 = a; cand = candidates(a, b); warns = []; routes = []
    for k, c in enumerate(cand): routes.append(dict(id="ABCDE"[k], coords=c["coords"], distance=c["distance"], duration=c["duration"], smp=_resample(c["coords"], lat0, lon0, max(60.0, c["distance"] / 90))))
    elev = None
    try:
        pts = [(la, lo) for r in routes for la, lo in zip(r["smp"]["lat"], r["smp"]["lon"])]; ev = []
        for i in range(0, len(pts), 100): ev += geo.elevations(pts[i:i + 100])
        elev, o = [], 0
        for r in routes: n = len(r["smp"]["lat"]); elev.append(ev[o:o + n]); o += n
    except Exception as e: warns.append("Ground heights unavailable (" + str(e)[:60] + ") - low-lying check skipped.")
    try: els = _fetch_context(routes)
    except Exception as e: els = []; warns.append("OpenStreetMap road/waterway details unavailable (" + str(e)[:60] + ") - risk uses drains, rain and ground height only.")
    for k, r in enumerate(routes): _static(r, els, lat0, lon0, elev[k] if elev else None)
    return dict(a=list(a), b=list(b), lat0=lat0, lon0=lon0, routes=routes, warnings=warns, built=time.time())

# ---------------------------------------------------------------- live flood risk of every sample
def drain_flood(o, h):
    """Flood risk (0-100) of a monitored drain now (h=0) or h*10 minutes ahead, from the same model blend the dashboard uses."""
    if h == 0 or not o.get("forecast"): return float(o.get("flood", 0))
    fc = o["forecast"]; k = min(h, len(fc) - 1); return 100 * float(np.clip(.4 * o.get("gnn", 0) / 100 + .35 * fc[k] / 100 + .25 * max(fc) / 100, 0, 1))

def sample_risk(r, ctx, drains, rain, incs, h=0):
    n = len(r["smp"]["lat"]); P = r["smp"]["xy"]; rf = float(np.clip(rain / 60, 0, 1))
    dterm = np.zeros(n); near = np.full(n, -1); closed = r["closed_osm"].copy(); flooded = np.zeros(n, bool); why = [""] * n
    if drains:
        Dxy = _xy([[d["lat"], d["lon"]] for d in drains], ctx["lat0"], ctx["lon0"]); dist = np.hypot(P[:, None, 0] - Dxy[None, :, 0], P[:, None, 1] - Dxy[None, :, 1])
        f = np.array([drain_flood(d, h) for d in drains]); w = (1 / (np.maximum(dist, 80) / 100) ** 2) * (dist <= 1000)
        idw = (w * f).sum(1) / np.maximum(w.sum(1), 1e-9); dn = dist.min(1); near = dist.argmin(1); dterm = idw * np.clip(1 - dn / 1000, 0, 1)
    c_d = .50 * dterm / 100; c_l = .22 * r["low"] * (.35 + .65 * rf); c_w = .18 * r["wp"] * (.3 + .7 * rf); c_u = .25 * r["under"] * (.2 + .8 * rf)
    risk = 100 * np.clip(c_d + c_l + c_w + c_u + .10 * rf, 0, 1)
    for inc in incs:
        if not inc.get("active", True): continue
        q = _xy([[inc["lat"], inc["lon"]]], ctx["lat0"], ctx["lon0"])[0]; m = np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) <= inc.get("radius", 60) + 30
        if inc["type"] == "flooded": risk = np.where(m, np.maximum(risk, 88), risk); flooded |= m
        else: closed |= m
        for i in np.flatnonzero(m): why[i] = {"closed": "road reported closed", "flooded": "flooding reported on the road", "blocked": "road reported blocked"}.get(inc["type"], "incident reported") + (f" ({inc['note']})" if inc.get("note") else "")
    risk = np.where(closed, 100, risk)
    return risk, dict(near=near, closed=closed, flooded=flooded, why=why, comps=np.c_[c_d, c_l, c_w, c_u])

def klass(x): return np.digitize(x, CUTS)           # 0 Safe .. 3 Critical

def _agg(risk, step, closed):
    mean = float(risk.mean()); mv = np.convolve(risk, np.ones(3) / 3, "valid") if len(risk) >= 3 else risk; peak = float(mv.max()); score = .5 * mean + .5 * peak
    crit = float((risk >= 75).sum() * step); high = float(((risk >= 50) & (risk < 75)).sum() * step); c = int(klass(score))
    if closed.any() or crit >= 100: c = 3
    elif high + crit >= 300: c = max(c, 2)
    return dict(score=round(score, 1), mean=round(mean, 1), peak=round(peak, 1), cls=c, critical_m=round(crit), high_m=round(high))

# ---------------------------------------------------------------- scoring + recommendation
def _norm_w(w, default):
    try: v = {k: max(0.0, float((w or {}).get(k, default[k]))) for k in default}
    except Exception: v = dict(default)
    s = sum(v.values()); return {k: x / s for k, x in v.items()} if s > 0 else {k: x / 100 for k, x in default.items()}

def evaluate(ctx, drains, rain, incs, mode="normal", weights=None):
    emerg = mode == "emergency"; W = _norm_w(weights, EMERGENCY_W if emerg else DEFAULT_W); slow = SLOW["emergency" if emerg else "normal"]; R = []
    for r in ctx["routes"]:
        risk, x = sample_risk(r, ctx, drains, rain, incs, 0); ag = _agg(risk, r["smp"]["step"], x["closed"]); cl = klass(risk); n = len(risk)
        adj = r["duration"] * float(np.mean([slow[c] for c in cl])); known = float(r["known"].mean() * 100); cscore = float(r["cond"].mean())
        fw = []
        for h, lab in ((0, "Now"), (2, "In 30 min"), (5, "In 60 min")):
            rk, xx = (risk, x) if h == 0 else sample_risk(r, ctx, drains, rain, incs, h); a2 = ag if h == 0 else _agg(rk, r["smp"]["step"], xx["closed"]); fw.append(dict(label=lab, score=a2["score"], cls=CLASSES[a2["cls"]]))
        # hazards: contiguous stretches that are High or worse
        haz = []; i = 0
        while i < n:
            if risk[i] >= 50:
                j = i
                while j + 1 < n and risk[j + 1] >= 50: j += 1
                k = i + int(np.argmax(risk[i:j + 1])); reasons = []
                if x["why"][k]: reasons.append(x["why"][k])
                elif x["closed"][k]: reasons.append("OpenStreetMap marks this road closed / under construction")
                names = ["near drain", "low-lying ground", "beside water", "underpass"]
                for c in np.argsort(-x["comps"][k])[:2]:
                    if x["comps"][k][c] >= .04:
                        if c == 0 and x["near"][k] >= 0: reasons.append(f"close to {drains[int(x['near'][k])]['name']} (drain risk {round(drain_flood(drains[int(x['near'][k])], 0))}/100)")
                        elif c == 1: reasons.append("low-lying ground - water collects here")
                        elif c == 2: reasons.append("beside " + (r["wname"][k] or "a waterway"))
                        elif c == 3: reasons.append("underpass / tunnel floods first")
                haz.append(dict(lat=round(float(r["smp"]["lat"][k]), 5), lon=round(float(r["smp"]["lon"][k]), 5), length_m=round((j - i + 1) * r["smp"]["step"]), risk=round(float(risk[k])), cls=CLASSES[int(klass(risk[k]))],
                                road=r["road"][k], closed=bool(x["closed"][k]), reasons=reasons[:3]))
                i = j + 1
            else: i += 1
        # class per real road vertex -> coloured segments that follow the true road geometry
        co = np.asarray(r["coords"], float); xy = _xy(co, ctx["lat0"], ctx["lon0"]); cum = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]; si = np.clip(np.round(cum / r["smp"]["step"]).astype(int), 0, n - 1); vc = cl[si]
        segs = []; cur = None
        for v in range(len(co) - 1):
            c = int(max(vc[v], vc[v + 1]))
            if cur and cur["c"] == c: cur["pts"].append([round(co[v + 1][0], 5), round(co[v + 1][1], 5)])
            else:
                cur = dict(c=c, pts=[[round(co[v][0], 5), round(co[v][1], 5)], [round(co[v + 1][0], 5), round(co[v + 1][1], 5)]]); segs.append(cur)
        prof = [[round(float(r["smp"]["s"][i]) / 1000, 2), round(float(risk[i]))] for i in np.linspace(0, n - 1, min(n, 60)).astype(int)]
        R.append(dict(id=r["id"], distance_km=round(r["distance"] / 1000, 2), duration_min=round(r["duration"] / 60, 1), adj_min=round(adj / 60, 1), delay_min=round((adj - r["duration"]) / 60, 1),
                      flood=dict(score=ag["score"], mean=ag["mean"], peak=ag["peak"], cls=CLASSES[ag["cls"]], critical_m=ag["critical_m"], high_m=ag["high_m"]), closed=bool(x["closed"].any()),
                      condition=dict(score=round(cscore, 2), label="Good" if cscore < .25 else "Fair" if cscore < .5 else "Poor", known_pct=round(known)), hazards=haz[:8], forecast=fw, profile=prof,
                      path=[[round(p[0], 5), round(p[1], 5)] for p in co.tolist()], segments=[dict(cls=CLASSES[s["c"]], pts=s["pts"]) for s in segs], _adj=adj, _d=r["distance"]))
    dmin = min(r["_d"] for r in R); tmin = min(r["_adj"] for r in R)
    for r in R:
        n_ = dict(distance=float(np.clip(r["_d"] / dmin - 1, 0, 1)), flood=(1.0 if r["closed"] else r["flood"]["score"] / 100), condition=r["condition"]["score"], time=float(np.clip(r["_adj"] / tmin - 1, 0, 1)))
        r["breakdown"] = {k: round(100 * W[k] * n_[k], 1) for k in W}; r["total"] = round(sum(W[k] * n_[k] for k in W), 4); r["suitability"] = round(100 * (1 - r["total"]))
    ok = [r for r in R if r["flood"]["cls"] in ("Safe", "Moderate") and not r["closed"]]; pool = ok or [r for r in R if r["flood"]["cls"] == "High" and not r["closed"]] or [r for r in R if not r["closed"]] or R
    best = min(pool, key=lambda r: r["total"]); caution = best not in ok
    fastest = min(R, key=lambda r: r["duration_min"]); shortest = min(R, key=lambda r: r["distance_km"])
    for r in R:
        r["recommendation"] = "Recommended" if r is best else "Alternative" if r in ok else "Avoid"; r["tags"] = [t for t, c in (("Fastest", r is fastest), ("Shortest", r is shortest)) if c]
        why = [f"{r['distance_km']} km, about {r['duration_min']} min without traffic ({r['adj_min']} min allowing for flooding)."]
        why.append(f"Flood risk {r['flood']['cls']} ({r['flood']['score']}/100): {r['flood']['critical_m']} m critical, {r['flood']['high_m']} m high.")
        if r["closed"]: why.append("Contains a closed or impassable road section.")
        why += [("%s (%d m): %s." % (h["road"] or "Road stretch", h["length_m"], "; ".join(h["reasons"]) or "high modelled flood risk")) for h in r["hazards"][:3]]
        why.append(f"Road condition {r['condition']['label'].lower()} - OpenStreetMap has surface data for {r['condition']['known_pct']}% of it.")
        r["why"] = why
    if best is fastest and not caution: head = f"Route {best['id']} is both the fastest and the safest - use it ({best['distance_km']} km, about {best['adj_min']} min)."
    elif not caution: head = f"Use Route {best['id']}: flood risk {best['flood']['cls']}, {best['distance_km']} km, about {best['adj_min']} min. The fastest route (Route {fastest['id']}) is {fastest['flood']['cls']} risk" + (" and has a closed section." if fastest["closed"] else ".")
    else: head = f"No fully safe route found. Route {best['id']} has the lowest risk ({best['flood']['cls']}). Delay the trip if you can, or ask rescue control."
    act = [i for i in incs if i.get("active", True)]
    for r in R: r.pop("_adj"); r.pop("_d")
    fw = [dict(label=lab, rec=next(f for f in best["forecast"] if f["label"] == lab), fast=next(f for f in fastest["forecast"] if f["label"] == lab)) for lab in ("Now", "In 30 min", "In 60 min")]
    k = dict(rec_id=best["id"], fast_id=fastest["id"], extra_km=round(best["distance_km"] - fastest["distance_km"], 2), extra_min=round(best["adj_min"] - fastest["adj_min"], 1), rec_min=best["adj_min"], fast_min=fastest["adj_min"], rec_delay=best["delay_min"],
             fast_delay=fastest["delay_min"], reduction=round(max(0.0, 100 * (1 - best["flood"]["score"] / max(fastest["flood"]["score"], 1)))) if fastest["flood"]["score"] > 0 else 0,
             risky_m_fastest=int(fastest["flood"]["high_m"] + fastest["flood"]["critical_m"]), incidents=len(act), drain_alerts=sum(1 for d in drains if d.get("level") == "High"), forecast=fw)
    return dict(routes=R, recommended=best["id"], caution=caution, headline=head, kpis=k, weights={a: round(100 * b) for a, b in W.items()}, mode=mode, rain=rain, warnings=ctx["warnings"], incidents=act,
                formula="Route score = " + " + ".join(f"{round(100 * W[a])}% x {a}" for a in ("distance", "flood", "condition", "time")) + " (each part 0-1, lower is better). Suitability = 100 - score.")

# ---------------------------------------------------------------- incidents you report, hospitals, place search
def load_incidents(): return geo._load("incidents.json") or []
def add_incident(lat, lon, kind, note=""):
    lst = load_incidents(); lst.append(dict(id=int(time.time() * 1000), type=kind if kind in ("closed", "flooded", "blocked") else "closed", lat=float(lat), lon=float(lon), radius=60, note=str(note)[:60], time=dt.datetime.now().strftime("%d %b %H:%M"), active=True))
    geo._save("incidents.json", lst); return lst
def resolve_incident(i):
    lst = load_incidents()
    for x in lst:
        if x["id"] == i: x["active"] = False
    geo._save("incidents.json", lst); return lst

def hospitals():
    c = geo._load("hospitals.json")
    if c is not None: return c
    els = geo.overpass(f'[out:json][timeout:30];nwr["amenity"="hospital"]({",".join(map(str, HOSP_BBOX))});out center 120;')["elements"]; out = []
    for e in els:
        p = geo._pt(e); nm = e.get("tags", {}).get("name")
        if p and nm and not any(abs(p[0] - o["lat"]) < 1e-4 and abs(p[1] - o["lon"]) < 1e-4 for o in out): out.append(dict(name=nm, lat=p[0], lon=p[1]))
    geo._save("hospitals.json", out); return out

def geocode(q):
    import urllib.parse
    j = geo._json("https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(dict(q=q, format="jsonv2", limit=6, countrycodes="in", viewbox="76.50,10.88,76.80,10.66", bounded=0)), timeout=15)
    return [dict(name=x.get("display_name", "")[:90], lat=float(x["lat"]), lon=float(x["lon"])) for x in j]
