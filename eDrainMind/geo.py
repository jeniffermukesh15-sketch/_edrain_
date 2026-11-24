"""Real Palakkad data for eDrainMind AI (all free, no API key).

  * Places ........ OpenStreetMap via the Overpass API  -> real lat/lon of Palakkad junctions, stands, hospitals ...
  * Ground height . Open-Meteo elevation API (SRTM)     -> water is routed DOWNHILL between drains
  * Rain history .. Open-Meteo archive API              -> data/rainfall.csv (trains the LSTM forecaster)
  * Waterways ..... OpenStreetMap drains / canals / streams drawn on the map

Everything is cached in data/ after the first successful download. If the internet is down the app still starts
with an approximate demo layout around Palakkad and says so on screen.
"""
import os, re, json, math, datetime as dt, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
CENTER = (10.775, 76.651)                        # Palakkad town centre
BBOX = (10.74, 76.60, 10.82, 76.70)              # south, west, north, east
UA = {"User-Agent": "eDrainMind-AI/3 (student project; contact: see README)"}
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]

# (label, regex on the OSM name).  The first 12 that are found near the town centre become drains.
TARGETS = [("Stadium Bus Stand", r"stadium bus"), ("KSRTC Bus Terminal", r"ksrtc"), ("Palakkad Fort", r"\bfort\b"),
           ("Palakkad Junction Station", r"palakkad junction|palakkad.*railway"), ("Victoria College", r"victoria"),
           ("District Hospital", r"district hospital"), ("Sultanpet Junction", r"sultanpet"), ("Olavakkode", r"olavakkode"),
           ("Kalpathi", r"kalpathi"), ("Chandranagar", r"chandranagar"), ("Kalmandapam", r"kalmandapam"),
           ("Govt Medical College", r"medical college"), ("Head Post Office", r"head post|general post"), ("Civil Station", r"civil station"),
           ("Jainimedu", r"jainimedu|jainamedu"), ("Town Hall", r"town hall"), ("Mercy College", r"mercy"), ("Court Road", r"\bcourt\b")]
DEPOT_RE = r"municipal|nagarasabha"

def _open(url, data=None, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=UA), timeout=timeout)
def _json(url, data=None, timeout=30): return json.load(_open(url, data, timeout))
def _cache(name): return os.path.join(DATA, name)
def _load(name):
    try:
        with open(_cache(name), encoding="utf-8") as f: return json.load(f)
    except Exception: return None
def _save(name, obj):
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(_cache(name), "w", encoding="utf-8") as f: json.dump(obj, f)
    except Exception as e: print("[geo] could not save", name, e)

def overpass(query):
    last = None
    for u in OVERPASS:
        try: return _json(u, urllib.parse.urlencode({"data": query}).encode(), timeout=40)
        except Exception as e: last = e
    raise last

def metres(a, b):
    """Distance in metres between (lat, lon) points (equirectangular - fine at town scale)."""
    k = 111320.0; return math.hypot((a[0] - b[0]) * k, (a[1] - b[1]) * k * math.cos(math.radians(a[0])))

def _pt(e):
    if "lat" in e: return (e["lat"], e["lon"])
    c = e.get("center"); return (c["lat"], c["lon"]) if c else None

def find_places(elements):
    """Pick the real element nearest the centre for each target; keep them at least 350 m apart so the pins do not pile up."""
    chosen = []
    for label, rx in TARGETS:
        best = None
        for e in elements:
            t = e.get("tags", {}); nm = " ".join([t.get("name", ""), t.get("name:en", "")]); p = _pt(e)
            if p and re.search(rx, nm, re.I) and BBOX[0] <= p[0] <= BBOX[2] and BBOX[1] <= p[1] <= BBOX[3]:
                d = metres(p, CENTER)
                if best is None or d < best[0]: best = (d, p, t.get("name") or label)
        if best and all(metres(best[1], c["pos"]) >= 350 for c in chosen): chosen.append(dict(label=label, osm=best[2], pos=best[1]))
        if len(chosen) == 12: break
    return chosen

def find_depot(elements):
    best = None
    for e in elements:
        nm = e.get("tags", {}).get("name", ""); p = _pt(e)
        if p and re.search(DEPOT_RE, nm, re.I): d = metres(p, CENTER); best = best if best and best[0] < d else (d, p)
    return best[1] if best else (CENTER[0] + .004, CENTER[1] - .008)

def elevations(pts):
    j = _json("https://api.open-meteo.com/v1/elevation?" + urllib.parse.urlencode({"latitude": ",".join(f"{p[0]:.5f}" for p in pts), "longitude": ",".join(f"{p[1]:.5f}" for p in pts)}), timeout=15)
    e = j["elevation"]; assert len(e) == len(pts); return [float(v) for v in e]

def downhill_edges(pts, elev, max_m=3500):
    """Each drain sends its water to the NEAREST drain that is LOWER than it. The lowest drain is the outfall."""
    n = len(pts); edges = []
    for i in range(n):
        lower = [j for j in range(n) if (elev[j], j) < (elev[i], i) and metres(pts[i], pts[j]) <= max_m]
        if lower: edges.append((i, min(lower, key=lambda j: metres(pts[i], pts[j]))))
    return edges

def build_network():
    """Download real places + heights. Raises on failure."""
    names = "Stadium|KSRTC|Fort|Palakkad Junction|Palakkad.*Railway|Victoria|District Hospital|Sultanpet|Olavakkode|Kalpathi|Chandranagar|Kalmandapam|Medical College|Post Office|Civil Station|Jainimedu|Jainamedu|Town Hall|Mercy|Court|Municipal|Nagarasabha"
    q = f'[out:json][timeout:40];nwr["name"~"{names}",i]({",".join(map(str, BBOX))});out center 600;'
    els = overpass(q)["elements"]; ch = find_places(els)
    if len(ch) < 6: raise RuntimeError(f"only {len(ch)} named places found in OpenStreetMap")
    pts = [c["pos"] for c in ch]
    try: elev = elevations(pts); src_h = "Ground heights: Open-Meteo / SRTM. Water flows to the nearest lower drain."
    except Exception as e:
        print("[geo] elevation failed:", e); elev = None; src_h = "Ground heights unavailable - flow assumed north to south."
        elev_order = [-p[0] for p in pts]
    order = elev if elev else elev_order
    ed = downhill_edges(pts, order)
    drains = [dict(id=f"D{k + 1}", name=c["label"], road=c["label"] + " area", lat=c["pos"][0], lon=c["pos"][1], elev=(round(elev[k], 1) if elev else None)) for k, c in enumerate(ch)]
    net = dict(ver=2, live=True, source="Real places from OpenStreetMap (Palakkad). " + src_h, drains=drains, edges=[[f"D{a + 1}", f"D{b + 1}"] for a, b in ed], depot=list(find_depot(els)))
    return net

def fallback_network():
    """Approximate demo layout around the real town centre (used only when OpenStreetMap cannot be reached)."""
    DR = [("D1", "College Road Jn", "College Road", 110, 90), ("D2", "College Gate", "College Road", 260, 150), ("D3", "Library Cross", "Library Road", 200, 300),
          ("D4", "Market Street", "Market Street", 420, 220), ("D5", "Bus Stand", "Bus Stand Road", 580, 160), ("D6", "Hospital Road", "Hospital Road", 700, 90),
          ("D7", "Temple Junction", "Temple Road", 810, 210), ("D8", "Canal Cross", "Canal Road", 620, 330), ("D9", "Station Road", "Station Road", 440, 400),
          ("D10", "School Lane", "School Lane", 270, 440), ("D11", "Park Avenue", "Park Avenue", 110, 380), ("D12", "Canal Outfall", "Canal Road", 780, 430)]
    ll = lambda x, y: (CENTER[0] - (y - 260) * 4e-5, CENTER[1] + (x - 450) * 4e-5)
    ed = [("D1", "D2"), ("D2", "D4"), ("D3", "D4"), ("D4", "D5"), ("D7", "D6"), ("D6", "D5"), ("D5", "D8"), ("D11", "D10"), ("D10", "D9"), ("D9", "D8"), ("D8", "D12")]
    return dict(live=False, source="OFFLINE DEMO: approximate positions around Palakkad (internet needed for real places).",
                drains=[dict(id=i, name=n, road=r, lat=ll(x, y)[0], lon=ll(x, y)[1], elev=None) for i, n, r, x, y in DR], edges=[list(e) for e in ed], depot=list(ll(40, 250)))

def load_network():
    net = _load("palakkad_network.json")
    if net and net.get("live") and net.get("ver") == 2: print("[geo] using cached real Palakkad network:", len(net["drains"]), "drains"); return net
    try:
        print("[geo] downloading real Palakkad places from OpenStreetMap ..."); net = build_network(); _save("palakkad_network.json", net)
        print("[geo] got", len(net["drains"]), "real places"); return net
    except Exception as e:
        print("[geo] online data unavailable:", str(e)[:120], "- using offline demo layout"); return fallback_network()

def ensure_rainfall():
    """Create data/rainfall.csv from real Palakkad rain history if the user has not supplied their own file."""
    p = _cache("rainfall.csv")
    if os.path.exists(p): return "Using existing data/rainfall.csv"
    try:
        lat, lon = os.getenv("WEATHER_LAT", str(CENTER[0])), os.getenv("WEATHER_LON", str(CENTER[1]))
        end = dt.date.today() - dt.timedelta(days=7); start = end - dt.timedelta(days=730)
        j = _json("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly="precipitation", timezone="Asia/Kolkata")), timeout=45)
        v = [0.0 if x is None else float(x) for x in j["hourly"]["precipitation"]]
        if len(v) < 500: raise RuntimeError("too little data")
        os.makedirs(DATA, exist_ok=True)
        with open(p, "w") as f: f.write("rain\n" + "\n".join(f"{x:.2f}" for x in v) + "\n")
        return f"Downloaded {len(v)} hours of real Palakkad rainfall ({start} to {end}) from Open-Meteo"
    except Exception as e: return "Real rainfall not downloaded (" + str(e)[:80] + ") - using synthetic rain"

def water_lines():
    """Real drains / canals / streams around Palakkad from OpenStreetMap (cached)."""
    c = _load("osm_water.json")
    if c is not None: return c
    q = f'[out:json][timeout:40];way["waterway"~"^(drain|ditch|canal|stream|river)$"]({",".join(map(str, BBOX))});out geom 800;'
    out = [dict(kind=e["tags"].get("waterway", ""), name=e["tags"].get("name", ""), pts=[[g["lat"], g["lon"]] for g in e.get("geometry", [])]) for e in overpass(q)["elements"] if e.get("geometry")]
    _save("osm_water.json", out); return out


# ---------------- extra data helpers (v3.2) ----------------
def rain_stats():
    """Summary of the real rainfall history file (shown in the Input data section)."""
    p = _cache("rainfall.csv")
    try:
        import numpy as np
        a = np.genfromtxt(p, delimiter=",", names=True)["rain"]; a = np.nan_to_num(a)
        return dict(hours=int(len(a)), max_mm_h=round(float(a.max()), 1), heavy_hours=int((a >= 7.6).sum()), rainy_hours=int((a >= 0.1).sum()), years=round(len(a) / 8766, 1))
    except Exception: return None

def load_custom(): return _load("custom_drains.json") or []
def save_custom(lst): _save("custom_drains.json", lst)

def rebuild_edges(drains, old):
    """Water flows downhill. With heights for every drain the network is rebuilt; otherwise old edges are kept and new spots join the nearest drain."""
    if all(d.get("elev") is not None for d in drains):
        e = downhill_edges([(d["lat"], d["lon"]) for d in drains], [d["elev"] for d in drains]); return [(drains[a]["id"], drains[b]["id"]) for a, b in e]
    out = [tuple(e) for e in old]; has = {a for a, _ in out}
    for d in drains:
        if d["id"] not in has and d["id"].startswith("C") and len(drains) > 1:
            near = min((x for x in drains if x is not d), key=lambda x: metres((d["lat"], d["lon"]), (x["lat"], x["lon"]))); out.append((d["id"], near["id"]))
    return out

def _len(pts): return sum(metres(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

def area_info(lat, lon, R=600):
    """Real drainage around any clicked point: OSM drains/canals/streams/culverts/manholes + SRTM heights (which way is downhill)."""
    res = dict(lat=lat, lon=lon, radius=R, waterways=[], points=[], osm_error=None, elev=None)
    q = (f'[out:json][timeout:30];(way["waterway"](around:{R},{lat},{lon});way["tunnel"="culvert"](around:{R},{lat},{lon});way["man_made"="pipeline"]["substance"~"water|rain|storm"](around:{R},{lat},{lon});'
         f'node["manhole"](around:{R},{lat},{lon});node["man_made"~"manhole|storm_drain"](around:{R},{lat},{lon});node["waterway"~"drain|stream_end|weir|sluice_gate|dam"](around:{R},{lat},{lon}););out geom 400;')
    try:
        for e in overpass(q)["elements"]:
            t = e.get("tags", {})
            if e["type"] == "way" and e.get("geometry"):
                pts = [[g["lat"], g["lon"]] for g in e["geometry"]]; kind = t.get("waterway") or ("culvert" if t.get("tunnel") == "culvert" else "pipe")
                res["waterways"].append(dict(kind=kind, name=t.get("name", ""), pts=pts, length=round(_len(pts))))
            elif e["type"] == "node": res["points"].append(dict(kind=t.get("manhole") or t.get("man_made") or t.get("waterway") or "drain point", lat=e["lat"], lon=e["lon"]))
    except Exception as e: res["osm_error"] = str(e)[:100]
    d = 200; dl = d / 111320; dn = d / (111320 * math.cos(math.radians(lat))); dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    ring = [(lat + a * dl, lon + b * dn) for a, b in dirs]
    try:
        ev = elevations([(lat, lon)] + ring); here = ev[0]; nb = ev[1:]; k = min(range(8), key=lambda i: nb[i])
        res["elev"] = dict(here=round(here, 1), ring_mean=round(sum(nb) / 8, 1), pocket=round(sum(nb) / 8 - here, 1),
                           downhill=(dict(lat=ring[k][0], lon=ring[k][1], dist=d, drop=round(here - nb[k], 1)) if nb[k] < here - 0.3 else None))
    except Exception as e: res["elev_error"] = str(e)[:100]
    return res
