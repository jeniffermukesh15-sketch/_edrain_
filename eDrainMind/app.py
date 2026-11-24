"""eDrainMind AI - Flask server.  Run:  python app.py   then open http://127.0.0.1:5000"""
import io, re, os, csv, json, base64, copy, math, ssl, time, smtplib, threading, urllib.request, urllib.error, datetime as dt
from email.message import EmailMessage
import numpy as np
from flask import Flask, jsonify, request, send_file, Response
from PIL import Image
import ml, geo, routes

def load_env():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if "=" in l and not l.strip().startswith("#"): k, v = l.strip().split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))
load_env()
app = Flask(__name__)
RAIN_SRC = geo.ensure_rainfall(); print("[data]", RAIN_SRC)
RAIN_STATS = geo.rain_stats()
NET = geo.load_network()
CUSTOM = geo.load_custom()      # spots you added with "Monitor this spot as a new drain" (saved in data/custom_drains.json)
for _k, _c in enumerate(CUSTOM, 1): NET["drains"].append(dict(id=f"C{_k}", name=_c["name"], road=_c["name"] + " area", lat=_c["lat"], lon=_c["lon"], elev=_c.get("elev")))
if CUSTOM: NET["edges"] = [list(e) for e in geo.rebuild_edges(NET["drains"], NET["edges"])]
print("Training models (15-60 seconds)...")
BLOCK = ml.train_blockage(); CV = ml.Vision(); FC = ml.train_forecaster()
print("Models ready:", ml.M["blockage"]["model"], "|", ml.M["vision"]["model"], "|", ml.M["forecaster"]["model"])

EDGES = [tuple(e) for e in NET["edges"]]; IDS = [d["id"] for d in NET["drains"]]
def _xy(lat, lon): return ((lon - geo.CENTER[1]) * 111320 * math.cos(math.radians(geo.CENTER[0])), (geo.CENTER[0] - lat) * 111320)   # metres from town centre
INFO = {d["id"]: dict(id=d["id"], name=d["name"], road=d["road"], lat=d["lat"], lon=d["lon"], elev=d["elev"], x=_xy(d["lat"], d["lon"])[0], y=_xy(d["lat"], d["lon"])[1]) for d in NET["drains"]}
DEPOT_LL = tuple(NET["depot"]); DEPOT = _xy(*DEPOT_LL); S = {}; MIN_PER_M = 1 / 250   # crews drive about 15 km/h in town traffic

def tick(St, n=1):
    for _ in range(n):
        St["step"] += 1
        for d in St["d"].values():
            lr = max(0.0, St["rain"] * d["mult"] + np.random.normal(0, 1.5)) if St["rain"] > 0 else 0.0
            d["waste"] = float(np.clip(d["waste"] + .0035 * lr / 10 + np.random.normal(0, .004), 0, 1))
            d["w"] = float(np.clip(.8 * d["w"] + .28 * lr * (1 + 1.2 * d["waste"]) + np.random.normal(0, 1), 0, 100))
            d["wh"] = (d["wh"] + [d["w"]])[-24:]; d["rh"] = (d["rh"] + [lr])[-24:]

def overflow_eta(w, fc, th=80):
    """Minutes until the water level crosses the overflow line (each step = 10 min); 0 = already over, None = not within the forecast."""
    if w >= th: return 0
    prev = w
    for k, v in enumerate(fc):
        if v >= th: return max(1, round(10 * k + 10 * (th - prev) / max(v - prev, 1e-6)))
        prev = v
    return None
def eta_text(e): return "Overflowing now" if e == 0 else f"Overflows in about {e} min" if e else "No overflow expected in the next hour"

def analyze(St):
    D = St["d"]; p = BLOCK.predict_proba(np.array([[D[i]["rh"][-1], D[i]["w"], D[i]["waste"], D[i]["past"], D[i]["age"]] for i in IDS]))[:, 1]
    g = ml.gnn(IDS, EDGES, p); out = []
    for k, i in enumerate(IDS):
        d = D[i]; fc = ml.forecast(FC, d["wh"], d["rh"]); peak = max(fc); flood = float(np.clip(.4 * g[k] + .35 * d["w"] / 100 + .25 * peak / 100, 0, 1)); eta = overflow_eta(d["w"], fc)
        o = dict(INFO[i], water=round(d["w"], 1), waste=round(d["waste"] * 100), blockage=round(float(p[k]) * 100), flood=round(flood * 100), forecast=fc, peak=peak,
                 level="High" if flood >= .6 else "Medium" if flood >= .35 else "Low", history=[round(v, 1) for v in d["wh"][-12:]], eta=eta, eta_text=eta_text(eta),
                 status="Possible blockage" if p[k] > .5 else "Flowing")
        ups = [INFO[u]["name"] for u, v in EDGES if v == i]; down = [INFO[v]["name"] for u, v in EDGES if u == i]
        t = [f"{o['name']} on {o['road']} is {o['level'].lower()} risk (score {o['flood']}/100). {o['eta_text']}."]
        if o["waste"] >= 50: t.append(f"The camera sees about {o['waste']}% obstruction, so blockage chance is {o['blockage']}%.")
        if ups and o["level"] != "Low": t.append("Water may back up into " + ", ".join(ups) + ".")
        o.update(upstream=ups, downstream=down, rain_now=round(d["rh"][-1], 1), past=int(d["past"]), age=round(d["age"], 1), cleaned=d.get("cleaned"), photo=d.get("photo"), gnn=round(float(g[k]) * 100, 1))
        o["explain"] = " ".join(t)
        o["rec"] = (f"Clean {o['name']} immediately." + (f" Then check {', '.join(down)}." if down else "")) if o["level"] == "High" else \
                   f"Inspect {o['name']} within 24 hours." if o["level"] == "Medium" else "No action needed. Routine inspection only."
        out.append(o)
    return out

def refresh():
    clock = lambda: (S["t0"] + dt.timedelta(minutes=10 * S["step"])).strftime("%H:%M"); out = analyze(S); new = []
    for o in out:
        if o["level"] == "High" and S["prev"].get(o["id"]) != "High":
            new.append(dict(time=clock(), id=o["id"], name=o["name"], road=o["road"], msg=f"{o['name']} ({o['road']}): {o['eta_text']}. {o['rec']}", mail=dict(text="Sending...")))
        S["prev"][o["id"]] = o["level"]
    if new:
        S["alerts"] = (new + S["alerts"])[:30]; print("[ALERT]", "; ".join(a["msg"] for a in new))
        body = f"eDrainMind AI alert at {clock()} (rainfall {S['rain']} mm/h)\n\n" + "\n".join("- " + a["msg"] for a in new)
        threading.Thread(target=send_email, args=(f"[eDrainMind] {len(new)} drain(s) at HIGH flood risk", body, new[0]["mail"], [a["mail"] for a in new[1:]]), daemon=True).start()
    lv = {o["id"]: o["level"] for o in out}
    for a in S["alerts"]: a["active"] = lv.get(a["id"]) == "High"
    roads = {}
    for o in out:
        r = roads.setdefault(o["road"], dict(road=o["road"], flood=0, level="Low", drains=[])); r["drains"].append(o["name"]); r["flood"] = max(r["flood"], o["flood"])
    for r in roads.values(): r["level"] = "High" if r["flood"] >= 60 else "Medium" if r["flood"] >= 35 else "Low"
    S["out"] = dict(rain=S["rain"], clock=clock(), step=S["step"], drains=out, edges=EDGES, alerts=S["alerts"], inputs=dict(rain_hist=RAIN_STATS), depot=list(DEPOT_LL), geo=dict(source=NET["source"], live=NET["live"], center=list(geo.CENTER), rain=RAIN_SRC), roads=sorted(roads.values(), key=lambda r: -r["flood"]),
                    counts={l: sum(o["level"] == l for o in out) for l in ("Low", "Medium", "High")},
                    cfg=dict(email=bool(os.getenv("SMTP_USER") and os.getenv("SMTP_PASS") and os.getenv("ALERT_TO")), ai=bool(os.getenv("LLM_API_KEY")), torch=ml.HAS_TORCH))

LAST_MAIL = 0.0
def send_email(subject, body, mail, others=(), force=False):
    """Gmail: tries port 587 (STARTTLS) and then 465 (SSL). Spaces in the app password are removed automatically."""
    global LAST_MAIL
    u, pw, to = os.getenv("SMTP_USER", "").strip(), os.getenv("SMTP_PASS", "").replace(" ", "").strip(), os.getenv("ALERT_TO", "").strip(); res = "Email not set up (see README)"
    if u and pw and to:
        if not force and time.time() - LAST_MAIL < 45: res = "Not emailed (limit: one alert email per 45 seconds)"
        else:
            LAST_MAIL = time.time(); m = EmailMessage(); m["Subject"], m["From"], m["To"] = subject, u, to; m.set_content(body)
            host = os.getenv("SMTP_HOST", "smtp.gmail.com"); port = int(os.getenv("SMTP_PORT", "587")); modes = [("ssl", 465)] if port == 465 else [("tls", port), ("ssl", 465)]
            for kind, p in modes:
                try:
                    ctx = ssl.create_default_context()
                    if kind == "ssl": s = smtplib.SMTP_SSL(host, p, timeout=15, context=ctx)
                    else: s = smtplib.SMTP(host, p, timeout=15); s.ehlo(); s.starttls(context=ctx); s.ehlo()
                    with s: s.login(u, pw); s.send_message(m)
                    res = "Email sent to " + to; break
                except smtplib.SMTPAuthenticationError: res = "Email failed: Gmail rejected the login. Use the 16-letter App password (2-Step Verification must be on), not your normal password."; break
                except Exception as e: res = f"Email failed on port {p}: {str(e)[:70]}. Your antivirus or Wi-Fi may block email - try a mobile hotspot."
    for x in [mail, *others]: x["text"] = res
    return res

LLM_ERR = ""
class LLMError(RuntimeError):
    def __init__(s, msg, code=0): super().__init__(msg); s.code = code
def _friendly(code, body):
    if code in (429, 503): return f"the AI service is busy right now (HTTP {code}, high demand) - try again in a minute"
    if code in (401, 403): return f"the API key was rejected (HTTP {code}) - check LLM_API_KEY in .env"
    if code == 404: return "model name not found - set LLM_MODEL to a current model name"
    return f"HTTP {code}: " + body.replace("\n", " ")[:140]
def _post_json(url, headers, body, retries=1, timeout=30):
    """POST JSON; one automatic retry after a short pause for temporary errors (429/500/502/503/504)."""
    for k in range(retries + 1):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), dict(headers, **{"Content-Type": "application/json"}))
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and k < retries: time.sleep(2); continue
            raise LLMError(_friendly(e.code, e.read().decode("utf-8", "ignore")), e.code)
        except Exception as e: raise LLMError(str(e)[:150])
def _models(base):
    m = [os.getenv("LLM_MODEL", "gemini-2.0-flash").strip()]; extra = os.getenv("LLM_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite" if "generativelanguage" in base else "")
    for x in extra.split(","):
        if x.strip() and x.strip() not in m: m.append(x.strip())
    return m
LLM_ERR = ""; LLM_USED = ""
def llm(system, user, image=None):
    """Free Gemini/Groq etc. If the main model is busy (503) it retries, then tries the fallback models. `image` = JPEG bytes for photo questions."""
    global LLM_ERR, LLM_USED
    key = os.getenv("LLM_API_KEY", "").strip()
    if not key: LLM_ERR = "no LLM_API_KEY found in .env"; return None
    base = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/"); errs = []
    b64 = base64.b64encode(image).decode() if image else None
    for model in _models(base):
        content = user if not image else [dict(type="text", text=user), dict(type="image_url", image_url=dict(url="data:image/jpeg;base64," + b64))]
        tries = [(base + "/chat/completions", {"Authorization": "Bearer " + key}, dict(model=model, max_tokens=1024, messages=[dict(role="system", content=system), dict(role="user", content=content)]),
                  lambda r: r["choices"][0]["message"]["content"])]
        if "generativelanguage" in base:
            parts = [dict(text=system + "\n\n" + user)] + ([dict(inline_data=dict(mime_type="image/jpeg", data=b64))] if image else [])
            tries.append((f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", {"x-goog-api-key": key}, dict(contents=[dict(parts=parts)], generationConfig=dict(maxOutputTokens=1024)),
                          lambda r: r["candidates"][0]["content"]["parts"][0]["text"]))
        for k, (url, h, body, parse) in enumerate(tries):
            try:
                txt = (parse(_post_json(url, h, body)) or "").strip()
                if txt: LLM_ERR = ""; LLM_USED = model; return txt
                errs.append("the model returned an empty reply")
            except LLMError as e:
                errs.append(str(e))
                if e.code in (429, 503, 404): break          # same model, same problem on the second endpoint - go to the next model
            except Exception as e: errs.append(str(e)[:120])
    LLM_ERR = errs[0] if errs else "unknown AI error"; print("[AI error]", " | ".join(errs)[:300]); return None

VIS_SYS = "You inspect photos of storm drains, gutters, culverts and drain pipes for a municipal flood-control team. Describe only what is visible."
VIS_Q = ('Reply with ONLY one JSON object, no markdown: {"is_drain": true or false, "blockage_percent": integer 0-100 = share of the drain opening or channel blocked by waste, silt, vegetation or debris '
         '(0 = clear and flowing, 100 = fully blocked), "waste_types": [short strings such as "plastic bottles", "plastic bags", "leaves", "silt", "construction debris"], '
         '"flow": one of "clear","slow","blocked","overflowing", "summary": one plain sentence for a non-technical officer}. If the photo is not a drain, pipe or water channel set is_drain false and blockage_percent 0. '
         'Healthy grass on the bank and normal flowing water are NOT blockage.')
def vision_llm(img):
    buf = io.BytesIO(); im = img.convert("RGB"); im.thumbnail((800, 800)); im.save(buf, "JPEG", quality=85)
    txt = llm(VIS_SYS, VIS_Q, image=buf.getvalue())
    if not txt: return None
    try:
        j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)); sc = float(np.clip(float(j.get("blockage_percent", 0)), 0, 100))
        return dict(is_drain=bool(j.get("is_drain", True)), score=sc, items=[str(x)[:30] for x in (j.get("waste_types") or [])][:6], flow=str(j.get("flow", "")), summary=str(j.get("summary", ""))[:240],
                    source="AI vision (" + LLM_USED + ")", confidence="high")
    except Exception as e:
        global LLM_ERR; LLM_ERR = "the AI reply could not be read as JSON"; print("[AI vision parse]", e, txt[:200]); return None
def local_vision(img):
    if ml.looks_like_sample(img): sc = CV.score(img) * 100; items = []; conf = "medium (built-in model trained on sample grate pictures)"
    else: sc, items = ml.rough_waste(img); sc *= 100; conf = "low (rough colour estimate - add LLM_API_KEY for real photo understanding)"
    return dict(is_drain=True, score=sc, items=items, flow="", summary="Built-in estimate from colours and shapes only.", source="Built-in model (no AI key / AI unavailable)", confidence=conf)

def context():
    s = S["out"]; return f"Time {s['clock']}, rainfall {s['rain']} mm/h. Drains:\n" + "\n".join(f"- {o['name']} ({o['road']}): {o['level']} risk {o['flood']}/100, water {o['water']}%, blockage chance {o['blockage']}%, {o['eta_text']}" for o in s["drains"])
SYS = "You are the assistant of a city drainage control room. Answer in plain, short language for a non-technical officer, using only the data given. Never invent numbers."

def local_answer(q):
    s = S["out"]; ql = q.lower(); bad = [r for r in s["roads"] if r["level"] != "Low"]; ds = sorted(s["drains"], key=lambda o: (o["eta"] if o["eta"] is not None else 999, -o["flood"]))
    for o in s["drains"]:
        if o["name"].lower() in ql: return f"{o['name']}: {o['level']} risk ({o['flood']}/100), water {o['water']}%, blockage chance {o['blockage']}%, {o['eta_text'].lower()}. {o['rec']}"
    if re.search(r"\b(time|clock|date|day)\b", ql): return f"Simulation time is {s['clock']}. Real time now: {dt.datetime.now().strftime('%H:%M, %d %b %Y')}."
    if "rain" in ql: return f"Rainfall in the simulator is {s['rain']} mm/h. Use the sliders on the right to try dry, drizzle, heavy, storm or cyclone."
    if any(w in ql for w in ("road", "flood", "risky", "waterlog")): return "Roads at risk: " + ", ".join(f"{r['road']} ({r['level']})" for r in bad) + "." if bad else "No road is at risk right now."
    if any(w in ql for w in ("first", "clean", "priority", "inspect")): return f"Start with {ds[0]['name']} ({ds[0]['road']}): {ds[0]['eta_text'].lower()}, risk {ds[0]['flood']}/100." if ds[0]["level"] != "Low" else "Nothing needs cleaning urgently."
    if any(w in ql for w in ("overflow", "hour", "when")):
        e = [o for o in ds if o["eta"] is not None]; return "; ".join(f"{o['name']}: {o['eta_text'].lower()}" for o in e[:4]) + "." if e else "No drain is expected to overflow in the next hour."
    c = s["counts"]; return f"Right now {c['High']} drains are high risk, {c['Medium']} medium and {c['Low']} low at {s['rain']} mm/h rainfall. (Built-in answer: ask about a drain name, roads at risk, what to clean first, overflow, rain or time.)"

def make_plan(n):
    ds = S["out"]["drains"]; tg = [o for o in ds if o["level"] == "High"] or [o for o in ds if o["level"] == "Medium"]; tg.sort(key=lambda o: (o["eta"] if o["eta"] is not None else 999, -o["flood"]))
    cr = [dict(id=k + 1, pos=DEPOT, t=0.0, stops=[], path=[list(DEPOT_LL)]) for k in range(n)]
    for o in tg:
        pt = (o["x"], o["y"]); c = min(cr, key=lambda c: c["t"] + math.dist(c["pos"], pt) * MIN_PER_M); arr = c["t"] + math.dist(c["pos"], pt) * MIN_PER_M
        c["stops"].append(dict(id=o["id"], name=o["name"], road=o["road"], level=o["level"], arrive=round(arr), late=o["eta"] is not None and arr > o["eta"], eta=o["eta"]))
        c["t"] = arr + 15; c["pos"] = pt; c["path"].append([o["lat"], o["lon"]])
    return dict(crews=[c for c in cr if c["stops"]], total=round(max([c["t"] for c in cr if c["stops"]] or [0])), targets=[o["id"] for o in tg], n=n)

def reset(rain=8.0):
    r = np.random.default_rng(7); S.clear()
    S.update(rain=rain, step=0, alerts=[], prev={}, t0=dt.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0),
             d={i: dict(w=float(r.uniform(5, 15)), waste=float(r.uniform(.03, .3)), past=int(r.poisson(2)), age=float(r.uniform(2, 28)), mult=float(r.uniform(.9, 1.1)), wh=[], rh=[], cleaned=None, photo=None) for i in IDS})
    tick(S, 8); S["step"] = 0; refresh()

@app.get("/")
def index(): return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html"), mimetype="text/html", max_age=0)
@app.get("/api/osm_water")
def osm_water():
    try: return jsonify(geo.water_lines())
    except Exception as e: return jsonify([])
@app.get("/api/state")
def state(): return jsonify(S["out"])
@app.post("/api/rain")
def rain(): S["rain"] = float(np.clip(request.json.get("rain", 0), 0, 100)); tick(S); refresh(); return jsonify(S["out"])
@app.post("/api/tick")
def do_tick(): tick(S, int(request.json.get("n", 1)) if request.is_json else 1); refresh(); return jsonify(S["out"])
@app.post("/api/reset")
def do_reset(): reset(); return jsonify(S["out"])
@app.post("/api/clean")
def clean():
    for i in ([request.json.get("id")] if request.json.get("id") else request.json.get("ids", [])):
        d = S["d"].get(i)
        if d: d["waste"] = .02; d["w"] *= .5; d["past"] += 1; d["cleaned"] = (S["t0"] + dt.timedelta(minutes=10 * S["step"])).strftime("%H:%M") + " (this session)"; d["photo"] = None
    refresh(); return jsonify(S["out"])
@app.get("/api/plan")
def plan(): return jsonify(make_plan(int(np.clip(int(request.args.get("crews", 2)), 1, 3))))
@app.post("/api/analyze_image")
def analyze_image():
    f = request.files.get("file")
    if not f: return jsonify(error="No image uploaded"), 400
    try: img = Image.open(f.stream); img.load()
    except Exception: return jsonify(error="That file is not a readable image"), 400
    did = request.form.get("drain_id"); r = vision_llm(img); note = ""
    if r is None: why = LLM_ERR; r = local_vision(img); note = " (AI vision not used: " + why + ")"
    sc = round(r["score"]); clock = (S["t0"] + dt.timedelta(minutes=10 * S["step"])).strftime("%H:%M")
    label = "This does not look like a drain photo" if not r["is_drain"] else "Drain looks clear" if sc < 20 else "Partly blocked" if sc < 50 else "Blocked - obstruction detected"
    used = bool(r["is_drain"] and did in S["d"])
    if used: S["d"][did]["waste"] = sc / 100; S["d"][did]["photo"] = dict(score=sc, label=label, summary=r["summary"], source=r["source"], confidence=r["confidence"], items=r["items"], time=clock)
    refresh()
    return jsonify(score=sc, label=label, summary=r["summary"], items=r["items"], flow=r["flow"], source=r["source"] + note, confidence=r["confidence"], risk_updated=used, state=S["out"])
@app.get("/api/sample_image")
def sample():
    buf = io.BytesIO(); ml.make_image(request.args.get("kind") == "blocked", int(request.args.get("seed", 5000))).save(buf, "PNG"); buf.seek(0); return send_file(buf, mimetype="image/png")
AREA = {}
@app.get("/api/area")
def area():
    """Drainage around any point you click on the map: real OSM drains + ground heights + estimated flood risk from the nearest monitored drains."""
    try: lat, lon = float(request.args["lat"]), float(request.args["lon"])
    except Exception: return jsonify(error="lat and lon are needed"), 400
    R = int(np.clip(float(request.args.get("r", 600)), 200, 1500)); key = (round(lat, 4), round(lon, 4), R)
    if key not in AREA:
        if len(AREA) > 200: AREA.clear()
        AREA[key] = geo.area_info(lat, lon, R)
    a = dict(AREA[key]); near = sorted(((geo.metres((lat, lon), (o["lat"], o["lon"])), o) for o in S["out"]["drains"]), key=lambda t: t[0])[:3]
    wts = [1 / max(d, 50) ** 2 for d, _ in near]; idw = sum(w * o["flood"] for w, (_, o) in zip(wts, near)) / sum(wts)
    pocket = (a.get("elev") or {}).get("pocket") or 0; low = float(np.clip(pocket / 6, 0, 1)); rf = float(np.clip(S["rain"] / 60, 0, 1))
    risk = round(100 * float(np.clip(.55 * idw / 100 + .25 * low + .20 * rf, 0, 1)))
    a.update(risk=risk, level="High" if risk >= 60 else "Medium" if risk >= 35 else "Low", rain=S["rain"],
             nearest=[dict(id=o["id"], name=o["name"], dist=round(d), flood=o["flood"], level=o["level"]) for d, o in near],
             total_length=sum(w["length"] for w in a["waterways"]))
    return jsonify(a)
@app.post("/api/add_drain")
def add_drain():
    j = request.get_json(silent=True) or {}
    try: lat, lon = float(j["lat"]), float(j["lon"])
    except Exception: return jsonify(error="lat and lon are needed"), 400
    name = str(j.get("name") or "").strip()[:40] or f"Spot {len(CUSTOM) + 1}"
    try: elev = geo.elevations([(lat, lon)])[0]
    except Exception: elev = None
    CUSTOM.append(dict(name=name, lat=lat, lon=lon, elev=elev)); geo.save_custom(CUSTOM); did = f"C{len(CUSTOM)}"
    NET["drains"].append(dict(id=did, name=name, road=name + " area", lat=lat, lon=lon, elev=(round(elev, 1) if elev is not None else None)))
    x, y = _xy(lat, lon); INFO[did] = dict(id=did, name=name, road=name + " area", lat=lat, lon=lon, elev=(round(elev, 1) if elev is not None else None), x=x, y=y); IDS.append(did)
    EDGES[:] = [tuple(e) for e in geo.rebuild_edges(NET["drains"], EDGES)]
    r = np.random.default_rng(100 + len(CUSTOM)); ref = S["d"][IDS[0]]
    S["d"][did] = dict(w=float(r.uniform(5, 15)), waste=float(r.uniform(.03, .3)), past=int(r.poisson(2)), age=float(r.uniform(2, 28)), mult=float(r.uniform(.9, 1.1)), wh=list(ref["wh"]), rh=list(ref["rh"]), cleaned=None, photo=None)
    refresh(); return jsonify(S["out"])
RCTX = {}      # route id -> real candidate routes + context (re-scored cheaply whenever rain / drains / incidents change)
def _pt(j, k):
    try: p = j[k]; return (float(p["lat"]), float(p["lon"]))
    except Exception: return None
@app.get("/api/geocode")
def geocode():
    q = (request.args.get("q") or "").strip()[:80]
    if len(q) < 3: return jsonify([])
    try: return jsonify(routes.geocode(q + (", Palakkad" if "palakkad" not in q.lower() else "")))
    except Exception as e: return jsonify(error="Place search needs internet: " + str(e)[:80]), 502
@app.get("/api/hospitals")
def hospitals():
    try: return jsonify(routes.hospitals())
    except Exception as e: return jsonify(error="Hospital list needs internet: " + str(e)[:80]), 502
@app.post("/api/route")
def route_plan():
    j = request.get_json(silent=True) or {}; a, b = _pt(j, "from"), _pt(j, "to")
    if not a or not b: return jsonify(error="Choose a start and a destination first."), 400
    d = geo.metres(a, b)
    if d < 150: return jsonify(error="Start and destination are almost the same place."), 400
    if d > 40000: return jsonify(error="Routes longer than 40 km are not supported."), 400
    try: ctx = routes.build_context(a, b)
    except Exception as e: return jsonify(error=str(e)), 502
    rid = str(int(time.time() * 1000)); RCTX[rid] = ctx
    while len(RCTX) > 20: RCTX.pop(next(iter(RCTX)))
    res = routes.evaluate(ctx, S["out"]["drains"], S["rain"], routes.load_incidents(), j.get("mode", "normal"), j.get("weights")); res.update(id=rid, a=list(a), b=list(b)); return jsonify(res)
@app.post("/api/route_eval")
def route_eval():
    j = request.get_json(silent=True) or {}; ctx = RCTX.get(str(j.get("id")))
    if not ctx: return jsonify(error="That route expired - plan it again.", expired=True), 404
    res = routes.evaluate(ctx, S["out"]["drains"], S["rain"], routes.load_incidents(), j.get("mode", "normal"), j.get("weights")); res.update(id=str(j.get("id")), a=ctx["a"], b=ctx["b"]); return jsonify(res)
@app.get("/api/incidents")
def incidents_list(): return jsonify([i for i in routes.load_incidents() if i.get("active", True)])
@app.post("/api/incidents")
def incidents_add():
    j = request.get_json(silent=True) or {}
    try: lat, lon = float(j["lat"]), float(j["lon"])
    except Exception: return jsonify(error="lat and lon are needed"), 400
    routes.add_incident(lat, lon, j.get("type", "closed"), j.get("note", "")); return jsonify([i for i in routes.load_incidents() if i.get("active", True)])
@app.delete("/api/incidents/<int:iid>")
def incidents_resolve(iid): routes.resolve_incident(iid); return jsonify([i for i in routes.load_incidents() if i.get("active", True)])
@app.get("/api/weather")
def weather():
    lat, lon, place = os.getenv("WEATHER_LAT", "10.79"), os.getenv("WEATHER_LON", "76.65"), os.getenv("WEATHER_PLACE", "Palakkad")
    try:
        j = json.load(urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation&forecast_days=2&timezone=auto", timeout=6))
        t, p = j["hourly"]["time"], j["hourly"]["precipitation"]; now = dt.datetime.now().strftime("%Y-%m-%dT%H:00"); i = next((k for k, x in enumerate(t) if x >= now), 0)
        hrs = [dict(t=t[k][11:16], mm=p[k] or 0) for k in range(i, min(i + 24, len(t)))]
        return jsonify(ok=True, hours=hrs, peak=max(h["mm"] for h in hrs), source="Live forecast from Open-Meteo", place=place, lat=lat, lon=lon)
    except Exception:
        demo = [0, 0, 1, 2, 6, 14, 22, 18, 9, 4, 1, 0] + [0] * 12
        return jsonify(ok=False, hours=[dict(t=f"{(dt.datetime.now().hour + k) % 24:02d}:00", mm=demo[k]) for k in range(24)], peak=22, source="Offline (no internet) - showing a demo storm", place=place, lat=lat, lon=lon)
@app.get("/api/whatif")
def whatif():
    St = dict(rain=float(request.args.get("rain", 0)), step=S["step"], d=copy.deepcopy(S["d"])); tick(St, 6); a = sorted(analyze(St), key=lambda o: -o["flood"])
    return jsonify(high=sum(o["level"] == "High" for o in a), medium=sum(o["level"] == "Medium" for o in a), top=[dict(name=o["name"], level=o["level"], eta_text=o["eta_text"]) for o in a[:4]])
@app.post("/api/ask")
def ask():
    q = (request.json.get("question") or "").strip()[:300]; a = llm(SYS, context() + "\n\nQuestion: " + q)
    return jsonify(answer=a or local_answer(q), source="AI assistant" if a else "Built-in answer - AI not used: " + LLM_ERR)
@app.get("/api/explain")
def explain():
    o = next((x for x in S["out"]["drains"] if x["id"] == request.args.get("id")), None)
    if not o: return jsonify(error="Unknown drain"), 404
    a = llm(SYS, context() + f"\n\nExplain in 3 short sentences why {o['name']} has {o['level']} risk and what the crew should do.")
    return jsonify(text=a or o["explain"] + " " + o["rec"], source="AI assistant" if a else "Built-in explanation - AI not used: " + LLM_ERR)
@app.post("/api/email/test")
def email_test():
    m = dict(text=""); return jsonify(status=send_email("[eDrainMind] Test email", "This is a test alert from eDrainMind AI. Email alerts are working.", m, force=True))
@app.get("/api/metrics")
def metrics():
    return jsonify(dict(ml.M, llm=bool(os.getenv("LLM_API_KEY")), email=S["out"]["cfg"]["email"], torch=ml.HAS_TORCH))
@app.get("/api/report.csv")
def report_csv():
    b = io.StringIO(); w = csv.writer(b); w.writerow(["Drain", "Road", "Water %", "Obstruction %", "Blockage %", "Flood risk", "Level", "Overflow", "Recommendation"])
    for o in S["out"]["drains"]: w.writerow([o["name"], o["road"], o["water"], o["waste"], o["blockage"], o["flood"], o["level"], o["eta_text"], o["rec"]])
    return Response(b.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=edrainmind_report.csv"})
@app.get("/api/report.txt")
def report_txt():
    s = S["out"]; pl = make_plan(2); summ = llm(SYS, context() + "\n\nWrite a 4-sentence situation summary for the commissioner.") or local_answer("summary")
    L = [f"eDrainMind AI - Situation report ({s['clock']}, rainfall {s['rain']} mm/h)", "=" * 60, summ, "", "Crew plan (2 crews):"]
    L += [f"Crew {c['id']}: " + " -> ".join(f"{t['name']} (+{t['arrive']} min)" for t in c["stops"]) for c in pl["crews"]] or ["No cleaning needed."]
    L += ["", "Roads at risk:"] + [f"- {r['road']}: {r['level']} ({r['flood']}/100)" for r in s["roads"] if r["level"] != "Low"] + ["", "Alerts:"] + [f"[{a['time']}] {a['msg']}" for a in s["alerts"]]
    return Response("\n".join(L), mimetype="text/plain", headers={"Content-Disposition": "attachment; filename=edrainmind_report.txt"})

reset()
if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, debug=False)
