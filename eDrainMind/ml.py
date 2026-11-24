"""eDrainMind AI - ML core. Uses PyTorch (LSTM + CNN) when installed, otherwise lightweight scikit-learn fallbacks."""
import os, numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
try:
    from xgboost import XGBClassifier
    def _gb(): return XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, eval_metric="logloss"), "XGBoost"
except Exception:
    def _gb(): return GradientBoostingClassifier(n_estimators=150, max_depth=3), "Gradient Boosting (XGBoost not installed)"

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"); M = {}   # M = evaluation metrics shown on the Model report page
FEATS = ["Rainfall", "Water level", "Obstruction", "Past blockages", "Drain age"]
def _split(n, frac=.8, seed=0): p = np.random.default_rng(seed).permutation(n); c = int(n * frac); return p[:c], p[c:]
def _csv(name):
    p = os.path.join(DATA, name); return np.genfromtxt(p, delimiter=",", names=True) if os.path.exists(p) else None

if HAS_TORCH:
    class LSTMNet(nn.Module):
        def __init__(s): super().__init__(); s.l = nn.LSTM(2, 32, batch_first=True); s.o = nn.Linear(32, 1)
        def forward(s, x): h, _ = s.l(x); return s.o(h[:, -1]).squeeze(-1)
    class CNNNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Flatten(), nn.Linear(16 * 14 * 14, 32), nn.ReLU(), nn.Linear(32, 2))
        def forward(s, x): return s.f(x)
    def _fit(net, X, y, loss_fn, epochs, lr=.005, bs=64):
        torch.manual_seed(0); opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            for idx in torch.randperm(len(X)).split(bs): opt.zero_grad(); loss_fn(net(X[idx]), y[idx]).backward(); opt.step()
        net.eval()

# ---------- 1. Blockage prediction (XGBoost) ----------
def train_blockage():
    a = _csv("drain_data.csv"); src = "Synthetic data"
    if a is not None:   # columns: rain,water,waste,past,age,blocked
        X = np.column_stack([a[k] for k in ("rain", "water", "waste", "past", "age")]); y = a["blocked"].astype(int); src = "Your file data/drain_data.csv"
    else:
        r = np.random.default_rng(0); n = 4000
        rain = r.gamma(2, 10, n).clip(0, 120); wl = r.uniform(5, 100, n); waste = r.beta(2, 4, n); past = r.poisson(2, n).astype(float); age = r.uniform(0, 30, n)
        X = np.c_[rain, wl, waste, past, age]; z = -5.2 + .02 * rain + .015 * wl + 5 * waste + .35 * past + .03 * age
        y = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    tr, te = _split(len(X)); m, name = _gb(); m.fit(X[tr], y[tr]); pr = m.predict(X[te])
    M["blockage"] = dict(model=name, data=src, n=len(X), accuracy=round(accuracy_score(y[te], pr), 3), precision=round(precision_score(y[te], pr, zero_division=0), 3),
                         recall=round(recall_score(y[te], pr, zero_division=0), 3), f1=round(f1_score(y[te], pr, zero_division=0), 3),
                         matrix=confusion_matrix(y[te], pr, labels=[0, 1]).tolist(), importance={k: round(float(v), 3) for k, v in zip(FEATS, m.feature_importances_)})
    return m

# ---------- 2. Computer vision (CNN) ----------
def make_image(blocked, seed=0, size=256):
    """Synthetic drain-grate photo for training and the demo sample buttons."""
    r = np.random.default_rng(seed); b = int(r.integers(80, 130))
    img = Image.new("RGB", (size, size), (b, b, b + int(r.integers(0, 10)))); d = ImageDraw.Draw(img)
    for x in range(20, size, 28): d.rectangle([x, 30, x + 12, size - 30], fill=(25, 25, 30))
    for _ in range(int(r.integers(3, 9))):
        x, y = (int(v) for v in r.integers(0, size, 2)); s = int(r.integers(4, 10)); d.ellipse([x, y, x + s, y + s], fill=(90, 70, 45))
    if blocked:
        pal = [(40, 110, 220), (220, 50, 50), (235, 235, 235), (240, 200, 40), (60, 170, 90), (255, 120, 30)]
        for _ in range(int(r.integers(8, 18))):
            x, y = (int(v) for v in r.integers(10, size - 60, 2)); w, h = (int(v) for v in r.integers(25, 70, 2))
            d.polygon([(x, y), (x + w, y + h // 5), (x + int(w * .8), y + h), (x + w // 10, y + int(h * .7))], fill=pal[int(r.integers(len(pal)))])
    a = np.asarray(img.filter(ImageFilter.GaussianBlur(.8)), float) + r.normal(0, 5, (size, size, 3))
    return Image.fromarray(a.clip(0, 255).astype("uint8"))

def _hsv(img, n=128):
    a = np.asarray(img.convert("RGB").resize((n, n)).convert("HSV"), float); return a[..., 0] * 360 / 255, a[..., 1] / 255, a[..., 2] / 255   # hue deg, sat, value
def looks_like_sample(img):
    """The built-in CNN was trained on grey drain-grate pictures only. True if this photo resembles those (mostly grey, low colour)."""
    h, s_, v = _hsv(img); return float((s_ < .18).mean()) > .6
def rough_waste(img):
    """Very rough colour-based estimate for REAL photos when no AI key is set: counts plastic-like colours (saturated blue/red/yellow/orange, bright white)
    and ignores green plants, brown earth and dark water. Returns (score 0-1, list of colours seen). Low confidence - an LLM vision key is far better."""
    h, s_, v = _hsv(img); n = h.size; seen = []
    green = (h > 70) & (h < 170) & (s_ > .15)
    blue = (h >= 190) & (h <= 260) & (s_ > .45) & (v > .35); red = ((h < 12) | (h > 340)) & (s_ > .55) & (v > .35)
    yellow = (h >= 40) & (h <= 65) & (s_ > .6) & (v > .55); orange = (h >= 15) & (h < 40) & (s_ > .7) & (v > .6); white = (s_ < .1) & (v > .9)
    m = (blue | red | yellow | orange | white) & ~green
    for name, k in (("blue plastic", blue), ("red plastic", red), ("yellow items", yellow), ("orange items", orange), ("white bags/foam", white)):
        if (k & ~green).mean() > .02: seen.append(name)
    return float(np.clip(m.mean() * 2.5, 0, 1)), seen

_K = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]], [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [[0, 1, 0], [1, -4, 1], [0, 1, 0]]], float)
def cnn_features(img):   # fallback: fixed convolution filters -> ReLU -> max-pool
    a = np.asarray(img.convert("RGB").resize((64, 64)), float) / 255; g = a.mean(2); out = []
    for k in _K:
        c = sum(k[i, j] * g[i:i + 62, j:j + 62] for i in range(3) for j in range(3)); out.append(np.maximum(c, 0)[:60, :60].reshape(6, 10, 6, 10).max((1, 3)).ravel())
    out.append(np.histogram(a.max(2) - a.min(2), bins=8, range=(0, 1))[0] / 4096); out.append(a.mean((0, 1))); return np.concatenate(out)
def _arr(img): return np.asarray(img.convert("RGB").resize((64, 64)), np.float32).transpose(2, 0, 1) / 255

class Vision:
    def __init__(s, n=300):
        imgs = [make_image(b, k) for b in (0, 1) for k in range(n)]; y = np.array([0] * n + [1] * n); tr, te = _split(2 * n)
        if HAS_TORCH:
            X = torch.tensor(np.stack([_arr(i) for i in imgs])); yt = torch.tensor(y); s.m = CNNNet(); _fit(s.m, X[tr], yt[tr], nn.CrossEntropyLoss(), 10, .003, 32)
            with torch.no_grad(): pr = s.m(X[te]).argmax(1).numpy()
            s.name = "PyTorch CNN (2 conv layers)"
        else:
            X = np.array([cnn_features(i) for i in imgs]); s.m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0)); s.m.fit(X[tr], y[tr]); pr = s.m.predict(X[te])
            s.name = "Convolution filters + neural classifier (install PyTorch for a real CNN)"
        M["vision"] = dict(model=s.name, data="Synthetic drain photos", n=2 * n, accuracy=round(accuracy_score(y[te], pr), 3), matrix=confusion_matrix(y[te], pr, labels=[0, 1]).tolist())
    def score(s, img):
        if HAS_TORCH:
            with torch.no_grad(): return float(torch.softmax(s.m(torch.tensor(_arr(img))[None]), 1)[0, 1])
        return float(s.m.predict_proba(cnn_features(img)[None])[0, 1])

# ---------- 3. Water-level forecasting (LSTM) ----------
class Forecaster:
    def __init__(s, X, y):
        tr, te = _split(len(X), .85)
        if HAS_TORCH: s.m = LSTMNet(); _fit(s.m, s._seq(X[tr]), torch.tensor(y[tr], dtype=torch.float32), nn.MSELoss(), 15, .005, 128); s.name = "PyTorch LSTM (32 units)"
        else: s.m = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, random_state=0).fit(X[tr], y[tr]); s.name = "Neural net on 6-step windows (install PyTorch for an LSTM)"
        s.mae = round(float(np.abs(s.predict(X[te]) - y[te]).mean() * 100), 2)
    @staticmethod
    def _seq(X): return torch.tensor(np.stack([X[:, :6], X[:, 6:]], 2), dtype=torch.float32)
    def predict(s, X):
        if HAS_TORCH:
            with torch.no_grad(): return s.m(s._seq(X)).numpy()
        return s.m.predict(X)

def train_forecaster():
    r = np.random.default_rng(1); X, y = [], []; a = _csv("rainfall.csv"); pool = np.clip(np.nan_to_num(a["rain"]), 0, 100) if a is not None and len(a) >= 50 else None
    wet = np.flatnonzero(np.convolve(pool, np.ones(40), "valid") > 8) if pool is not None else []   # real 40-hour windows that actually contain rain
    for k in range(300):
        real = pool is not None and k % 2 == 0          # half the training runs replay real Palakkad rain, half are synthetic storms (so heavy rain is also learned)
        waste, w, rain = r.uniform(0, .8), r.uniform(0, 40), r.uniform(0, 60); ws, rs = [], []
        st = int(r.choice(wet) if len(wet) else r.integers(0, len(pool) - 40)) if real else 0
        for t in range(40):
            if real: rain = float(pool[st + t])
            else:
                if r.random() < .1: rain = r.uniform(0, 80)
                rain = max(0, rain * .97 + r.normal(0, 1.5))
            w = float(np.clip(.8 * w + .28 * rain * (1 + 1.2 * waste) + r.normal(0, 1), 0, 100)); ws.append(w); rs.append(rain)
        for t in range(6, 40): X.append(ws[t - 6:t] + rs[t - 6:t]); y.append(ws[t])
    f = Forecaster(np.array(X) / 100, np.array(y) / 100)
    M["forecaster"] = dict(model=f.name, data="Real Palakkad rainfall (data/rainfall.csv, %d hours) + synthetic storms" % len(pool) if pool is not None else "Synthetic rainfall", n=len(X), mae=f.mae); return f

def forecast(m, wh, rh, steps=6):
    w, r, out, rain = list(wh[-6:]), list(rh[-6:]), [], rh[-1]
    for _ in range(steps):
        nxt = float(np.clip(m.predict(np.array(w[-6:] + r[-6:])[None] / 100)[0] * 100, 0, 100)); rain *= .97; w.append(nxt); r.append(rain); out.append(round(nxt, 1))
    return out

# ---------- 4. Graph neural network ----------
def gnn(ids, edges, p, layers=2):
    """Message passing: a blockage downstream backs water up into upstream drains (.6); weaker spill downstream (.25)."""
    ix = {k: i for i, k in enumerate(ids)}; A = np.zeros((len(ids), len(ids)))
    for u, v in edges: A[ix[u], ix[v]] += .6; A[ix[v], ix[u]] += .25
    p = np.asarray(p, float); h = p.copy()
    for _ in range(layers): h = np.clip(p + A @ h, 0, 1)
    return h
