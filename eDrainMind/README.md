# eDrainMind AI v4 - Drainage blockage, flood prediction and Smart Emergency Routes

## Run
```
pip install -r requirements.txt      # torch is large; skip it if you only want a quick demo
python app.py                         # open http://127.0.0.1:5000  (start-up training takes 15-60 s)
```

## NEW: Smart Emergency Route Recommendation (tab "Emergency routes")
Pick a start and destination (search a place, press "Map" and click, or quick-pick a real hospital). The AI finds several REAL road routes and recommends the safest one.

* **Candidates:** the alternative routes the OpenStreetMap routing service (OSRM) offers, plus detours forced through side waypoints, so a real alternative exists when the main road floods.
* **Flood risk every ~100 m** (Safe / Moderate / High / Critical) from: flood risk of nearby monitored drains (this app's models), how low the ground is (SRTM height), distance to real rivers/canals/drains, underpasses (OpenStreetMap tags), current rain, and closures you report.
* **Route score** = 30% distance + 40% flood risk + 20% road condition + 10% travel time (each part 0-1, lower is better). These are starting values, not optimal ones - change them with the sliders. Ambulance / rescue mode starts at 10 / 40 / 10 / 40.
* **Recommendation:** a route with a closed or Critical stretch is "Avoid"; High is avoided if any Safe/Moderate route exists; otherwise the best weighted score is "Recommended". If no route is safe the screen says so and shows the least-risk option.
* **Road closure detection:** use "Report a road problem" (closed / flooded / blocked) and click the road. Reports are saved in `data/incidents.json` and used immediately; OpenStreetMap closed/construction tags are used too.
* **Real-time monitoring:** change the rain, press +10 min, add an incident or clean a drain - routes are re-scored instantly (and every 20 s while the tab is open). The forecast window shows route risk now, in 30 and in 60 minutes.
* **Map view:** dark / satellite / street map; the recommended route glows cyan, other routes are dashed in their risk colour, the chosen route's risky stretches are painted orange/red with X markers, drains in flood alert are circled.

Data used: OSM routing (OSRM), Overpass (roads, waterways, hospitals), Open-Meteo elevation, Nominatim place search - all free, no key. Cached in `data/` (`hospitals.json`, `incidents.json`).

**Honest limits:** drain readings are still simulated until real sensors are connected; monitored drains cover only the central area, so far from them risk comes from ground height, waterways and underpasses; there is no free live-traffic feed, so times are typical road speeds (flooded stretches are slowed by an assumed factor - see `SLOW` in `routes.py`); the risk model is an estimate to support decisions, not a guarantee that a road is passable.

## Real Palakkad data (v3.1)
On the first start (internet needed, no key) the app downloads and caches in `data/`:
| What | Source | File |
|---|---|---|
| Real places used as drains (Stadium Bus Stand, KSRTC terminal, Fort, Junction station, hospital, colleges ...) | OpenStreetMap / Overpass | `data/palakkad_network.json` |
| Ground height of each drain; water flows to the nearest LOWER drain | Open-Meteo elevation (SRTM) | same file |
| 2 years of hourly Palakkad rainfall, used to train the rain/water-level forecaster | Open-Meteo archive | `data/rainfall.csv` |
| Real drains, canals and streams drawn on the map | OpenStreetMap | `data/osm_water.json` |
| Live 2-day rain forecast | Open-Meteo | (live) |
The map is a real Leaflet map: satellite (Esri) or street (OpenStreetMap), switch at the top right.
To refresh the data, delete the files in `data/` (keep `README.txt`) and restart. Offline, the app still starts with an approximate demo layout and says "OFFLINE DEMO" under the map.

**Honest note:** the places, heights, rain history and waterways are real. The water-level sensors, blockage labels and waste levels are still simulated because no public live drain-sensor feed exists for Palakkad. Drain positions are landmarks/junctions, not surveyed drain covers; replace `data/palakkad_network.json` with your municipality's drain list to use real asset positions.

## What is new in v3
- Fixed: the "Ask the assistant" box can be typed in (the key handler was cancelling every key except Enter).
- Realistic city map: asphalt roads, buildings, trees, parks, a canal, animated drain pipes and flood puddles that grow with the water level.

## What is new in v2
| Feature | Where | Needs setup? |
|---|---|---|
| Real LSTM (water level) and CNN (drain photos) in PyTorch | `ml.py` | `pip install torch` (auto-detected, falls back if missing) |
| Free AI assistant: chat box + "Explain with AI" + AI report summary | `app.py` `llm()` | Put a free key in `.env` (see `.env.example`) |
| Model report page: accuracy, precision, recall, F1, confusion matrix, feature importance | "Model report" tab | none |
| Real datasets: drop CSVs into `data/` (see `data/README.txt`) | `ml.py` | optional |
| Live rain forecast (Open-Meteo, free, no key) + "what if this rain arrives?" | Rain forecast card | internet |
| Email alerts when drains turn High (+ test button) | `send_email()` | Gmail app password in `.env` |
| Time-to-overflow ("overflows in about 25 min") | everywhere | none |
| Crew scheduler: 1-3 crews, urgent drains first, routes on the map, dispatch button | Crew plan card | none |

## Setup for email, AI and weather
Copy `.env.example` to `.env` and fill in what you want. Everything is optional; unconfigured features show "off" and the rest still works.
- Email: Gmail > Security > 2-step verification > App passwords.
- AI: free key from aistudio.google.com (Gemini) or console.groq.com; any OpenAI-compatible API works. Check the current free model name and set `LLM_MODEL`.
- Weather location: `WEATHER_LAT` / `WEATHER_LON`.

## Honest notes for your viva
- Sensors and training data are synthetic unless you add CSVs to `data/`. Water-level response uses a simple drain model.
- GNN is a 2-layer message-passing function with fixed weights (not trained). Time-to-overflow comes from the LSTM forecast crossing 80%.
- The crew scheduler is a greedy heuristic (most urgent drain first, assigned to the crew that can arrive soonest), not a proven-optimal route.
- Blockage accuracy on synthetic data is about 0.76 because labels are deliberately noisy.

## Files
`app.py` server, simulation, email, AI, planner | `ml.py` models and metrics | `templates/index.html` UI | `data/` optional CSVs | `.env.example`

## Troubleshooting
- **Email failed / connection closed:** the app tries Gmail port 587 and then 465. If both fail, your antivirus or Wi-Fi is blocking email: pause the antivirus "email shield" or use a mobile hotspot. App password spaces are removed automatically.
- **AI shows "Built-in answer - AI not used: ...":** the text after the colon is the real reason (wrong key, wrong model name, quota). Fix `LLM_API_KEY` / `LLM_MODEL` in `.env` and restart.
- **Photo check:** "Photo check" card (right column). Pick the drain, then upload or drag a photo, or press a sample button.
