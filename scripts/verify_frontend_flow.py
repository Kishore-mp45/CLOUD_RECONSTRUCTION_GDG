"""
scripts/verify_frontend_flow.py
===============================
Simulate and verify the complete Phase 10 frontend demo flow against the live server.
"""

import json
import urllib.request

base = "http://127.0.0.1:8000"

print("============================================================")
print("PHASE 10: PROFESSIONAL FRONTEND DEMO FLOW VERIFICATION")
print("============================================================")

print("\n1. [HEALTH CHECK] Querying GET /health...")
with urllib.request.urlopen(f"{base}/health") as res:
    health = json.loads(res.read().decode())
    print(f"   -> Status: {health['status']} | GPU: {health.get('gpu_name')} | Model Checkpoint: {health.get('model_checkpoint_available')}")

print("\n2. [SCENE SELECTION] Fetching scenes from GET /scenes?limit=10...")
with urllib.request.urlopen(f"{base}/scenes?limit=10") as res:
    scenes_data = json.loads(res.read().decode())
    total = scenes_data["total_count"]
    eligible = scenes_data["eligible_count"]
    print(f"   -> Total Scenes: {total} | Eligible Scenes: {eligible}")
    eligible_scene = next(s for s in scenes_data["scenes"] if s["is_eligible"])
    scene_id = eligible_scene["scene_id"]
    print(f"   -> Selected Eligible Scene: {scene_id} ({eligible_scene['cloud_density_percent']:.1f}% cloud cover)")

print(f"\n3. [IMAGE VIEWER] Fetching Cloudy S2 Preview from GET /scenes/{scene_id}/preview/s2...")
with urllib.request.urlopen(f"{base}/scenes/{scene_id}/preview/s2") as res:
    data = res.read()
    print(f"   -> HTTP {res.status} | Content-Type: {res.headers['content-type']} | Size: {len(data)} bytes")

print(f"\n4. [SAR VIEWER] Fetching Sentinel-1 SAR Preview from GET /scenes/{scene_id}/preview/s1...")
with urllib.request.urlopen(f"{base}/scenes/{scene_id}/preview/s1") as res:
    data = res.read()
    print(f"   -> HTTP {res.status} | Content-Type: {res.headers['content-type']} | Size: {len(data)} bytes")

print(f"\n5. [INFERENCE EXECUTION] Triggering POST /inference on {scene_id}...")
payload = json.dumps({"scene_id": scene_id, "tile_size": 256, "overlap": 64, "batch_size": 4}).encode("utf-8")
req = urllib.request.Request(f"{base}/inference", data=payload, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as res:
    job = json.loads(res.read().decode())
    job_id = job["job_id"]
    print(f"   -> Job Dispatched: {job_id} | Initial Status: {job['status']}")

print(f"\n6. [RESULT RETRIEVAL] Polling GET /results/{job_id}...")
with urllib.request.urlopen(f"{base}/results/{job_id}") as res:
    result = json.loads(res.read().decode())
    print(f"   -> Result ID: {result['result_id']} | Status: {result['status']} | Output Bands: {result['band_count']} | Inference Time: {result['inference_time_s']:.2f}s")

print(f"\n7. [METRICS INSPECTION] Fetching GET /metrics?result_id={result['result_id']}...")
with urllib.request.urlopen(f"{base}/metrics?result_id={result['result_id']}") as res:
    metrics = json.loads(res.read().decode())
    print(f"   -> Reference Metrics Available: {metrics.get('available')} | Note: {metrics.get('reason')}")

print("\n8. [MODEL INSPECTION] Querying GET /models...")
with urllib.request.urlopen(f"{base}/models") as res:
    models = json.loads(res.read().decode())
    print(f"   -> Architecture: {models['architecture']} | Parameters: {models['parameter_count']} | Inputs: {', '.join(models['input_modalities'])}")

print("\n9. [STATIC ASSETS] Testing GET / (index.html), /css/styles.css, /js/app.js...")
for path in ["/", "/css/styles.css", "/css/dashboard.css", "/css/components.css", "/js/app.js", "/js/viewer.js", "/js/scenes.js", "/js/inference.js", "/js/metrics.js", "/js/ui.js", "/js/api.js"]:
    with urllib.request.urlopen(f"{base}{path}") as res:
        print(f"   -> {path} : HTTP {res.status}")

print("\n============================================================")
print("PHASE 10 FRONTEND DEMO FLOW: ALL CHECKS OPERATIONAL (PASS)")
print("============================================================")
