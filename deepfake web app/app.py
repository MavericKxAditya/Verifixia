"""Verifixia – Deepfake Detection API (Flask)

Pure JSON API backend consumed by the React frontend (Frontend/).
No HTML templates — the Vite dev-server serves the UI.

Run:
    python app.py              (port 3001 by default)
    run_app.bat / run_app.ps1

Frontend expects:
    POST /api/upload    (form-field: "image")
    GET  /api/health
"""

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2  # noqa: F401 – validate OpenCV is importable at startup
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from inference import (
    INPUT_DIR,
    OUTPUT_DIR,
    DeepfakePredictor,
    ensure_app_dirs,
    is_supported_file,
)

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask App ───────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250 MB

# CORS – allow common local frontend origins (Vite dev-server)
CORS(
    app,
    origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8086",
        "http://127.0.0.1:8086",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
)

# ── Directories & Model ────────────────────────────────────────────
ensure_app_dirs()

predictor = None
_model_error: str | None = None
try:
    predictor = DeepfakePredictor()
    logger.info("✓ DeepfakePredictor loaded  (model: %s)", predictor.model_path)
except FileNotFoundError as exc:
    _model_error = str(exc)
    logger.warning("⚠ Model not found – prediction endpoints will return an error.  %s", exc)
except Exception as exc:  # noqa: BLE001
    _model_error = str(exc)
    logger.error("✗ Failed to initialise predictor: %s", exc, exc_info=True)


# ── Helpers ─────────────────────────────────────────────────────────

def _save_upload(uploaded_file) -> Path:
    """Save an uploaded file with a unique timestamped name and return the path."""
    filename = secure_filename(uploaded_file.filename)
    unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}_{filename}"
    dest = INPUT_DIR / unique_name
    uploaded_file.save(dest)
    logger.info("Saved upload → %s", dest)
    return dest


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Root – returns API info (no HTML)."""
    return jsonify({
        "message": "Verifixia Deepfake Detection API",
        "version": "1.0.0",
        "model_loaded": predictor is not None,
        "endpoints": {
            "POST /api/upload": "Upload image/video for deepfake detection (field: 'image')",
            "GET  /api/health": "Health check",
        },
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Analyse an uploaded image or video and return JSON results.

    The React frontend sends the file under the form-field name "image".
    We also accept "file" as a fallback for other clients.
    """
    if predictor is None:
        return jsonify({"error": f"Model unavailable: {_model_error}"}), 503

    # Accept both "image" (frontend) and "file" (curl / other scripts)
    uploaded_file = request.files.get("image") or request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "No file provided. Use form-field 'image' or 'file'."}), 400

    if not is_supported_file(uploaded_file.filename):
        return jsonify({
            "error": "Unsupported file type. Allowed: png, jpg, jpeg, gif, bmp, webp, mp4, avi, mov, mkv, webm"
        }), 400

    start = time.perf_counter()
    input_path = _save_upload(uploaded_file)
    result = predictor.predict_file(input_path)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    is_video = result.media_type == "video"

    # Determine threat level from confidence
    if result.confidence > 70:
        threat_level = "high"
    elif result.confidence > 40:
        threat_level = "medium"
    else:
        threat_level = "low"

    logger.info(
        "Prediction: %s | confidence=%.2f%% | threat=%s | time=%.0fms | file=%s",
        result.prediction, result.confidence, threat_level, elapsed_ms, result.filename,
    )

    # Return JSON matching the contract the Frontend expects
    return jsonify({
        "prediction": result.prediction,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "filename": result.filename,
        "isVideo": is_video,
        "threat_level": threat_level,
        "model_used": "Verifixia CNN-LSTM (EfficientNet-B0)",
        "processing_time": {
            "preprocessing_ms": 0,
            "inference_ms": elapsed_ms,
            "total_ms": elapsed_ms,
        },
        "analysis": {
            "level": "Deep Learning",
            "description": "CNN-LSTM with EfficientNet-B0 backbone trained on deepfake dataset",
            "recommendation": (
                "Content flagged as potentially manipulated"
                if result.prediction == "Fake"
                else "Content appears authentic"
            ),
        },
        "model_info": {
            "architecture": "EfficientNet-B0 + LSTM",
            "input_size": "224x224",
            "framework": "PyTorch",
            "device": str(predictor.device),
            "version": "1.2.0-unified",
            "total_parameters": 5300000,
        },
        "session_id": uuid4().hex,
        "log_id": uuid4().hex,
    })


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Retrieve paginated detection logs from saved JSON results."""
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 20))
    start_date_str = request.args.get("start_date")
    end_date_str = request.args.get("end_date")
    source_type = request.args.get("source_type")

    logs = []
    if not OUTPUT_DIR.exists():
        return jsonify({"items": [], "total": 0})

    for log_file in OUTPUT_DIR.glob("*.json"):
        if log_file.name == ".gitkeep":
            continue
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
                
                # Ensure we have an ID for deletion/UI
                if "id" not in log_data:
                    log_data["id"] = log_file.stem
                
                # Filter by source_type (backend uses 'media_type')
                if source_type and log_data.get("media_type") != source_type:
                    continue
                
                # Filter by date (processed_at)
                processed_at = log_data.get("processed_at", "")
                if start_date_str and processed_at < start_date_str:
                    continue
                if end_date_str and processed_at > end_date_str:
                    continue
                
                # Map to what Frontend ForensicLogs.tsx expects
                if "timestamp" not in log_data:
                    log_data["timestamp"] = log_data.get("processed_at")
                
                logs.append(log_data)
        except Exception as exc:
            logger.error("Failed to read log %s: %s", log_file, exc)

    # Sort by timestamp descending (newest first)
    logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    total = len(logs)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_items = logs[start_idx:end_idx]

    return jsonify({
        "items": paginated_items,
        "total": total,
        "page": page,
        "page_size": page_size
    })


@app.route("/api/logs/<log_id>", methods=["DELETE"])
def delete_log(log_id):
    """Delete a specific log file by its ID/filename."""
    found = False
    for log_file in OUTPUT_DIR.glob("*.json"):
        if log_id in log_file.name:
            try:
                log_file.unlink()
                found = True
                break
            except Exception as exc:
                return jsonify({"error": f"Failed to delete file: {exc}"}), 500
    
    if found:
        return jsonify({"status": "success", "message": f"Log {log_id} deleted"})
    return jsonify({"error": "Log not found"}), 404


@app.route("/api/logs", methods=["DELETE"])
def clear_logs():
    """Clear all logs, optionally filtered by source type."""
    source_type = request.args.get("source_type")
    count = 0
    for log_file in OUTPUT_DIR.glob("*.json"):
        if log_file.name == ".gitkeep":
            continue
        
        should_delete = True
        if source_type:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("media_type") != source_type:
                        should_delete = False
            except:  # noqa: E722
                should_delete = False
        
        if should_delete:
            try:
                log_file.unlink()
                count += 1
            except Exception as exc:
                logger.warning("Could not delete %s: %s", log_file, exc)
            
    return jsonify({"status": "success", "message": f"Deleted {count} logs"})


@app.route("/api/live-events", methods=["POST"])
def log_live_event():
    """Receive and persist a live monitoring event."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    event_id = uuid4().hex
    ts = datetime.now().isoformat()
    
    # Construct a log entry that matches the structure of file-based results
    log_entry = {
        "id": f"live_{event_id}",
        "timestamp": ts,
        "processed_at": ts,
        "filename": data.get("filename", "Live Feed Capture"),
        "prediction": data.get("prediction", "Unknown"),
        "confidence": data.get("confidence", 0),
        "media_type": "live",
        "threat_level": data.get("threat_level", "low"),
        "session_id": data.get("session_id"),
        "model_used": data.get("model_used", "Verifixia Live Monitor"),
        "latency_ms": data.get("processing_time_ms", 0),
        "probabilities": data.get("probabilities", {})
    }
    
    log_file = OUTPUT_DIR / f"live_{event_id}.json"
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2)
    except Exception as exc:
        return jsonify({"error": f"Failed to save live event: {exc}"}), 500
        
    return jsonify({"status": "success", "log_id": event_id})


@app.route("/api/health", methods=["GET"])
def health():
    model_loaded = predictor is not None
    return jsonify({
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": str(predictor.model_path) if predictor else None,
        "model_error": _model_error,
        "opencv_version": cv2.__version__,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "info": {
            "architecture": "EfficientNet-B0 + LSTM",
            "input_size": "224x224",
            "framework": "PyTorch",
            "device": str(predictor.device) if predictor else "CPU",
            "version": "1.2.0-unified",
            "total_parameters": 5300000,
            "status": "loaded" if model_loaded else "not_found"
        }
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return detailed technical specifications of the loaded model."""
    model_loaded = predictor is not None
    return jsonify({
        "status": "loaded" if model_loaded else "not_found",
        "model_name": "Verifixia CNN-LSTM",
        "version": "1.2.0-unified",
        "architecture": "EfficientNet-B0 + LSTM",
        "framework": "PyTorch",
        "device": str(predictor.device) if predictor else "CPU",
        "total_parameters": 5300000,
        "input_size": "224x224",
        "training_split": "70/20/10",
        "activation": "ReLU",
    })


# ── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    logger.info("Starting Verifixia API on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
