import { getAuthToken } from "./src/lib/auth";

export const BACKEND_PORT = 3001;
export const FRONTEND_PORT = 8086;

const defaultApiBase = (() => {
  if (typeof window === "undefined") {
    return `http://localhost:${BACKEND_PORT}`;
  }

  const { protocol, hostname } = window.location;
  const resolvedHost = hostname && hostname !== "0.0.0.0" ? hostname : "localhost";
  return `${protocol}//${resolvedHost}:${BACKEND_PORT}`;
})();

export const API_BASE = import.meta.env.VITE_API_BASE_URL || defaultApiBase;
const USE_MOCK_API = String(import.meta.env.VITE_USE_MOCK_API || "false") === "true";

async function buildAuthHeaders(base = {}) {
  const token = await getAuthToken();
  if (!token) return base;
  return {
    ...base,
    Authorization: `Bearer ${token}`,
  };
}

async function parseJsonResponse(res, label) {
  let data = null;
  try {
    data = await res.json();
  } catch (error) {
    console.error(`Backend did not return valid JSON for ${label}`, error);
    throw new Error("Invalid response from backend");
  }

  if (!res.ok || data?.error) {
    console.error(`Backend returned error for ${label}`, res.status, data);
    throw new Error(data?.error || `Backend error: ${res.status}`);
  }

  return data;
}

async function apiRequest(path, options = {}) {
  const headers = await buildAuthHeaders(options.headers || {});
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: options.credentials || "include",
  });
  return parseJsonResponse(res, path);
}

export async function fetchHealth() {
  return apiRequest("/api/health", { method: "GET" });
}

export async function fetchModelInfo() {
  try {
    return await apiRequest("/api/model-info", { method: "GET" });
  } catch (error) {
    console.warn("Fetching model info failed:", error);
    return {
      status: "not_loaded",
      message: "Model information unavailable",
    };
  }
}

export async function fetchUserProfile() {
  return apiRequest("/api/auth/profile", { method: "GET" });
}

export async function updateUserProfile(payload) {
  return apiRequest("/api/auth/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
}

export async function fetchDatabaseLogs(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    query.set(key, String(value));
  });

  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiRequest(`/api/database/logs${suffix}`, { method: "GET" });
}

export async function uploadImage(file) {
  const MAX_SIZE = 16 * 1024 * 1024;
  if (file.size > MAX_SIZE) {
    throw new Error(`File size exceeds maximum limit of 16MB. File is ${(file.size / 1024 / 1024).toFixed(1)}MB`);
  }

  const allowedImageTypes = ["image/png", "image/jpeg", "image/gif", "image/webp"];
  const allowedVideoTypes = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska", "video/webm"];
  const allowedTypes = [...allowedImageTypes, ...allowedVideoTypes];

  if (!allowedTypes.includes(file.type)) {
    throw new Error(`Invalid file type: ${file.type}. Allowed: images (PNG, JPG, GIF, WebP) and videos (MP4, MOV, AVI, MKV, WebM)`);
  }

  const formData = new FormData();
  formData.append("image", file);

  try {
    const headers = await buildAuthHeaders();
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      headers,
      body: formData,
      credentials: "include",
    });
    return await parseJsonResponse(res, "/api/upload");
  } catch (error) {
    if (USE_MOCK_API) {
      console.warn("Upload to backend failed, falling back to mock response:", error);
      return mockUploadResponse();
    }
    throw error;
  }
}

export async function fetchDetectionLogs(params = {}) {
  try {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") return;
      query.set(key, String(value));
    });

    const suffix = query.toString() ? `?${query.toString()}` : "";
    const data = await apiRequest(`/api/logs${suffix}`, { method: "GET" });

    if (Array.isArray(data)) {
      return {
        items: data,
        total: data.length,
        page: 1,
        page_size: data.length,
      };
    }
    return data;
  } catch (error) {
    if (USE_MOCK_API) {
      console.warn("Fetching detection logs failed, falling back to mock logs:", error);
      const items = mockLogsResponse();
      return {
        items,
        total: items.length,
        page: 1,
        page_size: items.length,
      };
    }
    throw error;
  }
}

export async function deleteDetectionLog(logId) {
  return apiRequest(`/api/logs/${encodeURIComponent(logId)}`, {
    method: "DELETE",
  });
}

export async function clearDetectionLogs(sourceType = "") {
  const query = sourceType ? `?source_type=${encodeURIComponent(sourceType)}` : "";
  return apiRequest(`/api/logs${query}`, {
    method: "DELETE",
  });
}

export async function logLiveEvent(payload) {
  return apiRequest("/api/live-events", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
}

export async function fetchStats() {
  try {
    return await apiRequest("/api/stats", { method: "GET" });
  } catch (error) {
    console.warn("Fetching stats failed, using empty defaults:", error);
    return null;
  }
}

function mockUploadResponse() {
  return new Promise((resolve) => {
    setTimeout(() => {
      const isFake = Math.random() > 0.5;
      const confidence = 70 + Math.random() * 30;

      resolve({
        prediction: isFake ? "Fake" : "Real",
        confidence,
        filename: "mock_upload.jpg",
        isVideo: false,
        threat_level: confidence > 80 ? "high" : confidence > 50 ? "medium" : "low",
        model_used: "Mock Model (Backend Unavailable)",
        processing_time: {
          preprocessing_ms: 10 + Math.random() * 20,
          inference_ms: 50 + Math.random() * 100,
          total_ms: 60 + Math.random() * 120,
        },
        analysis: {
          level: "Mock",
          description: "Backend unavailable. Using mock prediction.",
          recommendation: "Please ensure backend server is running for accurate results.",
        },
        model_info: {
          architecture: "N/A",
          input_size: "N/A",
          framework: "Mock",
          device: "cpu",
        },
      });
    }, 800);
  });
}

function mockLogsResponse() {
  const now = new Date();
  const entries = [];

  for (let i = 0; i < 12; i++) {
    const ts = new Date(now.getTime() - i * 5 * 60_000).toISOString();
    const isFake = Math.random() > 0.6;
    const confidence = 0.6 + Math.random() * 0.35;

    entries.push({
      timestamp: ts,
      filename: `mock_upload_${String(i + 1).padStart(3, "0")}.jpg`,
      prediction: isFake ? "Fake" : "Real",
      confidence,
    });
  }

  return entries;
}
