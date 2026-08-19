/**
 * frontend/js/api.js
 * ==================
 * Centralized REST API Client for ALLClear Cloud Removal System.
 */

const API_BASE = window.location.origin;

export const Api = {
  /**
   * Check API health and backend system status.
   */
  async getHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
      return await res.json();
    } catch (err) {
      console.error('[API] Health check error:', err);
      return { status: 'offline', error: err.message };
    }
  },

  /**
   * Retrieve satellite scenes with optional eligibility and pagination filtering.
   */
  async getScenes(eligible = null, limit = 100, offset = 0) {
    try {
      let url = `${API_BASE}/scenes?limit=${limit}&offset=${offset}`;
      if (eligible !== null) {
        url += `&eligible=${eligible}`;
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Failed to load scenes: ${res.statusText}`);
      return await res.json();
    } catch (err) {
      console.error('[API] getScenes error:', err);
      throw err;
    }
  },

  /**
   * Retrieve detailed metadata for a single scene.
   */
  async getScene(sceneId) {
    try {
      const res = await fetch(`${API_BASE}/scenes/${encodeURIComponent(sceneId)}`);
      if (!res.ok) throw new Error(`Scene '${sceneId}' not found`);
      return await res.json();
    } catch (err) {
      console.error('[API] getScene error:', err);
      throw err;
    }
  },

  /**
   * Trigger cloud removal inference on an eligible scene.
   */
  async runInference(sceneId, tileSize = 256, overlap = 64, batchSize = 4) {
    try {
      const res = await fetch(`${API_BASE}/inference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scene_id: sceneId,
          tile_size: tileSize,
          overlap: overlap,
          batch_size: batchSize,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Inference error (${res.status})`);
      }
      return await res.json();
    } catch (err) {
      console.error('[API] runInference error:', err);
      throw err;
    }
  },

  /**
   * Retrieve reconstruction result metadata by Job ID or Result ID.
   */
  async getResult(resultOrJobId) {
    try {
      const res = await fetch(`${API_BASE}/results/${encodeURIComponent(resultOrJobId)}`);
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Result error (${res.status})`);
      }
      return await res.json();
    } catch (err) {
      console.error('[API] getResult error:', err);
      throw err;
    }
  },

  /**
   * Retrieve quantitative metrics (PSNR, SSIM, MAE, RMSE) or aggregate benchmarks.
   */
  async getMetrics(resultId = null) {
    try {
      let url = `${API_BASE}/metrics`;
      if (resultId) {
        url += `?result_id=${encodeURIComponent(resultId)}`;
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Metrics request failed: ${res.statusText}`);
      return await res.json();
    } catch (err) {
      console.error('[API] getMetrics error:', err);
      throw err;
    }
  },

  /**
   * Inspect active DSen2-CR model specifications and checkpoint status.
   */
  async getModelInfo() {
    try {
      const res = await fetch(`${API_BASE}/models`);
      if (!res.ok) throw new Error(`Model info failed: ${res.statusText}`);
      return await res.json();
    } catch (err) {
      console.error('[API] getModelInfo error:', err);
      throw err;
    }
  },

  /**
   * Helper URL generators for image previews and downloads.
   */
  getScenePreviewUrl(sceneId, modality = 's2') {
    return `${API_BASE}/scenes/${encodeURIComponent(sceneId)}/preview/${modality}?t=${Date.now()}`;
  },

  getResultPreviewUrl(resultId, modality = 'reconstructed') {
    return `${API_BASE}/results/${encodeURIComponent(resultId)}/preview/${modality}?t=${Date.now()}`;
  },

  getDownloadUrl(resultId, fileType = 'geotiff') {
    return `${API_BASE}/download?result_id=${encodeURIComponent(resultId)}&file_type=${fileType}`;
  },
};
