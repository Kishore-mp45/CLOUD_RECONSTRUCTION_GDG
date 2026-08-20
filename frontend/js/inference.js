/**
 * frontend/js/inference.js
 * ========================
 * Inference Execution State Machine, Live Timers, and Result Download Management.
 */

import { Api } from './api.js';
import { UI } from './ui.js';
import { Scenes } from './scenes.js';
import { Viewer } from './viewer.js';
import { Metrics } from './metrics.js';

export const Inference = {
  isProcessing: false,
  startTime: null,
  timerInterval: null,
  latestResult: null,

  /**
   * Initialize inference module and bind CTA buttons.
   */
  init() {
    this.bindEvents();
  },

  /**
   * Bind Run Inference button and download buttons.
   */
  bindEvents() {
    const runBtn = document.getElementById('run-inference-btn');
    const downloadGeotiffBtn = document.getElementById('btn-download-geotiff');
    const downloadPngBtn = document.getElementById('btn-download-png');
    const viewJsonBtn = document.getElementById('btn-view-json-meta');

    if (runBtn) {
      runBtn.addEventListener('click', () => {
        this.startInference();
      });
    }

    if (downloadGeotiffBtn) {
      downloadGeotiffBtn.addEventListener('click', () => {
        if (!this.latestResult) {
          UI.showToast('No completed reconstruction output available to download.', 'warning');
          return;
        }
        const url = Api.getDownloadUrl(this.latestResult.result_id, 'geotiff');
        window.open(url, '_blank');
      UI.showToast('GeoTIFF download started.', 'success');
      });
    }

    if (downloadPngBtn) {
      downloadPngBtn.addEventListener('click', () => {
        if (!this.latestResult) {
          UI.showToast('No completed reconstruction preview available to download.', 'warning');
          return;
        }
        const url = Api.getDownloadUrl(this.latestResult.result_id, 'png');
        window.open(url, '_blank');
      UI.showToast('Image preview download started.', 'success');
      });
    }

    if (viewJsonBtn) {
      viewJsonBtn.addEventListener('click', () => {
        if (!this.latestResult && !Scenes.selectedScene) {
          UI.showToast('No scene or result metadata available.', 'warning');
          return;
        }
        const title = this.latestResult
          ? `Inference Result: ${this.latestResult.result_id}`
          : `Scene Metadata: ${Scenes.selectedScene.scene_id}`;
        const data = this.latestResult || Scenes.selectedScene;
        UI.openModal(title, data);
      });
    }
  },

  /**
   * Start inference execution flow.
   */
  async startInference() {
    const scene = Scenes.selectedScene;
    if (!scene) {
      UI.showToast('Choose an image first.', 'warning');
      return;
    }

    if (this.isProcessing) return;

    this.isProcessing = true;
    this.updateRunButtonState('PROCESSING');
    Viewer.showProcessingOverlay('Creating your cloud-free reconstruction…');

    // Start live elapsed timer
    this.startTime = performance.now();
    this.startStopwatch();

    try {
      // 1. Dispatch POST /inference
      const jobResponse = await Api.runInference(scene.scene_id, 256, 64, 4);
      const jobId = jobResponse.job_id;

      // 2. Poll for completion
      const result = await this.pollResult(jobId);

      // 3. Handle Completion
      this.latestResult = result;
      this.stopStopwatch();

      Viewer.onInferenceCompleted(result);
      Metrics.onInferenceCompleted(result);

      this.updateRunButtonState('COMPLETED');
      this.enableDownloadButtons(true);

      UI.showToast(
        `Reconstruction complete in ${result.inference_time_s ? result.inference_time_s.toFixed(2) : '1.2'}s!`,
        'success'
      );
    } catch (err) {
      console.error('Inference failed:', err);
      this.stopStopwatch();
      Viewer.hideProcessingOverlay();
      this.updateRunButtonState('FAILED');
      UI.showToast(`Inference failed: ${err.message}`, 'error');
    } finally {
      this.isProcessing = false;
    }
  },

  /**
   * Poll GET /results/{id} until status is completed.
   */
  async pollResult(jobId, maxAttempts = 30, intervalMs = 1000) {
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const res = await Api.getResult(jobId);
        if (res && res.status === 'completed') {
          return res;
        }
      } catch (err) {
        // Continue polling if job in progress
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error('Inference timed out waiting for output file generation.');
  },

  /**
   * Start live UI stopwatch.
   */
  startStopwatch() {
    const timerEl = document.getElementById('stat-latency-val');
    if (!timerEl) return;

    this.timerInterval = setInterval(() => {
      const elapsed = ((performance.now() - this.startTime) / 1000).toFixed(2);
      timerEl.innerHTML = `${elapsed} <span class="stat-unit">s (live)</span>`;
    }, 100);
  },

  /**
   * Stop UI stopwatch.
   */
  stopStopwatch() {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
  },

  /**
   * Update the CTA button text, icon, and disabled status.
   */
  updateRunButtonState(state) {
    const btn = document.getElementById('run-inference-btn');
    if (!btn) return;

    if (state === 'PROCESSING') {
      btn.disabled = true;
      btn.innerHTML = `
        <div class="status-dot busy" style="display: inline-block;"></div>
        Reconstructing Scene...
      `;
    } else if (state === 'COMPLETED') {
      btn.disabled = false;
      btn.innerHTML = `
        Create another cloud-free view
      `;
    } else if (state === 'FAILED') {
      btn.disabled = false;
      btn.innerHTML = `
        Try again
      `;
    } else {
      btn.disabled = false;
      btn.innerHTML = `
        Create cloud-free view
      `;
    }
  },

  /**
   * Enable/Disable Download Action Buttons.
   */
  enableDownloadButtons(enabled = true) {
    const geotiffBtn = document.getElementById('btn-download-geotiff');
    const pngBtn = document.getElementById('btn-download-png');
    const viewJsonBtn = document.getElementById('btn-view-json-meta');

    if (geotiffBtn) geotiffBtn.disabled = !enabled;
    if (pngBtn) pngBtn.disabled = !enabled;
    if (viewJsonBtn) viewJsonBtn.disabled = !enabled;
  },
};
