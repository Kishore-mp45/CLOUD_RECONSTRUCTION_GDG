/**
 * frontend/js/viewer.js
 * =====================
 * Multi-Modality Imagery Viewers & Interactive Draggable Before/After Split Slider.
 */

import { Api } from './api.js';

export const Viewer = {
  currentMode: 'slider', // 'slider' | '3panel' | '4panel'
  currentScene: null,
  currentResult: null,
  sliderPosition: 50.0, // percentage (0% to 100%)
  isDragging: false,

  /**
   * Initialize viewer, bind slider dragging, and view mode buttons.
   */
  init() {
    this.bindSliderEvents();
    this.bindViewModeButtons();
  },

  /**
   * Bind mouse, touch, and keyboard interactions for the interactive split slider.
   */
  bindSliderEvents() {
    const container = document.getElementById('split-slider-container');
    const handle = document.getElementById('split-handle');

    if (!container || !handle) return;

    const onPointerMove = (e) => {
      if (!this.isDragging) return;
      const rect = container.getBoundingClientRect();
      let x = e.clientX - rect.left;
      x = Math.max(0, Math.min(x, rect.width));
      this.sliderPosition = (x / rect.width) * 100;
      this.updateSliderUI();
    };

    const onPointerUp = () => {
      this.isDragging = false;
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
    };

    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.isDragging = true;
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);
      window.addEventListener('pointercancel', onPointerUp);
    });

    container.addEventListener('pointerdown', (e) => {
      // If clicking inside container but not handle directly, jump slider to click
      if (e.target !== handle && !handle.contains(e.target)) {
        const rect = container.getBoundingClientRect();
        let x = e.clientX - rect.left;
        x = Math.max(0, Math.min(x, rect.width));
        this.sliderPosition = (x / rect.width) * 100;
        this.updateSliderUI();
      }
    });

    // Keyboard Accessibility (Left/Right Arrows)
    container.setAttribute('tabindex', '0');
    container.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') {
        this.sliderPosition = Math.max(0, this.sliderPosition - 5);
        this.updateSliderUI();
      } else if (e.key === 'ArrowRight') {
        this.sliderPosition = Math.min(100, this.sliderPosition + 5);
        this.updateSliderUI();
      }
    });
  },

  /**
   * Update slider DOM styles to reflect current slider position.
   */
  updateSliderUI() {
    const beforeLayer = document.getElementById('split-before-layer');
    const handle = document.getElementById('split-handle');
    const beforeImg = document.getElementById('split-before-img');
    const container = document.getElementById('split-slider-container');

    if (!beforeLayer || !handle || !container) return;

    beforeLayer.style.width = `${this.sliderPosition}%`;
    handle.style.left = `${this.sliderPosition}%`;

    // Keep before image fixed at 100% container width to prevent stretching
    if (beforeImg) {
      beforeImg.style.width = `${container.clientWidth}px`;
    }
  },

  /**
   * Bind view mode toggle buttons (Split Slider, 3-Panel, 4-Panel).
   */
  bindViewModeButtons() {
    const btns = document.querySelectorAll('.view-btn');
    btns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        this.setViewMode(mode);
      });
    });
  },

  /**
   * Switch the active visualization mode.
   */
  setViewMode(mode) {
    this.currentMode = mode;

    document.querySelectorAll('.view-btn').forEach((b) => {
      if (b.dataset.mode === mode) b.classList.add('active');
      else b.classList.remove('active');
    });

    const sliderContainer = document.getElementById('split-slider-container');
    const panelsGrid = document.getElementById('panels-grid-container');

    if (mode === 'slider') {
      if (sliderContainer) sliderContainer.style.display = 'block';
      if (panelsGrid) panelsGrid.style.display = 'none';
      this.updateSliderUI();
    } else {
      if (sliderContainer) sliderContainer.style.display = 'none';
      if (panelsGrid) {
        panelsGrid.style.display = 'grid';
        if (mode === '3panel') {
          panelsGrid.className = 'panels-grid-3';
        } else {
          panelsGrid.className = 'panels-grid-3'; // 3/4 panel responsive grid
        }
      }
    }
  },

  /**
   * Triggered when a new scene is selected from the catalog.
   */
  onSceneSelected(scene) {
    this.currentScene = scene;
    this.currentResult = null;

    const cloudyUrl = Api.getScenePreviewUrl(scene.scene_id, 's2');
    const sarUrl = Api.getScenePreviewUrl(scene.scene_id, 's1');

    // Update Split Slider Before (Cloudy S2)
    const splitBeforeImg = document.getElementById('split-before-img');
    const splitAfterImg = document.getElementById('split-after-img');

    if (splitBeforeImg) splitBeforeImg.src = cloudyUrl;
    // Set placeholder until reconstruction runs
    if (splitAfterImg) splitAfterImg.src = cloudyUrl;

    // Update 3-Panel Images
    const sarPanelImg = document.getElementById('panel-img-sar');
    const cloudyPanelImg = document.getElementById('panel-img-cloudy');
    const reconPanelImg = document.getElementById('panel-img-recon');

    if (sarPanelImg) sarPanelImg.src = sarUrl;
    if (cloudyPanelImg) cloudyPanelImg.src = cloudyUrl;
    if (reconPanelImg) reconPanelImg.src = cloudyUrl;

    // Reset processing state
    this.hideProcessingOverlay();
  },

  /**
   * Triggered when inference completes successfully.
   */
  onInferenceCompleted(result) {
    this.currentResult = result;
    this.hideProcessingOverlay();

    const reconUrl = Api.getResultPreviewUrl(result.result_id, 'reconstructed');
    const cloudyUrl = Api.getResultPreviewUrl(result.result_id, 'cloudy');

    // Update Split Slider
    const splitBeforeImg = document.getElementById('split-before-img');
    const splitAfterImg = document.getElementById('split-after-img');

    if (splitBeforeImg) splitBeforeImg.src = cloudyUrl;
    if (splitAfterImg) splitAfterImg.src = reconUrl;

    // Update 3-Panel Reconstructed Image
    const reconPanelImg = document.getElementById('panel-img-recon');
    if (reconPanelImg) reconPanelImg.src = reconUrl;

    // Reset slider to 50% for immediate dramatic comparison
    this.sliderPosition = 50.0;
    this.updateSliderUI();
  },

  /**
   * Display radar scan / inference processing animation.
   */
  showProcessingOverlay(message = 'Reconstructing multi-spectral optical bands with DSen2-CR...') {
    const overlay = document.getElementById('processing-overlay');
    const msgEl = document.getElementById('processing-status-msg');
    if (overlay) overlay.style.display = 'flex';
    if (msgEl) msgEl.textContent = message;
  },

  /**
   * Hide processing overlay.
   */
  hideProcessingOverlay() {
    const overlay = document.getElementById('processing-overlay');
    if (overlay) overlay.style.display = 'none';
  },
};
