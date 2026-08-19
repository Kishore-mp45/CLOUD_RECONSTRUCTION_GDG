/**
 * frontend/js/scenes.js
 * =====================
 * Scene Catalog Management & Phase 7 Cloud Density Filtering.
 */

import { Api } from './api.js';
import { UI } from './ui.js';
import { Viewer } from './viewer.js';

export const Scenes = {
  allScenes: [],
  selectedScene: null,
  currentThreshold: 60.0,

  /**
   * Initialize scenes module, fetch catalog, bind event listeners.
   */
  async init() {
    this.bindEvents();
    await this.loadScenes();
  },

  /**
   * Bind threshold slider and search events.
   */
  bindEvents() {
    const slider = document.getElementById('threshold-slider');
    const thresholdVal = document.getElementById('threshold-val');
    const searchInput = document.getElementById('scene-search-input');

    if (slider) {
      slider.addEventListener('input', (e) => {
        this.currentThreshold = parseFloat(e.target.value);
        if (thresholdVal) {
          thresholdVal.textContent = `${this.currentThreshold.toFixed(0)}%`;
        }
        this.applyFilterAndRender();
      });
    }

    if (searchInput) {
      searchInput.addEventListener('input', () => {
        this.applyFilterAndRender();
      });
    }
  },

  /**
   * Fetch all scenes from the backend.
   */
  async loadScenes() {
    const listContainer = document.getElementById('scene-list-container');
    if (listContainer) {
      listContainer.innerHTML = '<div class="skeleton skeleton-text" style="height: 48px;"></div><div class="skeleton skeleton-text" style="height: 48px;"></div>';
    }

    try {
      const response = await Api.getScenes(null, 150, 0);
      this.allScenes = response.scenes || [];
      this.applyFilterAndRender();

      // Automatically select the first eligible scene if none selected
      const firstEligible = this.allScenes.find(
        (s) => s.cloud_density_percent >= this.currentThreshold
      );
      if (firstEligible) {
        this.selectScene(firstEligible.scene_id);
      }
    } catch (err) {
      console.error('Failed to load scenes:', err);
      if (listContainer) {
        listContainer.innerHTML = `
          <div class="empty-state">
            <div class="empty-state-icon">⚠️</div>
            <h4>Failed to load scenes</h4>
            <p>${err.message}</p>
          </div>
        `;
      }
      UI.showToast(`Error connecting to scene catalog: ${err.message}`, 'error');
    }
  },

  /**
   * Filter scenes according to the current cloud density threshold & search term.
   */
  applyFilterAndRender() {
    const listContainer = document.getElementById('scene-list-container');
    const eligibleCountEl = document.getElementById('eligible-count-val');
    const filteredCountEl = document.getElementById('filtered-count-val');
    const totalCountEl = document.getElementById('total-count-val');
    const searchInput = document.getElementById('scene-search-input');

    const query = (searchInput?.value || '').trim().toLowerCase();

    const filtered = this.allScenes.filter((scene) => {
      const matchesSearch =
        !query ||
        scene.scene_id.toLowerCase().includes(query) ||
        scene.roi_id.toLowerCase().includes(query);
      return matchesSearch;
    });

    const eligibleCount = filtered.filter(
      (s) => s.cloud_density_percent >= this.currentThreshold
    ).length;
    const belowCount = filtered.length - eligibleCount;

    if (eligibleCountEl) eligibleCountEl.textContent = eligibleCount;
    if (filteredCountEl) filteredCountEl.textContent = belowCount;
    if (totalCountEl) totalCountEl.textContent = filtered.length;

    if (!listContainer) return;

    if (filtered.length === 0) {
      listContainer.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">🔍</div>
          <h4>No scenes found</h4>
          <p>Try lowering the search term or adjusting filters.</p>
        </div>
      `;
      return;
    }

    listContainer.innerHTML = '';

    filtered.forEach((scene) => {
      const isEligible = scene.cloud_density_percent >= this.currentThreshold;
      const isSelected = this.selectedScene && this.selectedScene.scene_id === scene.scene_id;

      const card = document.createElement('div');
      card.className = `scene-item-card ${isSelected ? 'selected' : ''}`;
      card.dataset.sceneId = scene.scene_id;

      let badgeClass = 'eligible';
      if (scene.cloud_density_percent >= 80) badgeClass = 'high';
      else if (scene.cloud_density_percent < this.currentThreshold) badgeClass = 'clear';

      card.innerHTML = `
        <div class="scene-item-top">
          <div class="scene-item-id" title="${scene.scene_id}">${scene.scene_id}</div>
          <div class="density-badge ${badgeClass}">
            <span>☁</span> ${scene.cloud_density_percent.toFixed(1)}%
          </div>
        </div>
        <div class="scene-item-bottom">
          <span>📍 ${scene.roi_id}</span>
          <span>📅 ${scene.acquisition_time || '2022'}</span>
          <span>${isEligible ? '✅ PASS' : '❌ FILTER'}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        this.selectScene(scene.scene_id);
      });

      listContainer.appendChild(card);
    });
  },

  /**
   * Select a specific scene and update all viewers and summary cards.
   */
  async selectScene(sceneId) {
    try {
      const sceneDetail = await Api.getScene(sceneId);
      this.selectedScene = sceneDetail;

      // Update highlighted card in list
      document.querySelectorAll('.scene-item-card').forEach((card) => {
        if (card.dataset.sceneId === sceneId) {
          card.classList.add('selected');
        } else {
          card.classList.remove('selected');
        }
      });

      // Update compact summary card
      this.renderSceneSummary(sceneDetail);

      // Notify viewers
      Viewer.onSceneSelected(sceneDetail);

      // Enable/Disable Reconstruct Button based on threshold eligibility
      const isEligible = sceneDetail.cloud_density_percent >= this.currentThreshold;
      const btn = document.getElementById('run-inference-btn');
      if (btn) {
        btn.disabled = !isEligible;
        btn.title = isEligible
          ? 'Run cloud removal reconstruction'
          : `Scene cloud density (${sceneDetail.cloud_density_percent.toFixed(1)}%) is below threshold (${this.currentThreshold.toFixed(1)}%)`;
      }
    } catch (err) {
      console.error('Failed to select scene:', err);
      UI.showToast(`Error loading scene details: ${err.message}`, 'error');
    }
  },

  /**
   * Render the compact selected scene summary box.
   */
  renderSceneSummary(scene) {
    const summaryContainer = document.getElementById('selected-scene-summary');
    if (!summaryContainer) return;

    summaryContainer.innerHTML = `
      <div class="meta-grid-2col">
        <div class="meta-item">
          <span class="meta-label">Scene ID</span>
          <span class="meta-val" title="${scene.scene_id}">${scene.scene_id}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Acquisition</span>
          <span class="meta-val">${scene.acquisition_time || '2022-06-15'}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Cloud Density</span>
          <span class="meta-val" style="color: var(--accent-amber);">${scene.cloud_density_percent.toFixed(1)}%</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">ROI Region</span>
          <span class="meta-val">${scene.roi_id}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">CRS / Grid</span>
          <span class="meta-val">${scene.crs || 'EPSG:32643'}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Resolution</span>
          <span class="meta-val">${scene.resolution ? `${scene.resolution}m` : '10.0m'}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">S2 Optical</span>
          <span class="meta-val" style="color: var(--accent-emerald);">13 Bands (Ready)</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">S1 SAR</span>
          <span class="meta-val" style="color: var(--accent-cyan);">VV + VH (Ready)</span>
        </div>
      </div>
    `;
  },
};
