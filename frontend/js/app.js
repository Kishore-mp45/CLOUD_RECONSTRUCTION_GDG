/**
 * frontend/js/app.js
 * ==================
 * Main Application Bootstrap Script for ALLClear Cloud Removal Frontend.
 */

import { Api } from './api.js';
import { UI } from './ui.js';
import { Scenes } from './scenes.js';
import { Viewer } from './viewer.js';
import { Inference } from './inference.js';
import { Metrics } from './metrics.js';

document.addEventListener('DOMContentLoaded', async () => {
  console.log('[ALLClear] Initializing Professional Geospatial Frontend...');

  // 1. Initialize Submodules
  Viewer.init();
  Inference.init();
  await Scenes.init();
  await Metrics.init();

  // 2. Bind Navigation Tabs
  const navTabs = document.querySelectorAll('.nav-tab');
  navTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const tabId = tab.dataset.tab;
      UI.switchTab(tabId);
    });
  });

  // 3. Bind Modal Close Events
  const modalCloseBtn = document.getElementById('modal-close-btn');
  const modalBackdrop = document.getElementById('modal-backdrop');
  if (modalCloseBtn) modalCloseBtn.addEventListener('click', () => UI.closeModal());
  if (modalBackdrop) {
    modalBackdrop.addEventListener('click', (e) => {
      if (e.target === modalBackdrop) UI.closeModal();
    });
  }

  // 4. Check Backend Health Status
  await updateSystemHealth();

  // 5. Global Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      UI.closeModal();
    }
  });

  console.log('[ALLClear] System Ready.');
});

/**
 * Query backend health endpoint and update status indicators in header.
 */
async function updateSystemHealth() {
  const apiStatusDot = document.getElementById('status-dot-api');
  const apiStatusText = document.getElementById('status-text-api');
  const gpuStatusDot = document.getElementById('status-dot-gpu');
  const gpuStatusText = document.getElementById('status-text-gpu');
  const modelStatusDot = document.getElementById('status-dot-model');
  const modelStatusText = document.getElementById('status-text-model');

  try {
    const health = await Api.getHealth();

    if (health.status === 'ok') {
      if (apiStatusDot) apiStatusDot.className = 'status-dot online';
      if (apiStatusText) apiStatusText.textContent = 'API Online';

      if (health.device && health.device.cuda_available) {
        if (gpuStatusDot) gpuStatusDot.className = 'status-dot online';
        if (gpuStatusText) gpuStatusText.textContent = health.device.device_name ? `${health.device.device_name.replace('NVIDIA GeForce ', '')}` : 'GPU Ready';
      } else {
        if (gpuStatusDot) gpuStatusDot.className = 'status-dot ready';
        if (gpuStatusText) gpuStatusText.textContent = 'CPU Mode';
      }

      if (health.model && health.model.checkpoint_exists) {
        if (modelStatusDot) modelStatusDot.className = 'status-dot ready';
        if (modelStatusText) modelStatusText.textContent = 'DSen2-CR Loaded';
      } else {
        if (modelStatusDot) modelStatusDot.className = 'status-dot busy';
        if (modelStatusText) modelStatusText.textContent = 'Model Missing';
      }
    } else {
      if (apiStatusDot) apiStatusDot.className = 'status-dot error';
      if (apiStatusText) apiStatusText.textContent = 'API Offline';
    }
  } catch (err) {
    if (apiStatusDot) apiStatusDot.className = 'status-dot error';
    if (apiStatusText) apiStatusText.textContent = 'Connection Error';
  }
}
