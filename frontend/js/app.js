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
  await loadHistoryTable();

  // 2. Bind Navigation Tabs
  const navTabs = document.querySelectorAll('.nav-tab');
  navTabs.forEach((tab) => {
    tab.addEventListener('click', async () => {
      const tabId = tab.dataset.tab;
      UI.switchTab(tabId);
      if (tabId === 'history') {
        await loadHistoryTable();
      }
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

      if (health.cuda_available) {
        if (gpuStatusDot) gpuStatusDot.className = 'status-dot online';
        if (gpuStatusText) gpuStatusText.textContent = health.gpu_name ? health.gpu_name.replace('NVIDIA GeForce ', '') : 'GPU Ready';
      } else {
        if (gpuStatusDot) gpuStatusDot.className = 'status-dot ready';
        if (gpuStatusText) gpuStatusText.textContent = 'CPU Mode';
      }

      if (health.model_checkpoint_available) {
        if (modelStatusDot) modelStatusDot.className = 'status-dot ready';
        if (modelStatusText) modelStatusText.textContent = 'Model ready';
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

/**
 * Load live audit events into the Processing History table.
 */
export async function loadHistoryTable() {
  const tbody = document.getElementById('history-table-body');
  if (!tbody) return;

  try {
    const data = await Api.getHistory(50);
    const events = data.events || [];

    if (events.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-tertiary);">No audit events recorded yet.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    events.forEach((ev) => {
      const tr = document.createElement('tr');
      const timeStr = ev.created_at ? new Date(ev.created_at).toLocaleTimeString() : '—';
      const durationStr = ev.duration_s ? `${ev.duration_s.toFixed(2)}s` : '—';
      const statusColor = ev.status === 'success' ? 'var(--accent-emerald)' : 'var(--accent-crimson)';

      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-size: 11px;">#${ev.id}</td>
        <td><span class="badge-tag">${ev.entity_type}</span></td>
        <td><code style="font-size: 11px; color: var(--text-cyan);">${ev.entity_id}</code></td>
        <td><code style="font-size: 11px;">${ev.action}</code></td>
        <td><span style="color: ${statusColor}; font-weight: 600;">● ${ev.status.toUpperCase()}</span></td>
        <td style="font-family: var(--font-mono); font-size: 11px;">${durationStr} (${timeStr})</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.warn('Could not load history table:', err);
  }
}
