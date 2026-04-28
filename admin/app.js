/* admin/app.js — Admin Dashboard SPA logic */
const API = '/admin/api';
let token = '';
let ws = null;
let logAutoScroll = false;  // Start paused by default so logs don't fly by
let logFilter = 'ALL';
let logSearch = '';
let currentPage = 'overview';
let _pendingLogs = [];      // Buffer for incoming logs
let _logFlushInterval = null;  // Interval for flushing logs
let _logsPerSecond = 10;    // Rate limit: max logs to render per second

// ── Auth ─────────────────────────────────────────────────────────
async function login() {
  const u = document.getElementById('login-user').value;
  const p = document.getElementById('login-pass').value;
  const err = document.getElementById('login-err');
  err.textContent = '';
  try {
    const r = await fetch(`${API}/login`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({username:u, password:p})
    });
    const d = await r.json();
    if (!r.ok) { err.textContent = d.detail || 'Login failed'; return; }
    token = d.token;
    localStorage.setItem('xmem_admin_token', token);
    showApp();
  } catch(e) { err.textContent = 'Connection failed'; }
}

function logout() {
  fetch(`${API}/logout`, {method:'POST', credentials:'include'});
  token = ''; localStorage.removeItem('xmem_admin_token');
  if (ws) ws.close();
  document.getElementById('login-screen').style.display = '';
  document.querySelector('.app').style.display = 'none';
}

function showApp() {
  document.getElementById('login-screen').style.display = 'none';
  document.querySelector('.app').style.display = 'flex';
  navigate('overview');
  connectLogs();
  loadAll();
}

// ── Navigation ───────────────────────────────────────────────────
function navigate(page) {
  currentPage = page;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const el = document.getElementById('page-' + page);
  if (el) el.classList.add('active');
  const nav = document.querySelector(`[data-page="${page}"]`);
  if (nav) nav.classList.add('active');
  if (page === 'overview') loadOverview();
  if (page === 'llm') loadLLM();
  if (page === 'github') loadGitHub();
}

// ── API helper ───────────────────────────────────────────────────
async function apiFetch(path) {
  const r = await fetch(`${API}${path}`, {
    headers: {'Authorization': `Bearer ${token}`},
    credentials: 'include'
  });
  if (r.status === 401) { logout(); return null; }
  return await r.json();
}

// ── Load all data ────────────────────────────────────────────────
async function loadAll() {
  loadOverview();
  loadLLM();
  loadGitHub();
  // Auto-refresh every 30s
  setInterval(() => {
    if (currentPage === 'overview') loadOverview();
    if (currentPage === 'llm') loadLLM();
  }, 30000);
}

// ── Overview page ────────────────────────────────────────────────
async function loadOverview() {
  const [metrics, analytics] = await Promise.all([
    apiFetch('/server/metrics'),
    apiFetch('/analytics/summary')
  ]);
  if (!metrics) return;

  // Server status cards
  const upH = metrics.uptime_seconds ? (metrics.uptime_seconds / 3600).toFixed(1) : '0';
  document.getElementById('card-uptime').textContent = upH + 'h';
  document.getElementById('card-status').innerHTML = metrics.pipelines_ready
    ? '<span class="badge badge-ok">READY</span>'
    : '<span class="badge badge-err">DOWN</span>';
  document.getElementById('card-env').textContent = metrics.environment || 'N/A';

  if (analytics) {
    document.getElementById('card-users').textContent = analytics.unique_users_24h || 0;

    // API calls table
    const rows = (analytics.api_calls_24h || []).map(r =>
      `<tr><td>${r._id?.method||''}</td><td>${r._id?.path||''}</td><td>${r.count}</td><td>${Math.round(r.avg_latency||0)}ms</td><td>${r.errors||0}</td></tr>`
    ).join('');
    document.getElementById('api-calls-body').innerHTML = rows || '<tr><td colspan="5" style="color:var(--text2)">No data yet</td></tr>';

    // Hourly chart
    renderHourlyChart(analytics.hourly_volume || []);
  }
}

// ── LLM page ─────────────────────────────────────────────────────
async function loadLLM() {
  const data = await apiFetch('/analytics/summary');
  if (!data) return;

  // Token usage cards
  const tu = data.token_usage_7d || {};
  document.getElementById('card-total-tokens').textContent = formatNum(tu.total || 0);
  document.getElementById('card-input-tokens').textContent = formatNum(tu.total_input || 0);
  document.getElementById('card-output-tokens').textContent = formatNum(tu.total_output || 0);
  document.getElementById('card-llm-calls').textContent = formatNum(tu.call_count || 0);

  // LLM stats table
  const rows = (data.llm_stats_24h || []).map(r =>
    `<tr><td>${r._id?.provider||''}</td><td style="font-family:'JetBrains Mono',monospace;font-size:11px">${r._id?.model||''}</td><td>${r._id?.agent||''}</td><td>${r.count}</td><td>${formatNum(r.total_tokens||0)}</td><td>${Math.round(r.avg_latency||0)}ms</td><td>${r.errors||0}</td></tr>`
  ).join('');
  document.getElementById('llm-stats-body').innerHTML = rows || '<tr><td colspan="7" style="color:var(--text2)">No LLM calls recorded yet</td></tr>';

  // Daily LLM chart
  renderDailyLLMChart(data.daily_llm_calls || []);
}

// ── GitHub page ──────────────────────────────────────────────────
async function loadGitHub() {
  const data = await apiFetch('/github/traffic');
  if (!data) return;

  if (data.error) {
    document.getElementById('gh-error').textContent = data.error;
    return;
  }

  const views = data.views || {};
  const clones = data.clones || {};
  const stars = data.stars || {};

  document.getElementById('gh-views').textContent = views.count || 0;
  document.getElementById('gh-unique-views').textContent = views.uniques || 0;
  document.getElementById('gh-clones').textContent = clones.count || 0;
  document.getElementById('gh-stars').textContent = stars.stargazers_count || 0;

  // Views chart
  renderGitHubChart(views.views || [], clones.clones || []);

  // Referrers table
  const refRows = (data.referrers || []).map(r =>
    `<tr><td>${r.referrer||''}</td><td>${r.count||0}</td><td>${r.uniques||0}</td></tr>`
  ).join('');
  document.getElementById('gh-referrers-body').innerHTML = refRows || '<tr><td colspan="3" style="color:var(--text2)">No data</td></tr>';

  // Popular paths table
  const pathRows = (data.paths || []).map(r =>
    `<tr><td style="font-family:'JetBrains Mono',monospace;font-size:11px">${r.path||''}</td><td>${r.count||0}</td><td>${r.uniques||0}</td></tr>`
  ).join('');
  document.getElementById('gh-paths-body').innerHTML = pathRows || '<tr><td colspan="3" style="color:var(--text2)">No data</td></tr>';
}

// ── Live Logs — WebSocket with HTTP polling fallback ─────────────
let _wsFailCount = 0;
const _WS_MAX_FAILURES = 3;       // Switch to polling after this many WS failures
let _wsBackoff = 3000;            // Current reconnect delay (grows exponentially)
let _logMode = 'ws';              // 'ws' | 'poll'
let _pollTimer = null;
let _lastLogId = -1;              // Track highest log id for incremental polling
const _POLL_INTERVAL_MS = 2000;   // Poll every 2 seconds

function connectLogs() {
  // If we've exceeded max WS failures, switch to polling permanently
  if (_wsFailCount >= _WS_MAX_FAILURES) {
    if (_logMode !== 'poll') {
      _logMode = 'poll';
      console.warn('[logs] WebSocket unavailable — switching to HTTP polling');
      _updateLogStatus('polling');
      _startPolling();
    }
    return;
  }

  if (ws) { try { ws.close(); } catch(e) {} }

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  _updateLogStatus('connecting');

  try {
    ws = new WebSocket(`${proto}://${location.host}/admin/ws/logs?token=${token}`);
  } catch (e) {
    _onWsFail();
    return;
  }

  ws.onopen = () => {
    _wsFailCount = 0;
    _wsBackoff = 3000;
    _logMode = 'ws';
    _updateLogStatus('live');
    console.info('[logs] WebSocket connected');
  };

  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.type === 'ping') return;
    if (d.id !== undefined && d.id > _lastLogId) _lastLogId = d.id;
    appendLog(d);
  };

  ws.onclose = (ev) => {
    // Code 4001 = auth rejected, don't retry
    if (ev.code === 4001) {
      _updateLogStatus('auth-error');
      return;
    }
    _onWsFail();
  };

  ws.onerror = () => {
    // onerror is always followed by onclose, so we just suppress here
  };
}

function _onWsFail() {
  _wsFailCount++;
  if (_wsFailCount >= _WS_MAX_FAILURES) {
    // Switch to polling
    connectLogs();
  } else {
    // Retry WebSocket with exponential backoff
    _updateLogStatus('reconnecting');
    const delay = Math.min(_wsBackoff, 30000);
    _wsBackoff = Math.round(_wsBackoff * 1.5);
    setTimeout(connectLogs, delay);
  }
}

function _startPolling() {
  _stopPolling();
  // Fetch initial batch
  _pollLogs();
  // Then poll on interval
  _pollTimer = setInterval(_pollLogs, _POLL_INTERVAL_MS);
}

function _stopPolling() {
  if (_pollTimer) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
}

async function _pollLogs() {
  try {
    const url = `${API}/logs/recent?since_id=${_lastLogId}`;
    const r = await fetch(url, {
      headers: { 'Authorization': `Bearer ${token}` },
      credentials: 'include',
    });
    if (r.status === 401) { logout(); return; }
    if (!r.ok) return;

    const entries = await r.json();
    for (const entry of entries) {
      if (entry.id !== undefined && entry.id > _lastLogId) _lastLogId = entry.id;
      appendLog(entry);
    }
  } catch (e) {
    // Network error — silently retry next interval
  }
}

function _updateLogStatus(state) {
  const el = document.getElementById('log-connection-status');
  if (!el) return;
  const labels = {
    'connecting':   '🔄 Connecting…',
    'live':         '🟢 Live (WebSocket)',
    'reconnecting': '🟡 Reconnecting…',
    'polling':      '🔵 Live (HTTP polling)',
    'auth-error':   '🔴 Auth failed',
  };
  el.textContent = labels[state] || state;
  el.className = 'log-status log-status-' + state;
}

let _seenLogIds = new Set();
let _pausedBuffer = [];

// Rate-limited log rendering
function appendLog(entry) {
  // Always add to pending queue - rate limited flush will handle it
  _pendingLogs.push(entry);

  // Keep pending queue from growing too large (drop old logs if too many)
  if (_pendingLogs.length > 2000) {
    _pendingLogs = _pendingLogs.slice(-1500);  // Keep most recent 1500
  }

  // Update stats display
  const statsEl = document.getElementById('log-stats');
  if (statsEl) {
    statsEl.textContent = `queued: ${_pendingLogs.length}`;
  }
}

// Flush logs at a controlled rate (called periodically)
function _flushLogs() {
  if (_pendingLogs.length === 0) return;

  // Calculate how many logs to render this flush
  const toRender = Math.min(_pendingLogs.length, Math.ceil(_logsPerSecond / 10));
  const entries = _pendingLogs.splice(0, toRender);

  const container = document.getElementById('log-container');
  if (!container) return;

  // Build HTML for all entries at once (much faster than individual DOM updates)
  const htmlFragments = [];
  for (const entry of entries) {
    // Skip if already seen
    if (entry.id !== undefined && _seenLogIds.has(entry.id)) continue;
    if (entry.id !== undefined) _seenLogIds.add(entry.id);

    // Apply filters
    if (logFilter !== 'ALL' && entry.level !== logFilter) continue;
    if (logSearch && !(entry.msg||'').toLowerCase().includes(logSearch.toLowerCase())
        && !(entry.logger||'').toLowerCase().includes(logSearch.toLowerCase())) continue;

    const ts = entry.ts ? entry.ts.split('T')[1]?.substring(0,8) || '' : '';
    htmlFragments.push(`<div class="log-line"><span class="ts">${ts}</span> <span class="lvl lvl-${entry.level}">${entry.level}</span> <span class="src">${entry.logger||''}</span> ${escHtml(entry.msg||'')}</div>`);
  }

  if (htmlFragments.length === 0) return;

  // Single DOM insert for all logs
  const wasAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;
  container.insertAdjacentHTML('beforeend', htmlFragments.join(''));

  // Keep max 1000 lines
  while (container.children.length > 1000) {
    container.removeChild(container.firstChild);
  }

  // Auto-scroll only if we were already near bottom and not paused
  if (logAutoScroll && wasAtBottom) {
    container.scrollTop = container.scrollHeight;
  }
}

function clearLogs() {
  const container = document.getElementById('log-container');
  if (container) container.innerHTML = '';
  _seenLogIds.clear();
  _pausedBuffer = [];
  _pendingLogs = [];
}

function setLogFilter(val) {
  logFilter = val;
  // Update button text to show active filter
  const select = document.querySelector('#page-logs select');
  if (select) select.value = val;
}

function setLogSearch(val) {
  logSearch = val;
}

function toggleAutoScroll() {
  logAutoScroll = !logAutoScroll;
  const btn = document.getElementById('btn-autoscroll');
  if (btn) btn.textContent = logAutoScroll ? '⏸ Pause' : '▶ Resume';

  // If resuming, scroll to bottom immediately
  if (logAutoScroll) {
    const container = document.getElementById('log-container');
    if (container) container.scrollTop = container.scrollHeight;
  }
}

function setLogRate(rate) {
  _logsPerSecond = parseInt(rate, 10) || 10;
}

// Internal append for direct rendering (used by flush)
function _appendLogInternal(entry) {
  // Skip if already seen (by id)
  if (entry.id !== undefined && _seenLogIds.has(entry.id)) return;
  if (entry.id !== undefined) _seenLogIds.add(entry.id);

  const container = document.getElementById('log-container');
  if (!container) return;

  // Filter
  if (logFilter !== 'ALL' && entry.level !== logFilter) return;
  if (logSearch && !(entry.msg||'').toLowerCase().includes(logSearch.toLowerCase())
      && !(entry.logger||'').toLowerCase().includes(logSearch.toLowerCase())) return;

  const div = document.createElement('div');
  div.className = 'log-line';
  const ts = entry.ts ? entry.ts.split('T')[1]?.substring(0,8) || '' : '';
  div.innerHTML = `<span class="ts">${ts}</span> <span class="lvl lvl-${entry.level}">${entry.level}</span> <span class="src">${entry.logger||''}</span> ${escHtml(entry.msg||'')}`;
  container.appendChild(div);

  // Keep max 1000 lines
  while (container.children.length > 1000) {
    const first = container.firstChild;
    if (first) container.removeChild(first);
  }

  if (logAutoScroll) container.scrollTop = container.scrollHeight;
}

// ── Charts (Chart.js) ────────────────────────────────────────────
let hourlyChart, dailyLLMChart, ghChart;

function renderHourlyChart(data) {
  const ctx = document.getElementById('chart-hourly');
  if (!ctx) return;
  const labels = data.map(d => `${d._id?.hour||0}:00`);
  const counts = data.map(d => d.count||0);
  const errors = data.map(d => d.errors||0);

  if (hourlyChart) hourlyChart.destroy();
  hourlyChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {label:'Requests',data:counts,backgroundColor:'rgba(108,92,231,0.6)',borderRadius:4},
        {label:'Errors',data:errors,backgroundColor:'rgba(255,107,107,0.6)',borderRadius:4}
      ]
    },
    options: chartOpts('Hourly Request Volume (24h)')
  });
}

function renderDailyLLMChart(data) {
  const ctx = document.getElementById('chart-daily-llm');
  if (!ctx) return;
  const labels = data.map(d => `${d._id?.month||0}/${d._id?.day||0}`);
  const calls = data.map(d => d.count||0);
  const tokens = data.map(d => d.tokens||0);

  if (dailyLLMChart) dailyLLMChart.destroy();
  dailyLLMChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {label:'LLM Calls',data:calls,borderColor:'#6c5ce7',backgroundColor:'rgba(108,92,231,0.1)',fill:true,tension:.3},
        {label:'Tokens (÷100)',data:tokens.map(t=>Math.round(t/100)),borderColor:'#00cec9',backgroundColor:'rgba(0,206,201,0.1)',fill:true,tension:.3}
      ]
    },
    options: chartOpts('Daily LLM Usage (7d)')
  });
}

function renderGitHubChart(views, clones) {
  const ctx = document.getElementById('chart-github');
  if (!ctx) return;
  const labels = views.map(v => v.timestamp?.substring(5,10) || '');
  const viewCounts = views.map(v => v.count||0);
  const cloneMap = {};
  clones.forEach(c => { cloneMap[c.timestamp?.substring(5,10)] = c.count||0; });
  const cloneCounts = labels.map(l => cloneMap[l]||0);

  if (ghChart) ghChart.destroy();
  ghChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {label:'Views',data:viewCounts,borderColor:'#74b9ff',backgroundColor:'rgba(116,185,255,0.1)',fill:true,tension:.3},
        {label:'Clones',data:cloneCounts,borderColor:'#00cec9',backgroundColor:'rgba(0,206,201,0.1)',fill:true,tension:.3}
      ]
    },
    options: chartOpts('GitHub Traffic (14d)')
  });
}

function chartOpts(title) {
  return {
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#8888a0',font:{size:11}}},title:{display:true,text:title,color:'#e4e4f0',font:{size:13}}},
    scales:{
      x:{ticks:{color:'#8888a0',font:{size:10}},grid:{color:'rgba(42,42,58,0.5)'}},
      y:{ticks:{color:'#8888a0',font:{size:10}},grid:{color:'rgba(42,42,58,0.5)'},beginAtZero:true}
    }
  };
}

// ── Helpers ──────────────────────────────────────────────────────
function formatNum(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return String(n);
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  token = localStorage.getItem('xmem_admin_token') || '';

  // Start the rate-limited log flush (100ms = 10fps max)
  _logFlushInterval = setInterval(_flushLogs, 100);

  // Update pause button to show correct initial state
  const btn = document.getElementById('btn-autoscroll');
  if (btn) btn.textContent = logAutoScroll ? '⏸ Pause' : '▶ Resume';

  if (token) {
    // Verify token still valid
    apiFetch('/server/metrics').then(r => {
      if (r) showApp(); else { token=''; localStorage.removeItem('xmem_admin_token'); }
    }).catch(() => {});
  }
  // Enter key on login
  document.getElementById('login-pass')?.addEventListener('keydown', e => { if(e.key==='Enter') login(); });
});
