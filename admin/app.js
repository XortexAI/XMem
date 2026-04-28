/* admin/app.js — Admin Dashboard SPA logic */
const API = '/admin/api';
let token = '';
let currentPage = 'overview';

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
  if (_sseSource) { _sseSource.close(); _sseSource = null; }
  document.getElementById('login-screen').style.display = '';
  document.querySelector('.app').style.display = 'none';
}

function showApp() {
  document.getElementById('login-screen').style.display = 'none';
  document.querySelector('.app').style.display = 'flex';
  navigate('overview');
  connectSystemLogs();
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

  // Update page title
  const titles = { overview:'Overview', logs:'Live Terminal', llm:'LLM & Costs', github:'GitHub Traffic' };
  const titleEl = document.getElementById('page-title');
  if (titleEl) titleEl.textContent = titles[page] || page;

  if (page === 'overview') loadOverview();
  if (page === 'llm') loadLLM();
  if (page === 'github') loadGitHub();
  if (page === 'logs' && !_sseSource) connectSystemLogs();
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

  // Total cost card
  const totalCost = tu.total_cost_usd || 0;
  document.getElementById('card-total-cost').textContent = formatCost(totalCost);

  // Cost breakdown table
  const costRows = (data.cost_by_model_7d || []).map(r =>
    `<tr><td style="font-family:'JetBrains Mono',monospace;font-size:11px">${r._id||'unknown'}</td><td>${r.call_count||0}</td><td>${formatNum(r.total_input||0)}</td><td>${formatNum(r.total_output||0)}</td><td style="color:#4fc3f7;font-weight:600">${formatCost(r.cost_usd||0)}</td></tr>`
  ).join('');
  document.getElementById('cost-breakdown-body').innerHTML = costRows || '<tr><td colspan="5" style="color:var(--text2)">No cost data yet</td></tr>';

  // LLM stats table (with cost column)
  const rows = (data.llm_stats_24h || []).map(r =>
    `<tr><td>${r._id?.provider||''}</td><td style="font-family:'JetBrains Mono',monospace;font-size:11px">${r._id?.model||''}</td><td>${r._id?.agent||''}</td><td>${r.count}</td><td>${formatNum(r.total_tokens||0)}</td><td style="color:#4fc3f7">${formatCost(r.cost_usd||0)}</td><td>${Math.round(r.avg_latency||0)}ms</td><td>${r.errors||0}</td></tr>`
  ).join('');
  document.getElementById('llm-stats-body').innerHTML = rows || '<tr><td colspan="8" style="color:var(--text2)">No LLM calls recorded yet</td></tr>';

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


// ═════════════════════════════════════════════════════════════════
// Live Logs — SSE (Server-Sent Events) + journalctl subprocess
//
// NO WebSocket needed. SSE works over plain HTTP and is NOT
// blocked by nginx/reverse proxies. This is the only log approach.
// ═════════════════════════════════════════════════════════════════

let _sseSource = null;
let _termAutoScroll = true;
let _termSearch = '';
let _termLineCount = 0;
let _termReconnectTimer = null;

function connectSystemLogs() {
  // Close existing connection
  if (_sseSource) { _sseSource.close(); _sseSource = null; }
  if (_termReconnectTimer) { clearTimeout(_termReconnectTimer); _termReconnectTimer = null; }

  _updateTermStatus('connecting');

  // Use EventSource with fetch-based approach for auth header support
  // EventSource doesn't support custom headers, so we use fetch + ReadableStream
  const url = `${API}/system-logs/stream`;

  fetch(url, {
    headers: { 'Authorization': `Bearer ${token}` },
    credentials: 'include',
  }).then(response => {
    if (response.status === 401) { logout(); return; }
    if (!response.ok) {
      _updateTermStatus('error');
      _scheduleTermReconnect();
      return;
    }

    _updateTermStatus('live');
    const reconnectBtn = document.getElementById('btn-term-reconnect');
    if (reconnectBtn) reconnectBtn.style.display = 'none';

    // Store a flag so we can abort
    const controller = new AbortController();
    _sseSource = { close: () => controller.abort() };

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    function readChunk() {
      reader.read().then(({ done, value }) => {
        if (done) {
          _sseSource = null;
          _updateTermStatus('disconnected');
          const btn = document.getElementById('btn-term-reconnect');
          if (btn) btn.style.display = '';
          _scheduleTermReconnect();
          return;
        }

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from the buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete last line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const text = line.substring(6); // Remove "data: " prefix
            _appendTermLine(text);
          } else if (line.startsWith('event: error')) {
            // Next data line will be the error message
          } else if (line.startsWith(':keepalive')) {
            // Ignore keepalive comments
          }
          // Ignore empty lines (SSE event separators)
        }

        readChunk();
      }).catch(err => {
        if (err.name === 'AbortError') return; // Intentional close
        console.error('[logs] Stream error:', err);
        _sseSource = null;
        _updateTermStatus('disconnected');
        const btn = document.getElementById('btn-term-reconnect');
        if (btn) btn.style.display = '';
        _scheduleTermReconnect();
      });
    }

    readChunk();

  }).catch(err => {
    console.error('[logs] Fetch error:', err);
    _updateTermStatus('error');
    _scheduleTermReconnect();
  });
}

function _scheduleTermReconnect() {
  if (_termReconnectTimer) return;
  _termReconnectTimer = setTimeout(() => {
    _termReconnectTimer = null;
    connectSystemLogs();
  }, 5000);
}

function _updateTermStatus(state) {
  const el = document.getElementById('term-connection-status');
  if (!el) return;
  const labels = {
    'connecting':   '🔄 Connecting…',
    'live':         '🟢 Live (SSE stream)',
    'disconnected': '🔴 Disconnected',
    'error':        '🔴 Connection Error',
    'auth-error':   '🔴 Auth Failed',
  };
  el.textContent = labels[state] || state;
  el.className = 'log-status log-status-' + (state === 'live' ? 'live' : state === 'connecting' ? 'connecting' : 'auth-error');
}

function _parseJournalLine(raw) {
  // journalctl short-iso format:
  // 2026-04-28T18:30:00+0530 hostname xmem[1234]: actual message
  const match = raw.match(/^(\S+)\s+(\S+)\s+(\S+?)(?:\[\d+\])?:\s*(.*)$/);
  if (match) {
    return { ts: match[1], host: match[2], service: match[3], msg: match[4] };
  }
  return { ts: '', host: '', service: '', msg: raw };
}

function _appendTermLine(raw) {
  if (!raw && raw !== '') return;

  // Apply search filter
  if (_termSearch && !raw.toLowerCase().includes(_termSearch.toLowerCase())) return;

  const container = document.getElementById('terminal-container');
  if (!container) return;

  const parsed = _parseJournalLine(raw);
  const div = document.createElement('div');
  div.className = 'term-line';

  // Detect error/warning lines
  const msgLower = (parsed.msg || '').toLowerCase();
  if (msgLower.includes('error') || msgLower.includes('exception') || msgLower.includes('traceback') || msgLower.includes('critical')) {
    div.classList.add('term-error');
  } else if (msgLower.includes('warning') || msgLower.includes('warn')) {
    div.classList.add('term-warning');
  }

  let msgHtml = escHtml(parsed.msg || raw);

  // Highlight search term if present
  if (_termSearch) {
    const regex = new RegExp(`(${_termSearch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    msgHtml = msgHtml.replace(regex, '<span class="term-match">$1</span>');
  }

  if (parsed.ts) {
    const shortTs = parsed.ts.length > 19 ? parsed.ts.substring(11, 19) : parsed.ts;
    div.innerHTML = `<span class="term-ts">${escHtml(shortTs)}</span> <span class="term-host">${escHtml(parsed.host)}</span> <span class="term-service">${escHtml(parsed.service)}</span> <span class="term-msg">${msgHtml}</span>`;
  } else {
    div.innerHTML = `<span class="term-msg">${msgHtml}</span>`;
  }

  const wasAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;
  container.appendChild(div);
  _termLineCount++;

  // Keep max 2000 lines
  while (container.children.length > 2000) {
    container.removeChild(container.firstChild);
  }

  // Update line count
  const countEl = document.getElementById('term-line-count');
  if (countEl) countEl.textContent = `${_termLineCount} lines`;

  // Auto-scroll
  if (_termAutoScroll && wasAtBottom) {
    container.scrollTop = container.scrollHeight;
  }
}

function clearTerminal() {
  const container = document.getElementById('terminal-container');
  if (container) container.innerHTML = '';
  _termLineCount = 0;
  const countEl = document.getElementById('term-line-count');
  if (countEl) countEl.textContent = '0 lines';
}

function toggleTermAutoScroll() {
  _termAutoScroll = !_termAutoScroll;
  const btn = document.getElementById('btn-term-autoscroll');
  if (btn) btn.textContent = _termAutoScroll ? '⏸ Auto-scroll ON' : '▶ Auto-scroll OFF';
  if (_termAutoScroll) {
    const container = document.getElementById('terminal-container');
    if (container) container.scrollTop = container.scrollHeight;
  }
}

function setTermSearch(val) {
  _termSearch = val;
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
        {label:'Requests',data:counts,backgroundColor:'rgba(124,108,240,0.5)',borderColor:'rgba(124,108,240,0.8)',borderWidth:1,borderRadius:6,borderSkipped:false},
        {label:'Errors',data:errors,backgroundColor:'rgba(248,113,113,0.5)',borderColor:'rgba(248,113,113,0.8)',borderWidth:1,borderRadius:6,borderSkipped:false}
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
        {label:'LLM Calls',data:calls,borderColor:'#a78bfa',backgroundColor:'rgba(167,139,250,0.08)',fill:true,tension:.4,pointBackgroundColor:'#a78bfa'},
        {label:'Tokens (÷100)',data:tokens.map(t=>Math.round(t/100)),borderColor:'#34d399',backgroundColor:'rgba(52,211,153,0.08)',fill:true,tension:.4,pointBackgroundColor:'#34d399'}
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
        {label:'Views',data:viewCounts,borderColor:'#60a5fa',backgroundColor:'rgba(96,165,250,0.08)',fill:true,tension:.4,pointBackgroundColor:'#60a5fa'},
        {label:'Clones',data:cloneCounts,borderColor:'#34d399',backgroundColor:'rgba(52,211,153,0.08)',fill:true,tension:.4,pointBackgroundColor:'#34d399'}
      ]
    },
    options: chartOpts('GitHub Traffic (14d)')
  });
}

function chartOpts(title) {
  return {
    responsive:true, maintainAspectRatio:false,
    plugins:{
      legend:{labels:{color:'#7878a0',font:{family:'Inter',size:11},usePointStyle:true,pointStyle:'circle',padding:16}},
      title:{display:false}
    },
    scales:{
      x:{ticks:{color:'#50506e',font:{family:'Inter',size:10}},grid:{color:'rgba(30,30,58,0.4)',drawBorder:false}},
      y:{ticks:{color:'#50506e',font:{family:'Inter',size:10}},grid:{color:'rgba(30,30,58,0.4)',drawBorder:false},beginAtZero:true}
    },
    elements:{line:{borderWidth:2},point:{radius:3,hoverRadius:5}}
  };
}

// ── Helpers ──────────────────────────────────────────────────────
function formatNum(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return String(n);
}

function formatCost(usd) {
  if (usd === 0 || usd === undefined) return '$0.00';
  if (usd < 0.01) return '$' + usd.toFixed(4);
  if (usd < 1) return '$' + usd.toFixed(3);
  return '$' + usd.toFixed(2);
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Init ─────────────────────────────────────────────────────────
function _updateClock() {
  const el = document.getElementById('topbar-clock');
  if (!el) return;
  const now = new Date();
  const h = String(now.getHours()).padStart(2,'0');
  const m = String(now.getMinutes()).padStart(2,'0');
  const s = String(now.getSeconds()).padStart(2,'0');
  el.textContent = `${h}:${m}:${s}`;
}

document.addEventListener('DOMContentLoaded', () => {
  token = localStorage.getItem('xmem_admin_token') || '';

  // Start clock
  _updateClock();
  setInterval(_updateClock, 1000);

  if (token) {
    // Verify token still valid
    apiFetch('/server/metrics').then(r => {
      if (r) showApp(); else { token=''; localStorage.removeItem('xmem_admin_token'); }
    }).catch(() => {});
  }
  // Enter key on login
  document.getElementById('login-pass')?.addEventListener('keydown', e => { if(e.key==='Enter') login(); });
});
