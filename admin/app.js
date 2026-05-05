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

  const titles = { overview:'Overview', logs:'Live Terminal', llm:'LLM & Costs', github:'GitHub Traffic', users:'Users', scanner:'Scanner Analytics', outreach:'Outreach' };
  const titleEl = document.getElementById('page-title');
  if (titleEl) titleEl.textContent = titles[page] || page;

  if (page === 'overview') loadOverview();
  if (page === 'llm') loadLLM();
  if (page === 'github') loadGitHub();
  if (page === 'users') loadUsers();
  if (page === 'scanner') loadScanner();
  if (page === 'outreach') loadOutreach();
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
  loadScanner();
  // Auto-refresh every 30s
  setInterval(() => {
    if (currentPage === 'overview') loadOverview();
    if (currentPage === 'llm') loadLLM();
    if (currentPage === 'users') loadUsers();
    if (currentPage === 'scanner') loadScanner();
    if (currentPage === 'outreach') loadOutreachJobs();
  }, 30000);
}

// ── Users page ──────────────────────────────────────────────────
let _usersData = [];
let _usersFilter = '';

async function loadUsers() {
  const data = await apiFetch('/users');
  if (!data || data.error) {
    document.getElementById('users-table-body').innerHTML = `<tr><td colspan="7" class="empty-state">${data?.error || 'Failed to load users'}</td></tr>`;
    return;
  }

  _usersData = data.users || [];

  // Update summary cards
  document.getElementById('users-total').textContent = data.total_users || 0;

  const totalKeys = _usersData.reduce((sum, u) => sum + (u.api_key_count || 0), 0);
  document.getElementById('users-total-keys').textContent = totalKeys;

  // Count active today (last_login within 24h)
  const now = new Date();
  const activeToday = _usersData.filter(u => {
    if (!u.last_login) return false;
    const lastLogin = new Date(u.last_login);
    return (now - lastLogin) < 24 * 60 * 60 * 1000;
  }).length;
  document.getElementById('users-active-today').textContent = activeToday;

  renderUsersTable();
}

function renderUsersTable() {
  const tbody = document.getElementById('users-table-body');

  let users = _usersData;
  if (_usersFilter) {
    const f = _usersFilter.toLowerCase();
    users = users.filter(u =>
      (u.name || '').toLowerCase().includes(f) ||
      (u.email || '').toLowerCase().includes(f) ||
      (u.username || '').toLowerCase().includes(f)
    );
  }

  if (users.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No users found</td></tr>';
    return;
  }

  const rows = users.map(u => {
    const avatar = u.picture
      ? `<img src="${escHtml(u.picture)}" alt="" class="user-avatar">`
      : `<div class="user-avatar-placeholder">${(u.name || '?').charAt(0).toUpperCase()}</div>`;
    const username = u.username ? `@${escHtml(u.username)}` : '<span style="color:var(--text3)">—</span>';
    const created = u.created_at ? formatDateShort(u.created_at) : '—';
    const lastLogin = u.last_login ? formatDateShort(u.last_login) : '—';

    return `<tr>
      <td>
        <div class="user-cell">
          ${avatar}
          <div class="user-info">
            <div class="user-name">${escHtml(u.name || 'Unknown')}</div>
            <div class="user-email">${escHtml(u.email || '')}</div>
          </div>
        </div>
      </td>
      <td>${escHtml(u.email || '—')}</td>
      <td>${username}</td>
      <td>${created}</td>
      <td>${lastLogin}</td>
      <td>${u.api_key_count || 0}</td>
      <td>
        <button class="action-btn" onclick="showUserTrailModal('${escHtml(u.id)}')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
          View Trail
        </button>
      </td>
    </tr>`;
  }).join('');

  tbody.innerHTML = rows;
}

function filterUsers(query) {
  _usersFilter = query;
  renderUsersTable();
}

async function showUserTrailModal(userId) {
  const modal = document.getElementById('user-trail-modal');
  const user = _usersData.find(u => u.id === userId);
  if (!user) return;

  // Show modal with loading state
  modal.style.display = 'flex';
  document.getElementById('trail-modal-title').textContent = `Activity Trail: ${escHtml(user.name || user.email)}`;
  document.getElementById('trail-table-body').innerHTML = '<tr><td colspan="5" class="empty-state">Loading...</td></tr>';
  document.getElementById('trail-calls-24h').textContent = '—';
  document.getElementById('trail-unique-routes').textContent = '—';
  document.getElementById('trail-unique-paths').innerHTML = '';

  // Update user info
  const avatar = user.picture
    ? `<img src="${escHtml(user.picture)}" alt="" style="width:48px;height:48px;border-radius:50%;object-fit:cover;border:2px solid var(--border);">`
    : `<div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:600;">${(user.name || '?').charAt(0).toUpperCase()}</div>`;

  document.getElementById('trail-user-info').innerHTML = `
    <div style="display:flex;align-items:center;gap:16px;">
      ${avatar}
      <div>
        <div style="font-size:16px;font-weight:600;color:var(--text);">${escHtml(user.name || 'Unknown')}</div>
        <div style="font-size:13px;color:var(--text2);">${escHtml(user.email || '')}</div>
        ${user.username ? `<div style="font-size:12px;color:var(--blue);margin-top:4px;">@${escHtml(user.username)}</div>` : ''}
      </div>
    </div>
  `;

  // Fetch trail data
  const trailData = await apiFetch(`/users/${encodeURIComponent(userId)}/trail?hours=24&limit=50`);
  if (!trailData || trailData.error) {
    document.getElementById('trail-table-body').innerHTML = `<tr><td colspan="5" class="empty-state">${trailData?.error || 'Failed to load trail'}</td></tr>`;
    return;
  }

  // Update stats
  document.getElementById('trail-calls-24h').textContent = trailData.total_calls_24h || 0;
  document.getElementById('trail-unique-routes').textContent = (trailData.unique_paths || []).length;

  // Show unique paths as tags
  const uniquePaths = trailData.unique_paths || [];
  if (uniquePaths.length > 0) {
    document.getElementById('trail-unique-paths').innerHTML = uniquePaths.map(p =>
      `<span class="path-tag">${escHtml(p)}</span>`
    ).join('');
  } else {
    document.getElementById('trail-unique-paths').innerHTML = '<span style="color:var(--text3);">No routes accessed</span>';
  }

  // Render trail table
  const trail = trailData.trail || [];
  if (trail.length === 0) {
    document.getElementById('trail-table-body').innerHTML = '<tr><td colspan="5" class="empty-state">No activity in the last 24 hours</td></tr>';
    return;
  }

  const rows = trail.map(t => {
    const ts = new Date(t.ts);
    const timeStr = ts.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const dateStr = ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const statusClass = t.status >= 400 ? 'style="color:var(--red)"' : (t.status >= 300 ? 'style="color:var(--amber)"' : 'style="color:var(--green)"');

    return `<tr>
      <td title="${ts.toISOString()}">${dateStr} ${timeStr}</td>
      <td><span style="font-family:var(--mono);font-size:11px;padding:2px 6px;background:var(--surface2);border-radius:4px;">${t.method || 'GET'}</span></td>
      <td style="font-family:var(--mono);font-size:12px;">${escHtml(t.path || '/')}</td>
      <td ${statusClass}>${t.status || '—'}</td>
      <td>${Math.round(t.latency_ms || 0)}ms</td>
    </tr>`;
  }).join('');

  document.getElementById('trail-table-body').innerHTML = rows;
}

function closeUserTrailModal() {
  document.getElementById('user-trail-modal').style.display = 'none';
}

function formatDateShort(dateStr) {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  const now = new Date();
  const diffMs = now - d;
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays}d ago`;

  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
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

// ── Scanner page ──────────────────────────────────────────────────
async function loadScanner() {
  const data = await apiFetch('/scanner/analytics');
  if (!data) return;

  if (data.error) {
    console.error('Scanner analytics error:', data.error);
    return;
  }

  // Update summary cards
  const jobs = data.scanner_jobs || {};
  const repos = data.user_repos || {};
  const runs = data.scan_runs || {};
  const stars = data.community_stars || {};
  const visibility = data.index_visibility || {};

  document.getElementById('scanner-jobs-total').textContent = formatNum(jobs.total || 0);
  document.getElementById('scanner-repos-total').textContent = formatNum(repos.total || 0);
  document.getElementById('scanner-runs-total').textContent = formatNum(runs.total || 0);
  document.getElementById('scanner-stars-total').textContent = formatNum(stars.total_stars || 0);
  document.getElementById('scanner-visibility-total').textContent = formatNum(visibility.total || 0);
  document.getElementById('scanner-stargazers').textContent = formatNum(stars.unique_users || 0);

  // Recent scanner jobs table
  const jobsRows = (jobs.recent || []).map(j => {
    const statusBadge = (status) => {
      const color = status === 'complete' ? 'var(--green)' : status === 'failed' ? 'var(--red)' : status === 'running' ? 'var(--amber)' : 'var(--text2)';
      return `<span style="color:${color};font-weight:600">${status || 'N/A'}</span>`;
    };
    const updated = j.updated_at ? formatDateShort(j.updated_at) : '—';
    return `<tr>
      <td style="font-family:'JetBrains Mono',monospace;font-size:11px">${escHtml(j.job_id || 'N/A')}</td>
      <td>${escHtml(j.username || 'N/A')}</td>
      <td>${escHtml(j.org || '')}/${escHtml(j.repo || '')}</td>
      <td>${statusBadge(j.phase1_status)}</td>
      <td>${statusBadge(j.phase2_status)}</td>
      <td>${updated}</td>
    </tr>`;
  }).join('');
  document.getElementById('scanner-jobs-body').innerHTML = jobsRows || '<tr><td colspan="6" style="color:var(--text2)">No scanner jobs found</td></tr>';

  // Recent scan runs table
  const runsRows = (runs.latest || []).map(r => {
    const status = r.status || 'unknown';
    const statusColor = status === 'success' ? 'var(--green)' : status === 'failed' ? 'var(--red)' : 'var(--text2)';
    const scanned = r.last_scanned_at ? formatDateShort(r.last_scanned_at) : '—';
    return `<tr>
      <td>${escHtml(r.org_id || 'N/A')}</td>
      <td>${escHtml(r.repo || 'N/A')}</td>
      <td style="color:${statusColor};font-weight:600">${status}</td>
      <td>${scanned}</td>
    </tr>`;
  }).join('');
  document.getElementById('scanner-runs-body').innerHTML = runsRows || '<tr><td colspan="4" style="color:var(--text2)">No scan runs found</td></tr>';

  // Top starred repos table
  const starsRows = (stars.top_repos || []).map(s => {
    const org = (s._id && s._id.org) || 'N/A';
    const repo = (s._id && s._id.repo) || 'N/A';
    return `<tr>
      <td>${escHtml(org)}</td>
      <td>${escHtml(repo)}</td>
      <td style="color:#fbbf24;font-weight:600">${s.star_count || 0}</td>
    </tr>`;
  }).join('');
  document.getElementById('scanner-stars-body').innerHTML = starsRows || '<tr><td colspan="3" style="color:var(--text2)">No stars data found</td></tr>';

  // Repos per user table
  const reposPerUserRows = (repos.per_user || []).map(u =>
    `<tr><td>${escHtml(u._id || 'N/A')}</td><td>${u.repo_count || 0}</td></tr>`
  ).join('');
  document.getElementById('scanner-repos-per-user-body').innerHTML = reposPerUserRows || '<tr><td colspan="2" style="color:var(--text2)">No user repos found</td></tr>';

  // Recent user repos table
  const recentReposRows = (repos.recent || []).map(r =>
    `<tr>
      <td>${escHtml(r.username || 'N/A')}</td>
      <td>${escHtml(r.github_org || 'N/A')}</td>
      <td>${escHtml(r.repo || 'N/A')}</td>
      <td>${escHtml(r.branch || 'main')}</td>
    </tr>`
  ).join('');
  document.getElementById('scanner-recent-repos-body').innerHTML = recentReposRows || '<tr><td colspan="4" style="color:var(--text2)">No recent repos found</td></tr>';

  // Job status breakdown table
  const statusRows = (jobs.by_status || []).map(s => {
    const p1 = (s._id && s._id.phase1) || 'N/A';
    const p2 = (s._id && s._id.phase2) || 'N/A';
    return `<tr><td>${escHtml(p1)}</td><td>${escHtml(p2)}</td><td>${s.count || 0}</td></tr>`;
  }).join('');
  document.getElementById('scanner-job-status-body').innerHTML = statusRows || '<tr><td colspan="3" style="color:var(--text2)">No status data found</td></tr>';
}


// ═════════════════════════════════════════════════════════════════
// Outreach — GitHub Email Scraper + Email Sender
// ═════════════════════════════════════════════════════════════════

let _outreachJobsData = [];
let _outreachSelectedJobId = null;
let _outreachSSE = null;

async function apiPost(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: {'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json'},
    credentials: 'include',
    body: JSON.stringify(body),
  });
  if (r.status === 401) { logout(); return null; }
  return await r.json();
}

async function apiDelete(path) {
  const r = await fetch(`${API}${path}`, {
    method: 'DELETE',
    headers: {'Authorization': `Bearer ${token}`},
    credentials: 'include',
  });
  if (r.status === 401) { logout(); return null; }
  return await r.json();
}

function switchOutreachTab(tab) {
  document.querySelectorAll('.outreach-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.outreach-subtab').forEach(t => t.classList.remove('active'));
  const tabBtn = document.querySelector(`[data-otab="${tab}"]`);
  if (tabBtn) tabBtn.classList.add('active');
  const panel = document.getElementById('otab-' + tab);
  if (panel) panel.classList.add('active');

  if (tab === 'scraper') loadOutreachJobs();
  if (tab === 'send') { loadOutreachDraft(); populateJobSelects(); }
  if (tab === 'analytics') populateJobSelects();
}

async function loadOutreach() {
  loadOutreachPATs();
  loadOutreachJobs();
  loadOutreachDraft();
  populateJobSelects();
}

// ── PATs ──────────────────────────────────────────────────────────

async function loadOutreachPATs() {
  const data = await apiFetch('/outreach/pats');
  if (!data) return;
  const pats = data.pats || [];
  const tbody = document.getElementById('pats-table-body');
  if (pats.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No PATs added yet. Add a GitHub Personal Access Token above.</td></tr>';
    return;
  }
  tbody.innerHTML = pats.map(p => {
    const resetAt = p.reset_at ? new Date(p.reset_at).toLocaleString() : '—';
    const added = p.added_at ? formatDateShort(p.added_at) : '—';
    const remaining = p.remaining != null ? p.remaining : '?';
    const isActive = p.active !== false;
    const remainingColor = !isActive ? 'var(--red)' : remaining > 1000 ? 'var(--green)' : remaining > 100 ? 'var(--amber)' : 'var(--red)';
    const statusBadge = isActive
      ? `<span style="color:var(--green);font-size:11px">Active</span>`
      : `<span style="color:var(--red);font-size:11px">Invalid</span>`;
    return `<tr>
      <td>${escHtml(p.label || '')} ${statusBadge}</td>
      <td style="font-family:var(--mono);font-size:11px;color:var(--text3)">${escHtml(p.token_masked || '')}</td>
      <td style="color:${remainingColor};font-weight:600">${isActive ? remaining : 'N/A'}</td>
      <td style="font-size:12px">${resetAt}</td>
      <td>${added}</td>
      <td><button class="outreach-btn outreach-btn-danger" style="padding:4px 10px;font-size:11px" onclick="deletePAT('${p._id}')">Remove</button></td>
    </tr>`;
  }).join('');
}

async function addPAT() {
  const tokenVal = document.getElementById('pat-token-input').value.trim();
  const label = document.getElementById('pat-label-input').value.trim();
  const errEl = document.getElementById('pat-error');
  errEl.textContent = '';
  errEl.style.color = '';

  if (!tokenVal) { errEl.textContent = 'Token is required'; errEl.style.color = 'var(--red)'; return; }

  const data = await apiPost('/outreach/pats', { token: tokenVal, label });
  if (data && data.error) {
    errEl.textContent = data.detail || data.error || 'Failed to add PAT';
    errEl.style.color = 'var(--red)';
    return;
  }
  if (data && data.detail) {
    errEl.textContent = data.detail;
    errEl.style.color = 'var(--red)';
    return;
  }

  errEl.textContent = 'PAT validated and added successfully (user + repo access confirmed)';
  errEl.style.color = '#00cc66';
  document.getElementById('pat-token-input').value = '';
  document.getElementById('pat-label-input').value = '';
  loadOutreachPATs();
}

async function deletePAT(patId) {
  if (!confirm('Remove this PAT?')) return;
  await apiDelete(`/outreach/pats/${patId}`);
  loadOutreachPATs();
}

async function deleteAllPATs() {
  if (!confirm('Delete ALL PATs? You will need to add fresh ones.')) return;
  await apiDelete('/outreach/pats/all');
  loadOutreachPATs();
}

// ── Jobs ──────────────────────────────────────────────────────────

async function loadOutreachJobs() {
  const data = await apiFetch('/outreach/jobs');
  if (!data) return;
  _outreachJobsData = data.jobs || [];
  renderJobsTable();
  populateJobSelects();
}

function renderJobsTable() {
  const tbody = document.getElementById('jobs-table-body');
  if (_outreachJobsData.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No scraping jobs yet</td></tr>';
    return;
  }
  tbody.innerHTML = _outreachJobsData.map(j => {
    const statusColor = j.status === 'running' ? 'var(--amber)' : j.status === 'completed' ? 'var(--green)' : j.status === 'error' ? 'var(--red)' : 'var(--text3)';
    const isRunning = j.is_running || j.status === 'running';
    const actions = isRunning
      ? `<button class="outreach-btn outreach-btn-danger" style="padding:4px 10px;font-size:11px" onclick="event.stopPropagation();stopJob('${j._id}')">Stop</button>`
      : `<button class="outreach-btn" style="padding:4px 10px;font-size:11px" onclick="event.stopPropagation();resumeJob('${j._id}')">Resume</button>`;
    const errorRow = j.error ? `<div style="color:var(--red);font-size:11px;margin-top:4px">${escHtml(j.error)}</div>` : '';

    return `<tr onclick="selectJob('${j._id}')" style="cursor:pointer">
      <td style="font-size:12px">${escHtml(j.repo_slug || j.repo_url || '')}</td>
      <td><span style="color:${statusColor};font-weight:600">${j.status || 'unknown'}</span>${errorRow}</td>
      <td style="font-weight:600;color:var(--accent)">${j.emails_found || 0}</td>
      <td>${j.processed_index || 0} / ${j.total_stargazers_fetched || 0}</td>
      <td>${j.target_email_count || 0}</td>
      <td>${actions} <button class="outreach-btn" style="padding:4px 10px;font-size:11px;margin-left:4px" onclick="event.stopPropagation();selectJob('${j._id}')">View Emails</button></td>
    </tr>`;
  }).join('');
}

async function startScrapeJob() {
  const repo = document.getElementById('scraper-repo-input').value.trim();
  const target = parseInt(document.getElementById('scraper-target-input').value) || 500;
  const errEl = document.getElementById('scraper-error');
  errEl.textContent = '';

  if (!repo) { errEl.textContent = 'Repository URL is required'; return; }

  const data = await apiPost('/outreach/jobs/start', { repo_url: repo, target_email_count: target });
  if (data && (data.detail || data.error)) {
    errEl.textContent = data.detail || data.error;
    return;
  }

  loadOutreachJobs();
  if (data && data._id) {
    selectJob(data._id);
    connectJobSSE(data._id);
  }
}

async function stopJob(jobId) {
  await apiPost(`/outreach/jobs/${jobId}/stop`, {});
  if (_outreachSSE) { _outreachSSE.close(); _outreachSSE = null; }
  setTimeout(loadOutreachJobs, 500);
}

async function resumeJob(jobId) {
  const data = await apiPost(`/outreach/jobs/${jobId}/resume`, {});
  if (data && (data.detail || data.error)) {
    document.getElementById('scraper-error').textContent = data.detail || data.error;
    return;
  }
  loadOutreachJobs();
  selectJob(jobId);
  connectJobSSE(jobId);
}

async function selectJob(jobId) {
  _outreachSelectedJobId = jobId;
  const panel = document.getElementById('panel-job-emails');
  panel.style.display = '';

  const job = _outreachJobsData.find(j => j._id === jobId);
  if (job) {
    document.getElementById('job-emails-title').textContent = `Emails: ${job.repo_slug || ''}`;
    const progressWrap = document.getElementById('scraper-progress-wrap');
    progressWrap.style.display = '';
    const pct = job.target_email_count > 0 ? Math.min(100, ((job.emails_found || 0) / job.target_email_count) * 100) : 0;
    document.getElementById('scraper-progress-fill').style.width = pct + '%';
    document.getElementById('scraper-progress-text').textContent = `${job.emails_found || 0} / ${job.target_email_count} emails found | ${job.processed_index || 0} users scanned`;

    if (job.error) {
      showScraperStatus(job.error, 'error');
    } else if (job.status === 'running') {
      showScraperStatus('Scraping in progress...', 'status');
    } else if (job.status === 'completed') {
      showScraperStatus(`Completed. Found ${job.emails_found || 0} emails.`, 'done');
    } else {
      showScraperStatus('', 'status');
    }
  }

  const data = await apiFetch(`/outreach/jobs/${jobId}/emails`);
  if (!data) return;
  const emails = data.emails || [];
  document.getElementById('job-emails-count').textContent = emails.length;

  const tbody = document.getElementById('job-emails-body');
  if (emails.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No emails collected yet</td></tr>';
    return;
  }
  tbody.innerHTML = emails.map(e => {
    const foundAt = e.scraped_at ? new Date(e.scraped_at).toLocaleString() : '—';
    const sentBadge = e.sent
      ? '<span style="color:var(--green);font-weight:600">Yes</span>'
      : '<span style="color:var(--text3)">No</span>';
    return `<tr>
      <td>${escHtml(e.username || '')}</td>
      <td style="font-family:var(--mono);font-size:12px">${escHtml(e.email || '')}</td>
      <td style="font-size:12px">${foundAt}</td>
      <td>${sentBadge}</td>
    </tr>`;
  }).join('');

  if (job && (job.is_running || job.status === 'running')) {
    connectJobSSE(jobId);
  }
}

function connectJobSSE(jobId) {
  if (_outreachSSE) { _outreachSSE.close(); _outreachSSE = null; }

  const url = `${API}/outreach/jobs/${jobId}/stream`;
  const controller = new AbortController();
  _outreachSSE = { close: () => controller.abort() };

  fetch(url, {
    headers: { 'Authorization': `Bearer ${token}` },
    credentials: 'include',
    signal: controller.signal,
  }).then(response => {
    if (!response.ok) return;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let nextEventType = null;

    function readChunk() {
      reader.read().then(({ done, value }) => {
        if (done) { _outreachSSE = null; loadOutreachJobs(); return; }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            nextEventType = line.substring(7).trim();
          } else if (line.startsWith('data: ')) {
            const dataStr = line.substring(6);
            try {
              const parsed = JSON.parse(dataStr);

              if (nextEventType === 'done' || nextEventType === 'error') {
                const msg = parsed.message || parsed.error || `Job ${parsed.status || 'finished'}`;
                showScraperStatus(msg, nextEventType === 'error' ? 'error' : 'done');
                _outreachSSE = null;
                loadOutreachJobs();
                nextEventType = null;
                return;
              } else if (nextEventType === 'status') {
                showScraperStatus(parsed.message || '', parsed._type || 'status');
              } else {
                if (parsed._type && parsed._type !== 'email') {
                  showScraperStatus(parsed.message || '', parsed._type);
                } else {
                  appendEmailRow(parsed);
                  updateJobProgress();
                }
              }
            } catch(e) {}
            nextEventType = null;
          }
        }
        readChunk();
      }).catch(() => { _outreachSSE = null; });
    }
    readChunk();
  }).catch(() => { _outreachSSE = null; });
}

function showScraperStatus(message, type) {
  const errEl = document.getElementById('scraper-error');
  if (!errEl) return;
  if (!message) { errEl.style.display = 'none'; errEl.textContent = ''; return; }
  errEl.style.display = 'block';
  errEl.style.padding = '10px 14px';
  errEl.style.borderRadius = '6px';
  errEl.style.marginTop = '10px';
  errEl.style.fontSize = '13px';
  errEl.style.fontWeight = '500';
  if (type === 'error') {
    errEl.style.color = '#ff4444';
    errEl.style.background = 'rgba(255,68,68,0.08)';
    errEl.style.border = '1px solid rgba(255,68,68,0.3)';
    errEl.textContent = '⚠ ' + message;
  } else if (type === 'warning') {
    errEl.style.color = '#ffaa00';
    errEl.style.background = 'rgba(255,170,0,0.08)';
    errEl.style.border = '1px solid rgba(255,170,0,0.3)';
    errEl.textContent = message;
  } else if (type === 'done') {
    errEl.style.color = '#00cc66';
    errEl.style.background = 'rgba(0,204,102,0.08)';
    errEl.style.border = '1px solid rgba(0,204,102,0.3)';
    errEl.textContent = message;
  } else {
    errEl.style.color = 'var(--text2)';
    errEl.style.background = 'rgba(255,255,255,0.03)';
    errEl.style.border = '1px solid rgba(255,255,255,0.1)';
    errEl.textContent = message;
  }
}

function appendEmailRow(emailDoc) {
  const tbody = document.getElementById('job-emails-body');
  if (!tbody) return;
  if (tbody.querySelector('.empty-state')) tbody.innerHTML = '';

  const tr = document.createElement('tr');
  const foundAt = emailDoc.scraped_at ? new Date(emailDoc.scraped_at).toLocaleString() : 'Now';
  tr.innerHTML = `
    <td>${escHtml(emailDoc.username || '')}</td>
    <td style="font-family:var(--mono);font-size:12px">${escHtml(emailDoc.email || '')}</td>
    <td style="font-size:12px">${foundAt}</td>
    <td><span style="color:var(--text3)">No</span></td>
  `;
  tbody.prepend(tr);

  const countEl = document.getElementById('job-emails-count');
  if (countEl) {
    const current = parseInt(countEl.textContent) || 0;
    countEl.textContent = current + 1;
  }
}

function updateJobProgress() {
  const countEl = document.getElementById('job-emails-count');
  const count = parseInt(countEl?.textContent) || 0;
  const job = _outreachJobsData.find(j => j._id === _outreachSelectedJobId);
  const target = job ? job.target_email_count : 500;
  const pct = target > 0 ? Math.min(100, (count / target) * 100) : 0;
  const fill = document.getElementById('scraper-progress-fill');
  const text = document.getElementById('scraper-progress-text');
  if (fill) fill.style.width = pct + '%';
  if (text) text.textContent = `${count} / ${target}`;
  document.getElementById('scraper-progress-wrap').style.display = '';
}

// ── Draft & Send ──────────────────────────────────────────────────

async function loadOutreachDraft() {
  const data = await apiFetch('/outreach/drafts');
  if (!data) return;
  if (data.subject) document.getElementById('draft-subject').value = data.subject;
  if (data.body_html) document.getElementById('draft-body').value = data.body_html;
}

async function saveDraft() {
  const subject = document.getElementById('draft-subject').value;
  const body_html = document.getElementById('draft-body').value;
  const statusEl = document.getElementById('draft-status');
  statusEl.textContent = 'Saving...';
  const data = await apiPost('/outreach/drafts', { subject, body_html });
  statusEl.textContent = data && !data.error ? 'Saved!' : 'Failed to save';
  setTimeout(() => { statusEl.textContent = ''; }, 3000);
}

function previewDraft() {
  const subject = document.getElementById('draft-subject').value;
  const body = document.getElementById('draft-body').value;
  const preview = document.getElementById('panel-draft-preview');
  preview.style.display = '';
  const rendered = body.replace(/\{\{username\}\}/g, '<strong>@example_user</strong>');
  document.getElementById('draft-preview-content').innerHTML = `
    <div style="margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid var(--border)">
      <strong style="color:var(--text)">Subject:</strong>
      <span style="color:var(--text2);margin-left:8px">${escHtml(subject).replace(/\{\{username\}\}/g, '<strong style="color:var(--accent)">@example_user</strong>')}</span>
    </div>
    <div>${rendered}</div>
  `;
}

function populateJobSelects() {
  const selects = ['send-job-select', 'analytics-job-select'];
  for (const id of selects) {
    const el = document.getElementById(id);
    if (!el) continue;
    const current = el.value;
    el.innerHTML = '<option value="">Select a scraping job...</option>';
    for (const j of _outreachJobsData) {
      const opt = document.createElement('option');
      opt.value = j._id;
      opt.textContent = `${j.repo_slug || j.repo_url} (${j.emails_found || 0} emails)`;
      el.appendChild(opt);
    }
    if (current) el.value = current;
  }
}

async function sendOutreachEmails() {
  const jobId = document.getElementById('send-job-select').value;
  const errEl = document.getElementById('send-error');
  const resultEl = document.getElementById('send-result');
  errEl.textContent = '';
  resultEl.style.display = 'none';

  if (!jobId) { errEl.textContent = 'Select a scraping job first'; return; }
  if (!confirm('Send emails to ALL unsent recipients in this job?')) return;

  errEl.textContent = 'Sending... this may take a while';
  const data = await apiPost('/outreach/send', { job_id: jobId });
  errEl.textContent = '';

  if (data && (data.detail || data.error)) {
    errEl.textContent = data.detail || data.error;
    return;
  }

  if (data) {
    resultEl.style.display = '';
    resultEl.textContent = `Sent ${data.sent || 0} of ${data.total_attempted || 0} emails.` +
      (data.errors && data.errors.length > 0 ? ` ${data.errors.length} errors.` : '');
  }
}

// ── Analytics ─────────────────────────────────────────────────────

async function loadOutreachAnalytics() {
  const jobId = document.getElementById('analytics-job-select').value;
  if (!jobId) return;

  const [analyticsData, emailsData] = await Promise.all([
    apiFetch(`/outreach/analytics/${jobId}`),
    apiFetch(`/outreach/jobs/${jobId}/emails`),
  ]);

  if (analyticsData) {
    document.getElementById('oa-sent').textContent = analyticsData.sent || 0;
    document.getElementById('oa-opens').textContent = analyticsData.opened || 0;
    document.getElementById('oa-clicks').textContent = analyticsData.clicked || 0;
    document.getElementById('oa-total').textContent = analyticsData.total_emails || 0;
    document.getElementById('oa-open-rate').textContent = (analyticsData.open_rate || 0) + '%';
    document.getElementById('oa-click-rate').textContent = (analyticsData.click_rate || 0) + '%';
  }

  if (emailsData) {
    const emails = emailsData.emails || [];
    const tbody = document.getElementById('oa-emails-body');
    if (emails.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No emails for this job</td></tr>';
      return;
    }
    tbody.innerHTML = emails.map(e => {
      const sentBadge = e.sent
        ? `<span style="color:var(--green)">Yes</span>`
        : `<span style="color:var(--text3)">No</span>`;
      const openBadge = e.opened
        ? `<span style="color:var(--green)">Yes</span>`
        : `<span style="color:var(--text3)">No</span>`;
      const clickBadge = e.clicked
        ? `<span style="color:var(--green)">Yes</span>`
        : `<span style="color:var(--text3)">No</span>`;
      return `<tr>
        <td>${escHtml(e.username || '')}</td>
        <td style="font-family:var(--mono);font-size:12px">${escHtml(e.email || '')}</td>
        <td>${sentBadge}</td>
        <td>${openBadge}</td>
        <td>${clickBadge}</td>
      </tr>`;
    }).join('');
  }
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
        {label:'Requests',data:counts,backgroundColor:'rgba(16, 185, 129, 0.5)',borderColor:'#10b981',borderWidth:1,borderRadius:4},
        {label:'Errors',data:errors,backgroundColor:'rgba(239, 68, 68, 0.5)',borderColor:'#ef4444',borderWidth:1,borderRadius:4}
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
        {label:'LLM Calls',data:calls,borderColor:'#8b5cf6',backgroundColor:'rgba(139, 92, 246, 0.1)',fill:true,tension:0.4,pointBackgroundColor:'#8b5cf6',pointStyle:'circle'},
        {label:'Tokens (÷100)',data:tokens.map(t=>Math.round(t/100)),borderColor:'#10b981',backgroundColor:'rgba(16, 185, 129, 0.1)',fill:true,tension:0.4,pointBackgroundColor:'#10b981',pointStyle:'circle'}
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
        {label:'Views',data:viewCounts,borderColor:'#3b82f6',backgroundColor:'rgba(59, 130, 246, 0.1)',fill:true,tension:0.4,pointBackgroundColor:'#3b82f6',pointStyle:'circle'},
        {label:'Clones',data:cloneCounts,borderColor:'#10b981',backgroundColor:'rgba(16, 185, 129, 0.1)',fill:true,tension:0.4,pointBackgroundColor:'#10b981',pointStyle:'circle'}
      ]
    },
    options: chartOpts('GitHub Traffic (14d)')
  });
}

function chartOpts(title) {
  return {
    responsive:true, maintainAspectRatio:false,
    plugins:{
      legend:{labels:{color:'#a1a1aa',font:{family:'Inter',size:11},usePointStyle:true,pointStyle:'circle',padding:16}},
      title:{display:false}
    },
    scales:{
      x:{ticks:{color:'#71717a',font:{family:'Inter',size:10}},grid:{color:'rgba(63, 63, 70, 0.3)',drawBorder:false}},
      y:{ticks:{color:'#71717a',font:{family:'Inter',size:10}},grid:{color:'rgba(63, 63, 70, 0.3)',drawBorder:false},beginAtZero:true}
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
