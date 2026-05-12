import { createChart, ColorType, CrosshairMode, CandlestickSeries, HistogramSeries, LineSeries } from 'lightweight-charts';
import './style.css';

const API = '';
let chart, candleSeries, volumeSeries;
let rsiChart, rsiSeries, macdChart, macdLineSeries, macdSignalSeries, macdHistSeries;
let overlayLines = {};
let currentSymbol = 'RELIANCE';
let currentMarket = 'stock';
let currentInterval = '1d';
let activeIndicators = new Set(['ema_21', 'ema_50', 'rsi', 'volume_sma']);

// Live streaming state
let liveWs = null;
let liveCandle = null;
let liveCandleStart = 0;
let lastLivePrice = 0;
let autoRefreshTimer = null;
let lastCandleCount = 0;
let updateRequested = false;
let throttleTimer = null;

// ─── INIT ───
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  initRsiChart();
  initMacdChart();
  setupEventListeners();
  loadSymbol('RELIANCE', 'stock');
  connectScannerWS();
  triggerScan();
});

// ─── CHART SETUP ───
function initChart() {
  const container = document.getElementById('chart-container');
  chart = createChart(container, {
    layout: { background: { type: ColorType.Solid, color: '#0a0e17' }, textColor: '#94a3b8', fontFamily: "'Inter', sans-serif", fontSize: 11 },
    grid: { vertLines: { color: '#1e293b22' }, horzLines: { color: '#1e293b22' } },
    crosshair: { mode: CrosshairMode.Normal, vertLine: { color: '#6366f180', width: 1, style: 2 }, horzLine: { color: '#6366f180', width: 1, style: 2 } },
    rightPriceScale: { borderColor: '#1e293b', scaleMargins: { top: 0.1, bottom: 0.2 } },
    timeScale: { borderColor: '#1e293b', timeVisible: true, secondsVisible: false },
    handleScroll: { vertTouchDrag: false },
  });
  candleSeries = chart.addSeries(CandlestickSeries, { upColor: '#22c55e', downColor: '#ef4444', borderUpColor: '#22c55e', borderDownColor: '#ef4444', wickUpColor: '#22c55e88', wickDownColor: '#ef444488' });
  volumeSeries = chart.addSeries(HistogramSeries, { priceFormat: { type: 'volume' }, priceScaleId: 'vol' });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
  new ResizeObserver(entries => { const { width, height } = entries[0].contentRect; chart.applyOptions({ width, height }); }).observe(container);
}

function initRsiChart() {
  const container = document.getElementById('rsi-container');
  rsiChart = createChart(container, {
    layout: { background: { type: ColorType.Solid, color: '#0a0e17' }, textColor: '#94a3b8', fontFamily: "'Inter', sans-serif", fontSize: 10 },
    grid: { vertLines: { color: '#1e293b22' }, horzLines: { color: '#1e293b22' } },
    rightPriceScale: { borderColor: '#1e293b' },
    timeScale: { visible: false },
    crosshair: { mode: CrosshairMode.Normal },
    height: 120,
  });
  rsiSeries = rsiChart.addSeries(LineSeries, { color: '#a78bfa', lineWidth: 1.5, priceFormat: { precision: 1 } });
  new ResizeObserver(entries => { rsiChart.applyOptions({ width: entries[0].contentRect.width }); }).observe(container);
}

function initMacdChart() {
  const container = document.getElementById('macd-container');
  macdChart = createChart(container, {
    layout: { background: { type: ColorType.Solid, color: '#0a0e17' }, textColor: '#94a3b8', fontFamily: "'Inter', sans-serif", fontSize: 10 },
    grid: { vertLines: { color: '#1e293b22' }, horzLines: { color: '#1e293b22' } },
    rightPriceScale: { borderColor: '#1e293b' },
    timeScale: { visible: false },
    crosshair: { mode: CrosshairMode.Normal },
    height: 120,
  });
  macdLineSeries = macdChart.addSeries(LineSeries, { color: '#06b6d4', lineWidth: 1.5 });
  macdSignalSeries = macdChart.addSeries(LineSeries, { color: '#f59e0b', lineWidth: 1.5 });
  macdHistSeries = macdChart.addSeries(HistogramSeries, { color: '#6366f1' });
  new ResizeObserver(entries => { macdChart.applyOptions({ width: entries[0].contentRect.width }); }).observe(container);
}

// ─── LOAD SYMBOL ───
async function loadSymbol(symbol, market) {
  currentSymbol = symbol;
  currentMarket = market;
  document.getElementById('current-symbol').textContent = symbol;
  document.getElementById('current-price').textContent = '⏳';
  document.getElementById('current-change').textContent = '';
  disconnectLiveStream();
  const period = currentMarket === 'stock' ? '6mo' : '';
  const limit = currentMarket === 'crypto' ? 500 : '';
  let url;
  if (market === 'stock') {
    url = `${API}/api/stocks/${symbol}/analysis?interval=${currentInterval}&period=${period || '6mo'}`;
  } else {
    url = `${API}/api/crypto/${symbol}/analysis?interval=${currentInterval}&limit=${limit || 500}`;
  }
  try {
    const res = await fetch(url);
    const data = await res.json();
    if (data.error) { console.error(data.error); return; }
    lastCandleCount = data.candle_count || (data.candles || []).length;
    const src = market === 'crypto' ? 'Binance' : 'NSE/BSE via Yahoo';
    document.getElementById('data-source').textContent = `${lastCandleCount} candles @ ${currentInterval} · ${src}`;
    renderChart(data);
    renderIndicators(data.indicators || {});
    renderAnalysis(data);
    if (market === 'crypto') {
      connectLiveStream(symbol);
      fetchMultiTimeframe(symbol);
    } else {
      document.getElementById('multi-tf-strip').classList.add('hidden');
    }
  } catch (e) { console.error('Failed to load:', e); }
}

function parseTime(ts) {
  if (!ts) return 0;
  const d = new Date(ts.replace(' ', 'T'));
  if (isNaN(d.getTime())) return 0;
  return Math.floor(d.getTime() / 1000);
}

function renderChart(data) {
  const candles = (data.candles || []).map(c => ({ time: parseTime(c.timestamp), open: c.open, high: c.high, low: c.low, close: c.close })).filter(c => c.time > 0);
  const volumes = (data.candles || []).map(c => ({ time: parseTime(c.timestamp), value: c.volume || 0, color: c.close >= c.open ? '#22c55e30' : '#ef444430' })).filter(c => c.time > 0);
  if (!candles.length) return;
  candleSeries.setData(candles);
  volumeSeries.setData(volumes);
  chart.timeScale().fitContent();
  const last = data.candles[data.candles.length - 1];
  const prev = data.candles.length > 1 ? data.candles[data.candles.length - 2] : last;
  const price = last.close;
  const change = ((price - prev.close) / prev.close * 100).toFixed(2);
  document.getElementById('current-price').textContent = formatPrice(price);
  const changeEl = document.getElementById('current-change');
  changeEl.textContent = `${change >= 0 ? '+' : ''}${change}%`;
  changeEl.className = `symbol-change ${change >= 0 ? 'up' : 'down'}`;
}

function formatPrice(p) {
  if (currentMarket === 'crypto') {
    if (p >= 1) return '$' + p.toLocaleString('en-US', { maximumFractionDigits: 2 });
    return '$' + p.toFixed(6);
  }
  return '₹' + p.toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

// ─── OVERLAY INDICATORS ───
function renderIndicators(indicators) {
  Object.values(overlayLines).forEach(s => { try { chart.removeSeries(s); } catch(e){} });
  overlayLines = {};
  const colors = { ema_9: '#f59e0b', ema_21: '#06b6d4', ema_50: '#a78bfa', ema_200: '#ec4899', sma_20: '#8b5cf6', vwap: '#f97316', bb_upper: '#6366f144', bb_mid: '#6366f188', bb_lower: '#6366f144', supertrend: '#22d3ee' };
  for (const [name, data] of Object.entries(indicators)) {
    if (!activeIndicators.has(name)) continue;
    const info = getIndicatorInfo(name);
    if (!info || !info.overlay) continue;
    if (!data || !data.length) continue;
    const series = chart.addSeries(LineSeries, { color: colors[name] || '#6366f1', lineWidth: name.startsWith('bb_') ? 1 : 1.5, lineStyle: name.startsWith('bb_') ? 2 : 0 });
    const mapped = data.map(d => ({ time: parseTime(d.time), value: d.value })).filter(d => d.time > 0);
    if (mapped.length) { series.setData(mapped); overlayLines[name] = series; }
  }
  // RSI pane
  const rsiData = indicators.rsi;
  if (activeIndicators.has('rsi') && rsiData && rsiData.length) {
    document.getElementById('rsi-container').classList.remove('hidden');
    rsiSeries.setData(rsiData.map(d => ({ time: parseTime(d.time), value: d.value })).filter(d => d.time > 0));
    rsiChart.timeScale().fitContent();
  } else {
    document.getElementById('rsi-container').classList.add('hidden');
  }
  // MACD pane
  const ml = indicators.macd_line, ms = indicators.macd_signal, mh = indicators.macd_histogram;
  if (activeIndicators.has('macd_line') && ml && ml.length) {
    document.getElementById('macd-container').classList.remove('hidden');
    macdLineSeries.setData(ml.map(d => ({ time: parseTime(d.time), value: d.value })).filter(d => d.time > 0));
    if (ms) macdSignalSeries.setData(ms.map(d => ({ time: parseTime(d.time), value: d.value })).filter(d => d.time > 0));
    if (mh) macdHistSeries.setData(mh.map(d => ({ time: parseTime(d.time), value: d.value, color: d.value >= 0 ? '#22c55e80' : '#ef444480' })).filter(d => d.time > 0));
    macdChart.timeScale().fitContent();
  } else {
    document.getElementById('macd-container').classList.add('hidden');
  }
  renderIndicatorValues(indicators);
}

function renderIndicatorValues(indicators) {
  const el = document.getElementById('indicator-values');
  const rows = [];
  const show = ['rsi', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'atr', 'adx'];
  for (const name of show) {
    const d = indicators[name];
    if (d && d.length) {
      const val = d[d.length - 1].value;
      let cls = '';
      if (name === 'rsi') cls = val > 70 ? 'color:var(--red)' : val < 30 ? 'color:var(--green)' : '';
      rows.push(`<div class="ind-row"><span class="ind-label">${name.toUpperCase()}</span><span class="ind-val" style="${cls}">${val.toFixed(2)}</span></div>`);
    }
  }
  el.innerHTML = rows.join('');
}

// ─── ANALYSIS PANEL ───
function renderAnalysis(data) {
  const verdict = data.verdict || 'NEUTRAL';
  const vEl = document.getElementById('verdict-value');
  vEl.textContent = verdict;
  vEl.className = 'verdict-value ' + (verdict.includes('BUY') ? 'buy' : verdict.includes('SELL') ? 'sell' : 'neutral');
  const buy_c = data.signals?.length ? data.signals[0].buy_confluence || 0 : 0;
  const sell_c = data.signals?.length ? data.signals[0].sell_confluence || 0 : 0;
  document.getElementById('verdict-sub').textContent = `Buy strength: ${buy_c} | Sell strength: ${sell_c}`;
  const sEl = document.getElementById('analysis-signals');
  if (!data.signals || !data.signals.length) { sEl.innerHTML = '<div class="empty-state"><p>No active signals</p></div>'; return; }
  sEl.innerHTML = data.signals.slice(0, 8).map(s => renderSignalCard(s, currentSymbol)).join('');
}

function renderSignalCard(s, symbol) {
  const dir = (s.direction || '').toLowerCase();
  const dots = Array.from({length: 5}, (_, i) => `<span class="dot ${i < s.strength ? 'active' : ''}"></span>`).join('');
  return `<div class="signal-card ${dir}">
    <div class="signal-top"><span class="signal-symbol">${symbol || s.symbol || ''}</span><span class="signal-badge ${dir}">${s.direction}</span></div>
    <div class="signal-name">${s.name}</div>
    <div class="signal-reason">${s.reason}</div>
    <div class="signal-strength">${dots}</div>
  </div>`;
}

// ─── EVENTS ───
function setupEventListeners() {
  // Search
  const searchInput = document.getElementById('symbol-search');
  const searchResults = document.getElementById('search-results');
  let searchTimeout;
  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    const q = searchInput.value.trim();
    if (q.length < 1) { searchResults.classList.add('hidden'); return; }
    searchTimeout = setTimeout(() => doSearch(q), 300);
  });
  searchInput.addEventListener('focus', () => { if (searchInput.value.trim()) doSearch(searchInput.value.trim()); });
  document.addEventListener('click', (e) => { if (!e.target.closest('.search-section')) searchResults.classList.add('hidden'); });

  // Market toggle
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentMarket = btn.dataset.market;
    });
  });

  // Timeframe
  document.querySelectorAll('.tf-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentInterval = btn.dataset.tf;
      loadSymbol(currentSymbol, currentMarket);
    });
  });

  // Tabs
  document.querySelectorAll('.panel-tabs').forEach(tabGroup => {
    tabGroup.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const panel = tabGroup.parentElement;
        panel.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        panel.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const tabId = `tab-${btn.dataset.tab}`;
        const tabEl = panel.querySelector(`#${tabId}`);
        if (tabEl) tabEl.classList.add('active');
      });
    });
  });

  // Indicators modal
  document.getElementById('btn-indicators').addEventListener('click', () => showIndicatorModal());
  document.getElementById('close-indicators').addEventListener('click', () => document.getElementById('indicator-modal').classList.add('hidden'));
  document.getElementById('indicator-modal').addEventListener('click', (e) => { if (e.target.id === 'indicator-modal') e.target.classList.add('hidden'); });

  // Backtest
  document.getElementById('btn-backtest').addEventListener('click', () => {
    const rp = document.getElementById('right-panel');
    rp.querySelector('[data-tab="backtest"]').click();
  });
  document.getElementById('run-backtest-btn').addEventListener('click', runBacktest);

  // Scanner
  document.getElementById('run-scan-btn').addEventListener('click', triggerScan);
}

// ─── SEARCH ───
async function doSearch(q) {
  const results = document.getElementById('search-results');
  try {
    const [stockRes, cryptoRes] = await Promise.all([
      fetch(`${API}/api/stocks/search?q=${q}`).then(r => r.json()).catch(() => ({ results: [] })),
      fetch(`${API}/api/crypto/search?q=${q}`).then(r => r.json()).catch(() => ({ results: [] }))
    ]);
    const items = [];
    (stockRes.results || []).forEach(s => items.push(`<div class="search-result-item" data-symbol="${s.symbol}" data-market="stock"><span class="symbol">${s.symbol}</span><span class="exchange">NSE</span></div>`));
    (cryptoRes.results || []).forEach(s => items.push(`<div class="search-result-item" data-symbol="${s.symbol}" data-market="crypto"><span class="symbol">${s.display || s.symbol}</span><span class="exchange">Binance</span></div>`));
    if (!items.length) items.push('<div class="search-result-item"><span class="symbol">No results</span></div>');
    results.innerHTML = items.join('');
    results.classList.remove('hidden');
    results.querySelectorAll('[data-symbol]').forEach(item => {
      item.addEventListener('click', () => {
        loadSymbol(item.dataset.symbol, item.dataset.market);
        document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        document.querySelector(`.toggle-btn[data-market="${item.dataset.market}"]`).classList.add('active');
        results.classList.add('hidden');
        document.getElementById('symbol-search').value = '';
      });
    });
  } catch (e) { console.error(e); }
}

// ─── INDICATOR MODAL ───
function showIndicatorModal() {
  const allInds = [
    { cat: 'Trend', items: ['ema_9','ema_21','ema_50','ema_200','sma_20','vwap','supertrend'] },
    { cat: 'Momentum', items: ['rsi','macd_line','stoch_k','adx'] },
    { cat: 'Volatility', items: ['bb_upper','bb_mid','bb_lower','atr'] },
    { cat: 'Volume', items: ['volume_sma','volume_ratio','obv'] },
  ];
  const html = allInds.map(g => {
    const items = g.items.map(name => {
      const checked = activeIndicators.has(name) ? 'checked' : '';
      return `<div class="indicator-item"><label><input type="checkbox" data-ind="${name}" ${checked}/> ${name.toUpperCase().replace('_', ' ')}</label></div>`;
    }).join('');
    return `<div class="indicator-group"><h4>${g.cat}</h4>${items}</div>`;
  }).join('');
  document.getElementById('indicator-list').innerHTML = html;
  document.getElementById('indicator-modal').classList.remove('hidden');
  document.querySelectorAll('#indicator-list input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', () => {
      if (cb.checked) activeIndicators.add(cb.dataset.ind);
      else activeIndicators.delete(cb.dataset.ind);
      loadSymbol(currentSymbol, currentMarket);
    });
  });
}

function getIndicatorInfo(name) {
  const overlays = ['ema_9','ema_21','ema_50','ema_200','sma_20','vwap','supertrend','bb_upper','bb_mid','bb_lower'];
  return { overlay: overlays.includes(name) };
}

// ─── BACKTEST ───
async function runBacktest() {
  const btn = document.getElementById('run-backtest-btn');
  btn.disabled = true; btn.textContent = '⏳ Running...';
  const strategy = document.getElementById('bt-strategy').value;
  const capital = document.getElementById('bt-capital').value;
  const sl = document.getElementById('bt-sl').value;
  const tp = document.getElementById('bt-tp').value;
  try {
    const url = `${API}/api/backtest/run?symbol=${currentSymbol}&market=${currentMarket}&strategy=${strategy}&interval=${currentInterval}&initial_capital=${capital}&stop_loss=${sl}&take_profit=${tp}`;
    const res = await fetch(url, { method: 'POST' });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    renderBacktestResults(data);
  } catch (e) { console.error(e); }
  finally { btn.disabled = false; btn.textContent = '▶ Run Backtest'; }
}

function renderBacktestResults(data) {
  const container = document.getElementById('backtest-results');
  container.classList.remove('hidden');
  const m = data.metrics || {};
  const metricsHtml = [
    ['Total Trades', m.total_trades, ''],
    ['Win Rate', m.win_rate + '%', m.win_rate >= 50 ? 'positive' : 'negative'],
    ['Total P&L', '₹' + (m.total_pnl || 0).toLocaleString(), m.total_pnl >= 0 ? 'positive' : 'negative'],
    ['Return', m.return_pct + '%', m.return_pct >= 0 ? 'positive' : 'negative'],
    ['Max Drawdown', m.max_drawdown_pct + '%', 'negative'],
    ['Sharpe Ratio', m.sharpe_ratio, m.sharpe_ratio >= 1 ? 'positive' : 'negative'],
    ['Risk/Reward', m.risk_reward_ratio, m.risk_reward_ratio >= 1.5 ? 'positive' : 'negative'],
    ['Avg Win', '₹' + (m.avg_win || 0).toLocaleString(), 'positive'],
    ['Avg Loss', '₹' + (m.avg_loss || 0).toLocaleString(), 'negative'],
  ].map(([label, value, cls]) => `<div class="bt-metric-row"><span class="label">${label}</span><span class="value ${cls}">${value}</span></div>`).join('');
  document.getElementById('bt-metrics').innerHTML = metricsHtml;
  const trades = (data.trades || []).slice(-20);
  const tradesHtml = trades.map(t => {
    const cls = t.pnl >= 0 ? 'win' : 'loss';
    return `<div class="bt-trade-row"><span class="bt-trade-dir ${cls}">${t.pnl >= 0 ? 'WIN' : 'LOSS'}</span><span>${t.entry_price} → ${t.exit_price}</span><span class="value ${cls}">₹${t.pnl}</span></div>`;
  }).join('');
  document.getElementById('bt-trades').innerHTML = `<h4>Recent Trades (${trades.length})</h4>${tradesHtml}`;
}

// ─── SCANNER ───
let scannerWs = null;
function connectScannerWS() {
  try {
    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    scannerWs = new WebSocket(`${wsProto}//${location.host}/ws/scanner`);
    scannerWs.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'scan_complete') renderScanResults(data);
    };
    scannerWs.onclose = () => setTimeout(connectScannerWS, 5000);
    scannerWs.onerror = () => {};
  } catch(e) {}
}

async function triggerScan() {
  const btn = document.getElementById('run-scan-btn');
  btn.textContent = '⏳...';
  try {
    const res = await fetch(`${API}/api/scanner/run`, { method: 'POST' });
    const data = await res.json();
    if (data.summary) renderScanSummary(data.summary);
  } catch (e) { console.error(e); }
  btn.textContent = 'Scan Now';
}

function renderScanResults(data) {
  // Update signal list
  const signalList = document.getElementById('signal-list');
  const signals = data.top_signals || [];
  document.getElementById('signal-count').textContent = signals.length;
  if (!signals.length) {
    signalList.innerHTML = '<div class="empty-state"><p>No signals detected</p><p class="sub">Market may be flat</p></div>';
    return;
  }
  signalList.innerHTML = signals.slice(0, 15).map(s => renderSignalCard(s, s.symbol)).join('');
  signalList.querySelectorAll('.signal-card').forEach((card, i) => {
    card.addEventListener('click', () => {
      const s = signals[i];
      loadSymbol(s.symbol, s.market);
    });
  });
  // Scanner tab
  const scannerList = document.getElementById('scanner-results');
  const results = data.results || [];
  scannerList.innerHTML = results.slice(0, 20).map(r => {
    const dir = r.verdict?.includes('BUY') ? 'buy' : r.verdict?.includes('SELL') ? 'sell' : '';
    return `<div class="signal-card ${dir}" data-sym="${r.symbol}" data-mkt="${r.market}">
      <div class="signal-top"><span class="signal-symbol">${r.symbol}</span><span class="signal-badge ${dir}">${r.verdict || 'NEUTRAL'}</span></div>
      <div class="signal-reason">${r.signal_count} signals | Buy: ${r.buy_signals} Sell: ${r.sell_signals}</div>
    </div>`;
  }).join('');
  scannerList.querySelectorAll('.signal-card').forEach(card => {
    card.addEventListener('click', () => loadSymbol(card.dataset.sym, card.dataset.mkt));
  });
}

function renderScanSummary(summary) {
  const data = { type: 'scan_complete', top_signals: summary.top_buys?.concat(summary.top_sells || []) || [], results: [] };
  renderScanResults(data);
}

// ─── LIVE CRYPTO STREAMING ───
function getIntervalMs(interval) {
  const map = { '1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000, '30m': 1800000, '1h': 3600000, '2h': 7200000, '4h': 14400000, '1d': 86400000, '1w': 604800000 };
  return map[interval] || 60000;
}

function connectLiveStream(symbol) {
  disconnectLiveStream();
  const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProto}//${location.host}/ws/crypto/${symbol}`;
  try {
    liveWs = new WebSocket(wsUrl);
    liveWs.onopen = () => {
      document.getElementById('live-badge').classList.remove('hidden');
      startAutoRefresh();
    };
    liveWs.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.error) return;
        const price = data.price;
        const tradeTime = Math.floor(new Date(data.timestamp.replace(' ', 'T')).getTime() / 1000);
        const intervalMs = getIntervalMs(currentInterval);
        const candleStart = Math.floor(tradeTime / (intervalMs / 1000)) * (intervalMs / 1000);

        // Update price display with flash
        const priceEl = document.getElementById('current-price');
        const oldPrice = lastLivePrice;
        lastLivePrice = price;
        priceEl.textContent = formatPrice(price);
        if (oldPrice && price !== oldPrice) {
          priceEl.classList.remove('price-flash-up', 'price-flash-down');
          void priceEl.offsetWidth;
          priceEl.classList.add(price > oldPrice ? 'price-flash-up' : 'price-flash-down');
        }

        // Update or create live candle
        if (!liveCandle || candleStart !== liveCandleStart) {
          liveCandle = { time: candleStart, open: price, high: price, low: price, close: price };
          liveCandleStart = candleStart;
        } else {
          liveCandle.close = price;
          liveCandle.high = Math.max(liveCandle.high, price);
          liveCandle.low = Math.min(liveCandle.low, price);
        }
        updateRequested = true;
      } catch (err) { /* ignore parse errors */ }
    };
    startUpdateLoop();
    liveWs.onclose = () => {
      document.getElementById('live-badge').classList.add('hidden');
      // Reconnect after 3s if still same symbol
      setTimeout(() => {
        if (currentSymbol === symbol && currentMarket === 'crypto') {
          connectLiveStream(symbol);
        }
      }, 3000);
    };
    liveWs.onerror = () => {};
  } catch (e) {
    console.error('WebSocket connect failed:', e);
  }
}

function disconnectLiveStream() {
  if (liveWs) {
    liveWs.onclose = null; // prevent reconnect
    liveWs.close();
    liveWs = null;
  }
  if (throttleTimer) { clearInterval(throttleTimer); throttleTimer = null; }
  liveCandle = null;
  liveCandleStart = 0;
  lastLivePrice = 0;
  updateRequested = false;
  document.getElementById('live-badge').classList.add('hidden');
  if (autoRefreshTimer) { clearInterval(autoRefreshTimer); autoRefreshTimer = null; }
}

function startUpdateLoop() {
  if (throttleTimer) clearInterval(throttleTimer);
  throttleTimer = setInterval(() => {
    if (updateRequested && liveCandle) {
      candleSeries.update(liveCandle);
      updateRequested = false;
    }
  }, 100); // 10Hz update rate is smooth but lightweight
}

function startAutoRefresh() {
  if (autoRefreshTimer) clearInterval(autoRefreshTimer);
  autoRefreshTimer = setInterval(() => {
    if (currentMarket === 'crypto' && currentSymbol) {
      loadSymbol(currentSymbol, currentMarket);
    }
  }, 30000);
}

// ─── MULTI-TIMEFRAME ───
async function fetchMultiTimeframe(symbol) {
  const strip = document.getElementById('multi-tf-strip');
  const grid = document.getElementById('mtf-grid');
  try {
    const res = await fetch(`${API}/api/crypto/${symbol}/multi-tf?timeframes=1m,5m,15m,1h,4h,1d`);
    const data = await res.json();
    if (!data.timeframes) { strip.classList.add('hidden'); return; }
    strip.classList.remove('hidden');
    const tfs = ['1m', '5m', '15m', '1h', '4h', '1d'];
    grid.innerHTML = tfs.map(tf => {
      const info = data.timeframes[tf] || { verdict: 'NO DATA' };
      const verdict = info.verdict || 'NEUTRAL';
      const cls = verdict.includes('BUY') ? (verdict.includes('STRONG') ? 'strong-buy' : 'buy') :
                  verdict.includes('SELL') ? (verdict.includes('STRONG') ? 'strong-sell' : 'sell') :
                  verdict === 'NO DATA' ? 'nodata' : 'neutral';
      const isActive = tf === currentInterval ? 'active-tf' : '';
      const label = verdict.replace('STRONG ', 'S.');
      return `<div class="mtf-cell ${isActive}" data-tf="${tf}"><span class="mtf-tf">${tf}</span><span class="mtf-verdict ${cls}">${label}</span></div>`;
    }).join('');
    // Click MTF cells to switch timeframe
    grid.querySelectorAll('.mtf-cell').forEach(cell => {
      cell.addEventListener('click', () => {
        const tf = cell.dataset.tf;
        document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
        const tfBtn = document.querySelector(`.tf-btn[data-tf="${tf}"]`);
        if (tfBtn) tfBtn.classList.add('active');
        currentInterval = tf;
        loadSymbol(currentSymbol, currentMarket);
      });
    });
  } catch (e) {
    strip.classList.add('hidden');
  }
}
