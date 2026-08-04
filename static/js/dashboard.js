// ===== State =====
let currentPair = "USD/INR";
let currentPeriod = "6mo";
let currentInterval = "1d";
let mainChart = null;
let candleSeries = null;
let overlaySeries = {};
let rsiChart = null;
let rsiSeries = null;
let macdChart = null;
let macdLineSeries = null;
let macdSignalSeries = null;
let macdHistSeries = null;

// Interval -> allowed periods + default period
const INTERVAL_PERIODS = {
    "1m":  { periods: ["1d", "5d"],                   default: "5d"  },
    "5m":  { periods: ["1d", "5d", "1mo"],             default: "5d"  },
    "15m": { periods: ["1d", "5d", "1mo"],             default: "1mo" },
    "30m": { periods: ["5d", "1mo", "3mo"],            default: "1mo" },
    "1h":  { periods: ["1mo", "3mo", "6mo"],           default: "3mo" },
    "1d":  { periods: ["1mo", "3mo", "6mo", "1y"],     default: "6mo" },
};

const PERIOD_LABELS = {
    "1d": "1D", "5d": "5D", "1mo": "1M", "3mo": "3M", "6mo": "6M", "1y": "1Y",
};

function isIntraday(interval) {
    return interval !== "1d" && interval !== "1wk" && interval !== "1mo";
}

function getChartOptions() {
    return {
        layout: { background: { color: "#1a1a2e" }, textColor: "#90a4ae" },
        grid: { vertLines: { color: "#1e2d3d" }, horzLines: { color: "#1e2d3d" } },
        crosshair: { mode: 0 },
        rightPriceScale: { borderColor: "#2a2a4e" },
        timeScale: { borderColor: "#2a2a4e", timeVisible: isIntraday(currentInterval) },
    };
}

// ===== Initialization =====
document.addEventListener("DOMContentLoaded", () => {
    loadPairList();
    setupEventListeners();
    updatePeriodButtons(currentInterval);
    fetchAnalysis(currentPair, currentPeriod, currentInterval);
});

// ===== Load pair list from API =====
async function loadPairList() {
    try {
        const resp = await fetch("/api/pairs");
        const groups = await resp.json();
        const container = document.getElementById("pair-list");
        container.innerHTML = "";

        for (const [group, pairs] of Object.entries(groups)) {
            const label = document.createElement("div");
            label.className = "pair-group-label";
            label.textContent = group + " Pairs";
            container.appendChild(label);

            for (const pairName of Object.keys(pairs)) {
                const btn = document.createElement("button");
                btn.className = "pair-btn" + (pairName === currentPair ? " active" : "");
                btn.textContent = pairName;
                btn.addEventListener("click", () => selectPair(pairName));
                container.appendChild(btn);
            }
        }
    } catch (e) {
        console.error("Failed to load pairs:", e);
    }
}

// ===== Update period buttons based on selected interval =====
function updatePeriodButtons(interval) {
    const config = INTERVAL_PERIODS[interval] || INTERVAL_PERIODS["1d"];
    const container = document.getElementById("period-buttons");
    container.innerHTML = "";

    const intraday = isIntraday(interval);

    // For intraday, force the default period; otherwise keep current if valid
    if (intraday) {
        currentPeriod = config.default;
    } else if (!config.periods.includes(currentPeriod)) {
        currentPeriod = config.default;
    }

    for (const p of config.periods) {
        const btn = document.createElement("button");
        btn.dataset.period = p;
        btn.textContent = PERIOD_LABELS[p] || p;
        if (p === currentPeriod) btn.classList.add("active");
        if (intraday) {
            btn.disabled = true;
        } else {
            btn.addEventListener("click", () => {
                container.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                currentPeriod = p;
                fetchAnalysis(currentPair, currentPeriod, currentInterval);
            });
        }
        container.appendChild(btn);
    }
}

// ===== Event listeners =====
function setupEventListeners() {
    // Interval buttons
    document.querySelectorAll("#interval-buttons button").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll("#interval-buttons button").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            currentInterval = btn.dataset.interval;
            updatePeriodButtons(currentInterval);
            fetchAnalysis(currentPair, currentPeriod, currentInterval);
        });
    });

    // Overlay toggles
    document.getElementById("ovl-sma").addEventListener("change", updateOverlays);
    document.getElementById("ovl-ema").addEventListener("change", updateOverlays);
    document.getElementById("ovl-ema-fast").addEventListener("change", updateOverlays);
    document.getElementById("ovl-vwap").addEventListener("change", updateOverlays);
    document.getElementById("ovl-bb").addEventListener("change", updateOverlays);

    // Sidebar toggle (mobile)
    const toggle = document.getElementById("sidebar-toggle");
    if (toggle) {
        toggle.addEventListener("click", () => {
            document.getElementById("sidebar").classList.toggle("open");
        });
    }

    // Window resize
    window.addEventListener("resize", () => {
        if (mainChart) mainChart.applyOptions({ width: document.getElementById("main-chart").clientWidth });
        if (rsiChart) rsiChart.applyOptions({ width: document.getElementById("rsi-chart").clientWidth });
        if (macdChart) macdChart.applyOptions({ width: document.getElementById("macd-chart").clientWidth });
    });
}

// ===== Select pair =====
function selectPair(pair) {
    currentPair = pair;
    document.querySelectorAll(".pair-btn").forEach((b) => {
        b.classList.toggle("active", b.textContent === pair);
    });
    // Close sidebar on mobile
    document.getElementById("sidebar").classList.remove("open");
    fetchAnalysis(pair, currentPeriod, currentInterval);
}

// ===== Fetch analysis from API =====
async function fetchAnalysis(pair, period, interval) {
    const loading = document.getElementById("loading-overlay");
    const errorDiv = document.getElementById("error-message");
    loading.classList.add("visible");
    errorDiv.style.display = "none";

    const urlPair = pair.replace("/", "-");
    try {
        const resp = await fetch(`/api/analyze/${urlPair}?period=${period}&interval=${interval}`);
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || "Failed to fetch data");
        }
        const data = await resp.json();
        renderAll(data);
    } catch (e) {
        console.error(e);
        document.getElementById("error-text").textContent = e.message;
        errorDiv.style.display = "block";
    } finally {
        loading.classList.remove("visible");
    }
}

// ===== Render everything =====
function renderAll(data) {
    renderPriceHeader(data);
    renderCandlestickChart(data);
    renderRSIChart(data);
    renderMACDChart(data);
    renderOverallGauge(data.overall);
    renderVerdict(data.overall, data.mtf);
    renderSignalTable(data.signals);
}

// ===== Price header =====
function renderPriceHeader(data) {
    document.getElementById("pair-name").textContent = data.pair;
    document.getElementById("last-price").textContent = data.last_price;
    const changeEl = document.getElementById("price-change");
    const sign = data.change >= 0 ? "+" : "";
    changeEl.textContent = `${sign}${data.change} (${sign}${data.change_pct}%)`;
    changeEl.className = "price-change " + (data.change >= 0 ? "up" : "down");

    const badge = document.getElementById("overall-signal-badge");
    badge.textContent = data.overall.signal;
    badge.className = "signal-badge " + signalToClass(data.overall.signal);
}

// ===== Candlestick chart =====
function renderCandlestickChart(data) {
    const container = document.getElementById("main-chart");
    container.innerHTML = "";

    mainChart = LightweightCharts.createChart(container, {
        ...getChartOptions(),
        width: container.clientWidth,
        height: 400,
    });

    candleSeries = mainChart.addCandlestickSeries({
        upColor: "#4caf50",
        downColor: "#ef5350",
        borderUpColor: "#4caf50",
        borderDownColor: "#ef5350",
        wickUpColor: "#4caf50",
        wickDownColor: "#ef5350",
    });
    candleSeries.setData(data.chart);

    // Store overlay data for toggling
    window._overlayData = data.overlays;
    overlaySeries = {};
    applyOverlays();

    mainChart.timeScale().fitContent();
}

// ===== Apply overlays based on checkbox state =====
function applyOverlays() {
    if (!mainChart || !window._overlayData) return;

    // Remove existing overlay series
    for (const key of Object.keys(overlaySeries)) {
        try { mainChart.removeSeries(overlaySeries[key]); } catch(e) {}
    }
    overlaySeries = {};

    const od = window._overlayData;
    const showSma = document.getElementById("ovl-sma").checked;
    const showEma = document.getElementById("ovl-ema").checked;
    const showEmaFast = document.getElementById("ovl-ema-fast").checked;
    const showVwap = document.getElementById("ovl-vwap").checked;
    const showBb = document.getElementById("ovl-bb").checked;

    if (showSma) {
        if (od.sma20 && od.sma20.length) {
            overlaySeries.sma20 = mainChart.addLineSeries({ color: "#ffb74d", lineWidth: 1, title: "SMA20" });
            overlaySeries.sma20.setData(od.sma20);
        }
        if (od.sma50 && od.sma50.length) {
            overlaySeries.sma50 = mainChart.addLineSeries({ color: "#4fc3f7", lineWidth: 1, title: "SMA50" });
            overlaySeries.sma50.setData(od.sma50);
        }
        if (od.sma200 && od.sma200.length) {
            overlaySeries.sma200 = mainChart.addLineSeries({ color: "#ce93d8", lineWidth: 1, title: "SMA200" });
            overlaySeries.sma200.setData(od.sma200);
        }
    }

    if (showEma) {
        if (od.ema12 && od.ema12.length) {
            overlaySeries.ema12 = mainChart.addLineSeries({ color: "#66bb6a", lineWidth: 1, title: "EMA12", lineStyle: 2 });
            overlaySeries.ema12.setData(od.ema12);
        }
        if (od.ema26 && od.ema26.length) {
            overlaySeries.ema26 = mainChart.addLineSeries({ color: "#ef5350", lineWidth: 1, title: "EMA26", lineStyle: 2 });
            overlaySeries.ema26.setData(od.ema26);
        }
    }

    if (showEmaFast) {
        if (od.ema9 && od.ema9.length) {
            overlaySeries.ema9 = mainChart.addLineSeries({ color: "#00e676", lineWidth: 1, title: "EMA9", lineStyle: 2 });
            overlaySeries.ema9.setData(od.ema9);
        }
        if (od.ema21 && od.ema21.length) {
            overlaySeries.ema21 = mainChart.addLineSeries({ color: "#ff9100", lineWidth: 1, title: "EMA21", lineStyle: 2 });
            overlaySeries.ema21.setData(od.ema21);
        }
    }

    if (showVwap) {
        if (od.vwap && od.vwap.length) {
            overlaySeries.vwap = mainChart.addLineSeries({ color: "#00e5ff", lineWidth: 2, title: "VWAP" });
            overlaySeries.vwap.setData(od.vwap);
        }
    }

    if (showBb) {
        if (od.bb_upper && od.bb_upper.length) {
            overlaySeries.bb_upper = mainChart.addLineSeries({ color: "#4db6ac", lineWidth: 1, title: "BB Upper", lineStyle: 1 });
            overlaySeries.bb_upper.setData(od.bb_upper);
        }
        if (od.bb_middle && od.bb_middle.length) {
            overlaySeries.bb_middle = mainChart.addLineSeries({ color: "#4db6ac", lineWidth: 1, title: "BB Mid", lineStyle: 2 });
            overlaySeries.bb_middle.setData(od.bb_middle);
        }
        if (od.bb_lower && od.bb_lower.length) {
            overlaySeries.bb_lower = mainChart.addLineSeries({ color: "#4db6ac", lineWidth: 1, title: "BB Lower", lineStyle: 1 });
            overlaySeries.bb_lower.setData(od.bb_lower);
        }
    }
}

function updateOverlays() {
    applyOverlays();
}

// ===== RSI chart =====
function renderRSIChart(data) {
    const container = document.getElementById("rsi-chart");
    container.innerHTML = "";

    rsiChart = LightweightCharts.createChart(container, {
        ...getChartOptions(),
        width: container.clientWidth,
        height: 160,
    });

    rsiSeries = rsiChart.addLineSeries({ color: "#ce93d8", lineWidth: 1.5 });
    rsiSeries.setData(data.rsi);

    // Overbought / oversold lines
    if (data.rsi.length > 0) {
        const firstTime = data.rsi[0].time;
        const lastTime = data.rsi[data.rsi.length - 1].time;

        const ob = rsiChart.addLineSeries({ color: "#ef535066", lineWidth: 1, lineStyle: 2 });
        ob.setData([{ time: firstTime, value: 70 }, { time: lastTime, value: 70 }]);

        const os = rsiChart.addLineSeries({ color: "#4caf5066", lineWidth: 1, lineStyle: 2 });
        os.setData([{ time: firstTime, value: 30 }, { time: lastTime, value: 30 }]);
    }

    rsiChart.timeScale().fitContent();
}

// ===== MACD chart =====
function renderMACDChart(data) {
    const container = document.getElementById("macd-chart");
    container.innerHTML = "";

    macdChart = LightweightCharts.createChart(container, {
        ...getChartOptions(),
        width: container.clientWidth,
        height: 160,
    });

    macdLineSeries = macdChart.addLineSeries({ color: "#4fc3f7", lineWidth: 1.5, title: "MACD" });
    macdLineSeries.setData(data.macd);

    macdSignalSeries = macdChart.addLineSeries({ color: "#ffb74d", lineWidth: 1, title: "Signal" });
    macdSignalSeries.setData(data.macd_signal);

    // Histogram as a separate series
    if (data.macd_hist && data.macd_hist.length) {
        macdHistSeries = macdChart.addHistogramSeries({
            color: "#4caf50",
            priceFormat: { type: "price", precision: 5, minMove: 0.00001 },
        });
        const histData = data.macd_hist.map((d) => ({
            time: d.time,
            value: d.value,
            color: d.value >= 0 ? "#4caf5088" : "#ef535088",
        }));
        macdHistSeries.setData(histData);
    }

    macdChart.timeScale().fitContent();
}

// ===== Signal gauge =====
function renderOverallGauge(overall) {
    // Map score (-2..+2) to position (0%..100%)
    const pct = Math.max(0, Math.min(100, ((overall.score + 2) / 4) * 100));
    document.getElementById("gauge-needle").style.left = pct + "%";
    document.getElementById("gauge-score").textContent =
        `${overall.signal} (Score: ${overall.score}, Strength: ${overall.strength}%)`;
    document.getElementById("gauge-score").className = overall.score > 0 ? "score-positive" : overall.score < 0 ? "score-negative" : "score-neutral";
}

// ===== Verdict panel =====
function renderVerdict(overall, mtf) {
    const arrowEl = document.getElementById("verdict-arrow");
    const summaryEl = document.getElementById("verdict-summary");
    const confLabel = document.getElementById("confidence-label");
    const confFill = document.getElementById("confidence-bar-fill");
    const votesEl = document.getElementById("verdict-votes");

    const dir = overall.direction || "NEUTRAL";
    const dirLower = dir.toLowerCase();

    // Arrow
    if (dir === "UP") {
        arrowEl.textContent = "\u25B2";
    } else if (dir === "DOWN") {
        arrowEl.textContent = "\u25BC";
    } else {
        arrowEl.textContent = "\u2014";
    }
    arrowEl.className = dirLower;

    // Summary
    summaryEl.textContent = overall.summary || "";

    // Confidence bar
    const pct = overall.agreement_pct || 0;
    confLabel.textContent = `Agreement: ${pct}%`;
    confFill.style.width = pct + "%";
    confFill.className = dirLower;

    // Vote chips
    votesEl.innerHTML =
        `<span class="vote-chip up">\u25B2 Up: ${overall.up_count || 0}</span>` +
        `<span class="vote-chip down">\u25BC Down: ${overall.down_count || 0}</span>` +
        `<span class="vote-chip neutral-vote">\u2014 Neutral: ${overall.neutral_count || 0}</span>`;

    // Meta chips — confluence, regime, volatility
    const confluenceEl = document.getElementById("meta-confluence");
    const regimeEl = document.getElementById("meta-regime");
    const volatilityEl = document.getElementById("meta-volatility");

    if (confluenceEl) {
        const cl = overall.confluence_level || "Unknown";
        confluenceEl.textContent = cl + " Confluence";
        confluenceEl.className = "meta-chip meta-" + cl.toLowerCase().replace(/\s+/g, "-");
    }
    if (regimeEl) {
        const tr = overall.trend_regime || "Unknown";
        regimeEl.textContent = tr + " Market";
        regimeEl.className = "meta-chip meta-" + tr.toLowerCase();
    }
    if (volatilityEl) {
        const vol = overall.volatility || "Unknown";
        const atr = overall.atr_pct != null ? ` (ATR ${overall.atr_pct}%)` : "";
        volatilityEl.textContent = vol + " Volatility" + atr;
        volatilityEl.className = "meta-chip meta-vol-" + vol.toLowerCase().replace(/\s+/g, "-");
    }

    // Whipsaw chip (conditional)
    const whipsawEl = document.getElementById("meta-whipsaw");
    if (whipsawEl) {
        if (overall.is_whipsaw) {
            whipsawEl.textContent = "Whipsaw (" + overall.whipsaw_flips + " flips)";
            whipsawEl.className = "meta-chip meta-whipsaw-active";
            whipsawEl.style.display = "";
        } else {
            whipsawEl.style.display = "none";
        }
    }

    // S/R chip (conditional)
    const srEl = document.getElementById("meta-sr");
    if (srEl) {
        if (overall.sr_context) {
            srEl.textContent = overall.sr_context;
            if (overall.sr_warning && overall.sr_warning.indexOf("reinforced") !== -1) {
                srEl.className = "meta-chip meta-sr-reinforced";
            } else {
                srEl.className = "meta-chip meta-sr-caution";
            }
            srEl.title = overall.sr_warning || "";
            srEl.style.display = "";
        } else {
            srEl.style.display = "none";
        }
    }

    // MTF chip — use htf_label from API response (Daily/Weekly)
    const mtfEl = document.getElementById("meta-mtf");
    if (mtfEl && mtf) {
        const htfLabel = mtf.htf_label || (isIntraday(currentInterval) ? "Daily" : "Weekly");
        mtfEl.textContent = htfLabel + ": " + mtf.weekly_trend;
        mtfEl.className = "meta-chip meta-mtf-" + mtf.weekly_trend.toLowerCase();
        mtfEl.title = mtf.warning || "";
        mtfEl.style.display = "";
    } else if (mtfEl) {
        mtfEl.style.display = "none";
    }

    // Signal quality bar
    const qualityLabel = document.getElementById("quality-label");
    const qualityFill = document.getElementById("quality-bar-fill");
    if (qualityLabel && qualityFill && overall.signal_quality) {
        const sq = overall.signal_quality;
        qualityLabel.textContent = "Signal Quality: " + sq.score + "% (" + sq.label + ")";
        qualityFill.style.width = sq.score + "%";
        qualityFill.className = "quality-" + sq.label.toLowerCase().replace(/\s+/g, "-");
    }
}

// ===== Signal table =====
function renderSignalTable(signals) {
    const tbody = document.getElementById("signal-tbody");
    tbody.innerHTML = "";

    for (const s of signals) {
        const tr = document.createElement("tr");
        const cls = signalToClass(s.signal);
        const scoreCls = s.score > 0 ? "score-positive" : s.score < 0 ? "score-negative" : "score-neutral";
        const bet = s.bet || "--";
        const betCls = bet === "UP" ? "up" : bet === "DOWN" ? "down" : "neutral";
        tr.innerHTML = `
            <td><strong>${s.name}</strong></td>
            <td><span class="bet-cell ${betCls}">${bet === "UP" ? "\u25B2 UP" : bet === "DOWN" ? "\u25BC DOWN" : "--"}</span></td>
            <td>${s.value}</td>
            <td><span class="signal-cell ${cls}">${s.signal}</span></td>
            <td class="${scoreCls}">${s.score > 0 ? "+" : ""}${s.score}</td>
            <td>${s.explanation}</td>
        `;
        tbody.appendChild(tr);
    }
}

// ===== Helpers =====
function signalToClass(signal) {
    const s = signal.toLowerCase().replace(/\s+/g, "-");
    const map = {
        "strong-buy": "strong-buy",
        "buy": "buy",
        "slightly-bullish": "buy",
        "neutral": "neutral",
        "slightly-bearish": "sell",
        "sell": "sell",
        "strong-sell": "strong-sell",
        "no-data": "neutral",
    };
    return map[s] || "neutral";
}
