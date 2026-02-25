import { fetchJson } from "/js/api.js";
import { renderMeteogram as drawMeteogram, seriesHasMissingValues, SERIES_TYPES } from "/js/meteogram.js?v=20260225h";

const state = {
  metadata: null,
  datasetId: null,
  typeId: null,
  variableId: null,
  init: null,
  lead: 0,
  leadHours: [],
  initToLeads: {},
  expectedInitToLeads: {},
  isAnimating: false,
  pinnedPoint: null,
  seriesData: null,
  seriesDiagnostics: null,
  seriesRequestId: 0,
};
const METADATA_REFRESH_MS = 20_000;
const METADATA_REFRESH_WHILE_RUNNING_MS = 3_000;

const els = {
  dataset: document.getElementById("datasetSelect"),
  type: document.getElementById("typeSelect"),
  catalogInfo: document.getElementById("catalogInfo"),
  variable: document.getElementById("variableSelect"),
  init: document.getElementById("initSelect"),
  lead: document.getElementById("leadRange"),
  leadText: document.getElementById("leadText"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  speedSelect: document.getElementById("speedSelect"),
  validTimeText: document.getElementById("validTimeText"),
  legendRamp: document.getElementById("legendRamp"),
  legendLabels: document.getElementById("legendLabels"),
  tileFetchStatus: document.getElementById("tileFetchStatus"),
  meteogramBlock: document.getElementById("meteogramBlock"),
  meteogramPoint: document.getElementById("meteogramPoint"),
  meteogramCanvas: document.getElementById("meteogramCanvas"),
  fieldDebugInfo: document.getElementById("fieldDebugInfo"),
};

const map = new maplibregl.Map({
  container: "map",
  attributionControl: false,
  style: {
    version: 8,
    sources: {
      swissgray: {
        type: "raster",
        tiles: [
          "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg",
        ],
        tileSize: 256,
      },
    },
    layers: [
      {
        id: "swissgray",
        type: "raster",
        source: "swissgray",
      },
    ],
  },
  center: [8.25, 46.8],
  zoom: 7.2,
  minZoom: 5,
  maxZoom: 12,
});
map.addControl(
  new maplibregl.AttributionControl({
    compact: true,
    customAttribution:
      '<a href="https://maplibre.org/" target="_blank" rel="noopener noreferrer">MapLibre</a> | Map: <a href="https://www.swisstopo.admin.ch/en" target="_blank" rel="noopener noreferrer">Swisstopo</a> | Data: <a href="https://www.meteoswiss.admin.ch/" target="_blank" rel="noopener noreferrer">MeteoSwiss</a>',
  })
);
if (map.keyboard && typeof map.keyboard.disable === "function") {
  map.keyboard.disable();
}

const tooltip = document.createElement("div");
tooltip.className = "tooltip";
tooltip.style.display = "none";
document.body.appendChild(tooltip);

let hoverTimer = null;
let abortController = null;
let weatherSourceLoading = false;
let metadataPollTimer = null;
let metadataPollInFlight = false;
let animationTimer = null;
const prefetchInFlight = new Set();
const prefetchRecentAt = new Map();
const PREFETCH_RECENT_TTL_MS = 20_000;
const HOVER_DEBOUNCE_MS = 180;
const PREFETCH_AHEAD_COUNT = 1;
let tileRequestVersion = 0;
let tileRetryTimer = null;
let tileRetryBackoffMs = 500;
let tileUrlNonce = 0;
let lastHover = null;

async function bootstrap() {
  const initialUrlState = parseUrlState();
  await refreshMetadata({ preserveSelection: false });
  applyUrlState(initialUrlState);
  bindEvents();

  map.on("load", () => {
    addOrReplaceWeatherLayer();
    applyMapUrlState(initialUrlState);
    if (state.pinnedPoint) {
      loadSeriesForPinnedPoint();
    } else {
      renderMeteogram();
    }
  });

  map.on("mousemove", onMapMouseMove);
  map.on("click", onMapClick);
  map.on("mouseleave", () => {
    lastHover = null;
    if (hoverTimer) {
      clearTimeout(hoverTimer);
      hoverTimer = null;
    }
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    tooltip.style.display = "none";
  });
  map.on("dataloading", onMapDataLoading);
  map.on("sourcedata", onMapSourceData);
  map.on("error", onMapError);
  map.on("moveend", () => {
    updateUrlState();
  });

  scheduleMetadataPoll();
}

function scheduleMetadataPoll() {
  if (metadataPollTimer) {
    clearTimeout(metadataPollTimer);
  }
  const interval = anyDatasetRefreshing() ? METADATA_REFRESH_WHILE_RUNNING_MS : METADATA_REFRESH_MS;
  metadataPollTimer = setTimeout(async () => {
    if (!metadataPollInFlight) {
      metadataPollInFlight = true;
      try {
        await refreshMetadata({ preserveSelection: true });
      } catch (err) {
        console.error("Metadata refresh failed:", err);
      } finally {
        metadataPollInFlight = false;
      }
    }
    scheduleMetadataPoll();
  }, interval);
}

function populateControls(metadata) {
  const datasets = metadata.datasets;
  if (!datasets || datasets.length === 0) {
    els.dataset.innerHTML = "";
    els.type.innerHTML = "";
    els.variable.innerHTML = "";
    els.init.innerHTML = "";
    return;
  }
  state.datasetId = datasets[0].dataset_id;

  els.dataset.innerHTML = datasets
    .map((d) => `<option value="${d.dataset_id}">${d.display_name}</option>`)
    .join("");

  const selectedDataset = datasets[0];
  const selectedTypes = selectedDataset.types || [{ type_id: "control", display_name: "Control" }];
  state.typeId = selectedTypes[0].type_id;
  els.type.innerHTML = selectedTypes
    .map((t) => `<option value="${t.type_id}">${t.display_name}</option>`)
    .join("");
  state.variableId = selectedDataset.variables[0].variable_id;
  state.initToLeads = selectedDataset.init_to_leads || {};
  state.expectedInitToLeads = selectedDataset.expected_init_to_leads || {};
  state.leadHours = selectedDataset.lead_hours || [];
  els.variable.innerHTML = selectedDataset.variables
    .map((v) => `<option value="${v.variable_id}">${v.display_name}</option>`)
    .join("");

  const initTimes = selectedDataset.init_times || [];
  state.init = selectDefaultInit(selectedDataset.dataset_id, initTimes, state.initToLeads);
  renderInitOptions(selectedDataset, state.init);
  els.init.value = state.init;

  setLeadChoicesForCurrentInit(true);
  updateAnimationUi();
  renderLegend();
  renderRefreshStatus();
}

async function refreshMetadata({ preserveSelection }) {
  const metadata = await fetchJson("/api/metadata", { timeoutMs: 8000 });
  const previous = {
    datasetId: state.datasetId,
    typeId: state.typeId,
    variableId: state.variableId,
    init: state.init,
    lead: state.lead,
    pinnedPoint: state.pinnedPoint ? { ...state.pinnedPoint } : null,
  };
  state.metadata = metadata;
  populateControls(metadata);

  if (preserveSelection) {
    const datasets = metadata.datasets || [];
    const preservedDataset = datasets.find((d) => d.dataset_id === previous.datasetId);
    if (preservedDataset) {
      state.datasetId = previous.datasetId;
    }

    els.dataset.value = state.datasetId;
    const activeDataset = datasets.find((d) => d.dataset_id === state.datasetId) || datasets[0];
    if (activeDataset) {
      const activeTypes = activeDataset.types || [{ type_id: "control", display_name: "Control" }];
      els.type.innerHTML = activeTypes
        .map((t) => `<option value="${t.type_id}">${t.display_name}</option>`)
        .join("");
      if (activeTypes.some((t) => t.type_id === previous.typeId)) {
        state.typeId = previous.typeId;
      } else {
        state.typeId = activeTypes[0].type_id;
      }
      els.type.value = state.typeId;

      els.variable.innerHTML = activeDataset.variables
        .map((v) => `<option value="${v.variable_id}">${v.display_name}</option>`)
        .join("");
      state.initToLeads = activeDataset.init_to_leads || {};
      state.expectedInitToLeads = activeDataset.expected_init_to_leads || {};
      state.leadHours = activeDataset.lead_hours || [];

      if (activeDataset.variables.some((v) => v.variable_id === previous.variableId)) {
        state.variableId = previous.variableId;
      }
      els.variable.value = state.variableId;

      const initTimes = activeDataset.init_times || [];
      renderInitOptions(activeDataset, previous.init);

      if (initTimes.includes(previous.init)) {
        state.init = previous.init;
      } else {
        state.init = selectDefaultInit(activeDataset.dataset_id, initTimes, state.initToLeads);
      }
      els.init.value = state.init;

      const leads = state.initToLeads[state.init] || [];
      if (leads.includes(previous.lead)) {
        state.lead = previous.lead;
      }
      setLeadChoicesForCurrentInit(false);
      updateAnimationUi();
      renderLegend();
    }
  }

  const previousKey = `${previous.datasetId}|${previous.typeId}|${previous.variableId}|${previous.init}|${previous.lead}`;
  const currentKey = `${state.datasetId}|${state.typeId}|${state.variableId}|${state.init}|${state.lead}`;
  const shouldReloadTiles = !preserveSelection || previousKey !== currentKey;
  if (map.loaded() && shouldReloadTiles) {
    addOrReplaceWeatherLayer();
  }
  if (state.pinnedPoint && (!preserveSelection || previousKey !== currentKey)) {
    loadSeriesForPinnedPoint();
  }
  updateUrlState();
  renderRefreshStatus();
}

function bindEvents() {
  els.dataset.addEventListener("change", () => {
    stopAnimation();
    state.datasetId = els.dataset.value;
    const dataset = state.metadata.datasets.find((d) => d.dataset_id === state.datasetId);
    const datasetTypes = dataset.types || [{ type_id: "control", display_name: "Control" }];
    els.type.innerHTML = datasetTypes
      .map((t) => `<option value="${t.type_id}">${t.display_name}</option>`)
      .join("");
    state.typeId = datasetTypes[0].type_id;
    state.initToLeads = dataset.init_to_leads || {};
    state.expectedInitToLeads = dataset.expected_init_to_leads || {};
    state.leadHours = dataset.lead_hours || [];
    els.variable.innerHTML = dataset.variables
      .map((v) => `<option value="${v.variable_id}">${v.display_name}</option>`)
      .join("");
    state.variableId = dataset.variables[0].variable_id;
    const initTimes = dataset.init_times || [];
    state.init = selectDefaultInit(dataset.dataset_id, initTimes, state.initToLeads);
    renderInitOptions(dataset, state.init);
    els.init.value = state.init;
    setLeadChoicesForCurrentInit(true);
    renderLegend();
    renderRefreshStatus();
    addOrReplaceWeatherLayer();
    if (state.pinnedPoint) {
      loadSeriesForPinnedPoint();
    }
    updateUrlState();
  });

  els.type.addEventListener("change", () => {
    stopAnimation();
    state.typeId = els.type.value;
    addOrReplaceWeatherLayer();
    if (state.pinnedPoint) {
      loadSeriesForPinnedPoint();
    }
    updateUrlState();
  });

  els.variable.addEventListener("change", () => {
    stopAnimation();
    state.variableId = els.variable.value;
    renderLegend();
    addOrReplaceWeatherLayer();
    if (state.pinnedPoint) {
      loadSeriesForPinnedPoint();
    }
    updateUrlState();
  });

  els.init.addEventListener("change", () => {
    stopAnimation();
    state.init = els.init.value;
    setLeadChoicesForCurrentInit(false);
    updateAnimationUi();
    addOrReplaceWeatherLayer();
    if (state.pinnedPoint) {
      loadSeriesForPinnedPoint();
    }
    updateUrlState();
  });

  els.lead.addEventListener("input", () => {
    stopAnimation();
    const leadIndex = Number(els.lead.value);
    state.lead = state.leadHours[leadIndex] ?? 0;
    updateLeadLabel();
    addOrReplaceWeatherLayer();
    prefetchUpcomingLeads(PREFETCH_AHEAD_COUNT);
    renderMeteogram();
    updateUrlState();
  });

  els.playPauseBtn.addEventListener("click", () => {
    if (state.isAnimating) {
      stopAnimation();
    } else {
      startAnimation();
    }
  });

  els.speedSelect.addEventListener("change", () => {
    if (state.isAnimating) {
      restartAnimationTimer();
    }
    updateUrlState();
  });

  document.addEventListener("keydown", onLeadKeydown);
}

function updateLeadLabel() {
  if (!state.init) {
    els.leadText.textContent = "-";
    els.validTimeText.textContent = "-";
    return;
  }
  const displayLead = leadForDisplay(state.lead);
  const validDate = validTimeFromInitAndLead(state.init, displayLead);
  els.leadText.textContent = `+${displayLead} h`;
  els.validTimeText.textContent = formatSwissLocal(validDate);
}

function setLeadChoicesForCurrentInit(resetToFirst) {
  const leadsForInit = state.initToLeads[state.init] || state.leadHours || [];
  state.leadHours = leadsForInit;

  if (leadsForInit.length === 0) {
    stopAnimation();
    state.lead = 0;
    els.lead.min = "0";
    els.lead.max = "0";
    els.lead.value = "0";
    els.lead.disabled = true;
    updateLeadLabel();
    updateAnimationUi();
    return;
  }

  els.lead.disabled = false;
  els.lead.min = "0";
  els.lead.max = String(leadsForInit.length - 1);

  let index = leadsForInit.indexOf(state.lead);
  if (resetToFirst || index < 0) {
    index = 0;
    state.lead = leadsForInit[0];
  }
  els.lead.value = String(index);
  updateLeadLabel();
  updateAnimationUi();
}

function selectDefaultInit(datasetId, initTimes, initToLeads) {
  if (!initTimes || initTimes.length === 0) {
    return null;
  }
  for (const init of initTimes) {
    const leads = initToLeads[init] || [];
    if (isRunComplete(datasetId, init, leads)) {
      return init;
    }
  }
  return initTimes[0];
}

function renderInitOptions(dataset, selectedInit) {
  const initTimes = dataset.init_times || [];
  els.init.innerHTML = initTimes
    .map((init) => {
      const leads = state.initToLeads[init] || [];
      const suffix = isRunComplete(dataset.dataset_id, init, leads) ? "" : " (incomplete)";
      return `<option value="${init}">${formatInit(init)}${suffix}</option>`;
    })
    .join("");
  if (selectedInit) {
    els.init.value = selectedInit;
  }
}

function isRunComplete(datasetId, initStr, availableLeads) {
  const expected = state.expectedInitToLeads[initStr] || [];
  if (expected.length === 0) {
    return true;
  }
  const available = new Set(availableLeads.map((v) => Number(v)));
  for (const lead of expected) {
    if (!available.has(lead)) {
      return false;
    }
  }
  return true;
}

function renderLegend() {
  const dataset = state.metadata.datasets.find((d) => d.dataset_id === state.datasetId);
  const variable = dataset.variables.find((v) => v.variable_id === state.variableId);
  const legend = variable.legend;

  if (!legend || !legend.colors || !legend.thresholds) {
    els.legendRamp.style.background = "#d7dde5";
    els.legendLabels.innerHTML = "";
    return;
  }

  const segments = [];
  const n = legend.colors.length;
  for (let i = 0; i < n; i += 1) {
    const pct0 = (i / n) * 100;
    const pct1 = ((i + 1) / n) * 100;
    const [r, g, b] = legend.colors[i];
    segments.push(`rgb(${r},${g},${b}) ${pct0}% ${pct1}%`);
  }
  els.legendRamp.style.background = `linear-gradient(90deg, ${segments.join(", ")})`;

  const minValue = legend.thresholds[0];
  const maxValue = legend.thresholds[legend.thresholds.length - 1];
  els.legendLabels.innerHTML = `<span>&le; ${formatLegendValue(minValue)} ${variable.unit}</span><span>&ge; ${formatLegendValue(maxValue)} ${variable.unit}</span>`;
}

function renderRefreshStatus() {
  if (!state.metadata || !state.metadata.datasets || state.metadata.datasets.length === 0) {
    els.catalogInfo.textContent = "Catalog status unavailable";
    els.catalogInfo.classList.remove("loading");
    return;
  }
  const dataset = state.metadata.datasets.find((d) => d.dataset_id === state.datasetId) || state.metadata.datasets[0];
  const refresh = dataset.refresh || {};
  if (refresh.refreshing) {
    els.catalogInfo.textContent = "Refreshing catalog...";
    els.catalogInfo.classList.add("loading");
    return;
  }
  els.catalogInfo.classList.remove("loading");
  if (refresh.last_refreshed_at) {
    const dt = new Date(refresh.last_refreshed_at);
    if (!Number.isNaN(dt.getTime())) {
      const timeText = dt.toLocaleTimeString("de-CH", {
        timeZone: "Europe/Zurich",
        hour: "2-digit",
        minute: "2-digit",
      });
      els.catalogInfo.textContent = `Catalog updated ${timeText}`;
      return;
    }
  }
  els.catalogInfo.textContent = "Catalog up to date";
}

function anyDatasetRefreshing() {
  if (!state.metadata || !state.metadata.datasets) {
    return false;
  }
  return state.metadata.datasets.some((d) => d.refresh && d.refresh.refreshing);
}

function weatherTileUrl() {
  const nonce = tileUrlNonce;
  return `/api/tiles/${encodeURIComponent(state.datasetId)}/${encodeURIComponent(
    state.variableId
  )}/${encodeURIComponent(state.init)}/${state.lead}/{z}/{x}/{y}.png?type_id=${encodeURIComponent(
    state.typeId || "control"
  )}&_r=${nonce}`;
}

function addOrReplaceWeatherLayer() {
  const sourceId = "weather-source";
  const layerId = "weather-layer";

  if (!map.loaded()) {
    return;
  }
  if (!state.init || state.leadHours.length === 0) {
    clearTileRetry();
    if (map.getLayer(layerId)) {
      map.removeLayer(layerId);
    }
    if (map.getSource(sourceId)) {
      map.removeSource(sourceId);
    }
    setTileLoadingVisible(false);
    if (els.fieldDebugInfo) {
      els.fieldDebugInfo.textContent = "";
    }
    return;
  }

  tileRequestVersion += 1;
  clearTileRetry();
  setTileLoadingVisible(true);
  loadFieldDebugInfo(tileRequestVersion);
  const source = map.getSource(sourceId);
  if (!source) {
    map.addSource(sourceId, {
      type: "raster",
      tiles: [weatherTileUrl()],
      tileSize: 256,
    });

    map.addLayer({
      id: layerId,
      type: "raster",
      source: sourceId,
      paint: {
        "raster-opacity": 0.78,
        "raster-fade-duration": 180,
      },
    });
    prefetchUpcomingLeads(PREFETCH_AHEAD_COUNT);
    return;
  }

  // Keep previous tiles on screen while new tiles load to avoid flicker.
  if (typeof source.setTiles === "function") {
    source.setTiles([weatherTileUrl()]);
  } else {
    if (map.getLayer(layerId)) {
      map.removeLayer(layerId);
    }
    map.removeSource(sourceId);
    map.addSource(sourceId, {
      type: "raster",
      tiles: [weatherTileUrl()],
      tileSize: 256,
    });
    map.addLayer({
      id: layerId,
      type: "raster",
      source: sourceId,
      paint: {
        "raster-opacity": 0.78,
        "raster-fade-duration": 180,
      },
    });
  }
  prefetchUpcomingLeads(PREFETCH_AHEAD_COUNT);
}

function onMapDataLoading(event) {
  if (event.sourceId === "weather-source") {
    weatherSourceLoading = true;
    setTileLoadingVisible(true);
  }
}

function onMapSourceData(event) {
  if (event.sourceId !== "weather-source") {
    return;
  }
  if (event.isSourceLoaded) {
    weatherSourceLoading = false;
    tileRetryBackoffMs = 500;
    clearTileRetry();
    setTileLoadingVisible(false);
    loadFieldDebugInfo(tileRequestVersion);
    refreshHoverValueIfNeeded();
  } else {
    weatherSourceLoading = true;
    setTileLoadingVisible(true);
  }
}

function refreshHoverValueIfNeeded() {
  if (!lastHover || tooltip.style.display === "none") {
    return;
  }
  requestHoverValue(lastHover.lat, lastHover.lon);
}

async function requestHoverValue(lat, lon) {
  if (abortController) {
    abortController.abort();
  }
  abortController = new AbortController();
  try {
    const url = `/api/value?dataset_id=${encodeURIComponent(state.datasetId)}&type_id=${encodeURIComponent(
      state.typeId || "control"
    )}&variable_id=${encodeURIComponent(state.variableId)}&init=${encodeURIComponent(
      state.init
    )}&lead=${state.lead}&lat=${lat}&lon=${lon}`;
    const data = await fetchJson(url, { signal: abortController.signal });
    const text = `${data.value.toFixed(2)} ${unitForVariable(state.variableId)}`;
    tooltip.textContent = text;
  } catch (err) {
    if (err.name !== "AbortError") {
      tooltip.textContent = err.message && err.message.includes("503") ? "Loading..." : "No value";
    }
  }
}

function onMapError(event) {
  if (!event || event.sourceId !== "weather-source" || !event.error) {
    return;
  }
  const status = Number(event.error.status || event.error.statusCode || event.error?.response?.status);
  if (status === 503) {
    scheduleTileRetry();
  }
}

function setTileLoadingVisible(visible) {
  if (visible) {
    els.tileFetchStatus.textContent = "Fetching tiles...";
    els.tileFetchStatus.classList.add("loading");
  } else {
    els.tileFetchStatus.textContent = "Tiles idle";
    els.tileFetchStatus.classList.remove("loading");
  }
}

function scheduleTileRetry() {
  if (tileRetryTimer) {
    return;
  }
  const requestVersion = tileRequestVersion;
  tileRetryTimer = setTimeout(() => {
    tileRetryTimer = null;
    if (requestVersion !== tileRequestVersion) {
      return;
    }
    const source = map.getSource("weather-source");
    if (!source || typeof source.setTiles !== "function") {
      return;
    }
    tileUrlNonce += 1;
    source.setTiles([weatherTileUrl()]);
    tileRetryBackoffMs = Math.min(2500, Math.round(tileRetryBackoffMs * 1.4));
    scheduleTileRetry();
  }, tileRetryBackoffMs);
}

function clearTileRetry() {
  if (tileRetryTimer) {
    clearTimeout(tileRetryTimer);
    tileRetryTimer = null;
  }
}

async function loadFieldDebugInfo(requestVersion) {
  if (!els.fieldDebugInfo) {
    return;
  }
  const datasetId = state.datasetId;
  const typeId = state.typeId || "control";
  const variableId = state.variableId;
  const init = state.init;
  const lead = state.lead;
  if (!datasetId || !variableId || !init || !Number.isFinite(lead)) {
    els.fieldDebugInfo.textContent = "";
    return;
  }
  try {
    const payload = await fetchJson(
      `/api/field-debug?dataset_id=${encodeURIComponent(datasetId)}&type_id=${encodeURIComponent(
        typeId
      )}&variable_id=${encodeURIComponent(variableId)}&init=${encodeURIComponent(init)}&lead=${lead}`,
      { timeoutMs: 4000 }
    );
    if (requestVersion !== tileRequestVersion) {
      return;
    }
    if (payload?.status === "loading" || !payload?.debug) {
      els.fieldDebugInfo.textContent = "Source: loading...";
      return;
    }
    const files = payload?.debug?.source_files || [];
    if (files.length === 0) {
      els.fieldDebugInfo.textContent = "Source: n/a";
      return;
    }
    const preview = files.slice(0, 2).join(", ");
    const suffix = files.length > 2 ? ` (+${files.length - 2} more)` : "";
    els.fieldDebugInfo.textContent = `Source: ${preview}${suffix}`;
  } catch (_err) {
    if (requestVersion !== tileRequestVersion) {
      return;
    }
    const msg = String(_err?.message || "");
    els.fieldDebugInfo.textContent = msg.includes("503") ? "Source: loading..." : "Source: unavailable";
  }
}

function animationIntervalMs() {
  const raw = Number(els.speedSelect.value);
  return Number.isFinite(raw) && raw > 0 ? raw : 900;
}

function updateAnimationUi() {
  if (!els.playPauseBtn) {
    return;
  }
  els.playPauseBtn.textContent = state.isAnimating ? "Pause" : "Play";
  els.playPauseBtn.disabled = els.lead.disabled || state.leadHours.length === 0;
}

function restartAnimationTimer() {
  if (!state.isAnimating) {
    return;
  }
  if (animationTimer) {
    clearInterval(animationTimer);
  }
  animationTimer = setInterval(stepAnimation, animationIntervalMs());
}

function startAnimation() {
  if (els.lead.disabled || state.leadHours.length === 0) {
    return;
  }
  state.isAnimating = true;
  updateAnimationUi();
  restartAnimationTimer();
  updateUrlState();
}

function stopAnimation() {
  if (animationTimer) {
    clearInterval(animationTimer);
    animationTimer = null;
  }
  if (state.isAnimating) {
    state.isAnimating = false;
    updateAnimationUi();
    updateUrlState();
  }
}

function stepAnimation() {
  if (els.lead.disabled || state.leadHours.length === 0) {
    stopAnimation();
    return;
  }
  const currentIndex = Number(els.lead.value);
  const nextIndex = currentIndex >= state.leadHours.length - 1 ? 0 : currentIndex + 1;
  els.lead.value = String(nextIndex);
  state.lead = state.leadHours[nextIndex];
  updateLeadLabel();
  addOrReplaceWeatherLayer();
  prefetchUpcomingLeads(PREFETCH_AHEAD_COUNT);
  renderMeteogram();
  updateUrlState();
}

function prefetchUpcomingLeads(count) {
  if (!state.datasetId || !state.variableId || !state.init || state.leadHours.length === 0) {
    return;
  }
  const currentIndex = Number(els.lead.value);
  const leadTargets = [];
  for (let k = 1; k <= count; k += 1) {
    const idx = (currentIndex + k) % state.leadHours.length;
    leadTargets.push(state.leadHours[idx]);
  }
  for (const lead of leadTargets) {
    prefetchLead(lead);
  }
}

function prefetchLead(lead) {
  const key = `${state.datasetId}|${state.typeId}|${state.variableId}|${state.init}|${lead}`;
  const now = Date.now();
  const last = prefetchRecentAt.get(key);
  if (prefetchInFlight.has(key)) {
    return;
  }
  if (last && now - last < PREFETCH_RECENT_TTL_MS) {
    return;
  }
  prefetchInFlight.add(key);
  fetchJson(
    `/api/prefetch?dataset_id=${encodeURIComponent(state.datasetId)}&type_id=${encodeURIComponent(
      state.typeId || "control"
    )}&variable_id=${encodeURIComponent(state.variableId)}&init=${encodeURIComponent(
      state.init
    )}&lead=${lead}`,
    { timeoutMs: 12000 }
  )
    .catch(() => {})
    .finally(() => {
      prefetchInFlight.delete(key);
      prefetchRecentAt.set(key, Date.now());
    });
}

function onLeadKeydown(event) {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
    return;
  }
  if (els.lead.disabled || state.leadHours.length === 0) {
    return;
  }

  const target = event.target;
  if (target) {
    const tag = target.tagName;
    if (tag === "TEXTAREA" || tag === "SELECT") {
      return;
    }
    if (tag === "INPUT" && target !== els.lead) {
      return;
    }
  }

  const currentIndex = Number(els.lead.value);
  const delta = event.key === "ArrowRight" ? 1 : -1;
  const nextIndex = Math.max(0, Math.min(state.leadHours.length - 1, currentIndex + delta));
  if (nextIndex === currentIndex) {
    return;
  }

  event.preventDefault();
  stopAnimation();
  els.lead.value = String(nextIndex);
  state.lead = state.leadHours[nextIndex];
  updateLeadLabel();
  addOrReplaceWeatherLayer();
  prefetchUpcomingLeads(PREFETCH_AHEAD_COUNT);
  renderMeteogram();
  updateUrlState();
}

function onMapMouseMove(e) {
  if (!state.init || !state.datasetId || !state.variableId) {
    return;
  }
  tooltip.style.display = "block";
  tooltip.style.left = `${e.originalEvent.pageX}px`;
  tooltip.style.top = `${e.originalEvent.pageY}px`;
  tooltip.textContent = "Loading...";
  lastHover = {
    lat: e.lngLat.lat,
    lon: e.lngLat.lng,
  };

  if (hoverTimer) {
    clearTimeout(hoverTimer);
  }

  hoverTimer = setTimeout(async () => {
    if (weatherSourceLoading) {
      tooltip.textContent = "Loading...";
      return;
    }
    await requestHoverValue(lastHover.lat, lastHover.lon);
  }, HOVER_DEBOUNCE_MS);
}

function onMapClick(e) {
  state.pinnedPoint = {
    lat: Number(e.lngLat.lat),
    lon: Number(e.lngLat.lng),
  };
  if (els.meteogramBlock && typeof els.meteogramBlock.open === "boolean") {
    els.meteogramBlock.open = true;
  }
  loadSeriesForPinnedPoint();
  updateUrlState();
}

async function loadSeriesForPinnedPoint() {
  if (!state.pinnedPoint || !state.init) {
    renderMeteogram();
    return;
  }
  const { lat, lon } = state.pinnedPoint;
  const requestId = ++state.seriesRequestId;
  els.meteogramPoint.textContent = `Loading ${lat.toFixed(3)}, ${lon.toFixed(3)}...`;
  const typeParam = SERIES_TYPES.join(",");
  const baseUrl = `/api/series?dataset_id=${encodeURIComponent(
    state.datasetId
  )}&variable_id=${encodeURIComponent(
    state.variableId
  )}&init=${encodeURIComponent(state.init)}&lat=${lat}&lon=${lon}&types=${encodeURIComponent(
    typeParam
  )}`;
  try {
    const cachedData = await fetchJson(`${baseUrl}&cached_only=true`, { timeoutMs: 10000 });
    if (requestId !== state.seriesRequestId) {
      return;
    }
    state.seriesData = cachedData;
    state.seriesDiagnostics = cachedData.diagnostics || null;
    renderMeteogram();

    if (seriesHasMissingValues(cachedData, SERIES_TYPES)) {
      try {
        const fullData = await fetchJson(`${baseUrl}&cached_only=false`, { timeoutMs: 60000 });
        if (requestId !== state.seriesRequestId) {
          return;
        }
        state.seriesData = fullData;
        state.seriesDiagnostics = fullData.diagnostics || null;
        renderMeteogram();
      } catch (_err) {
        // Keep cached partial result if full backfill fails.
        if (requestId === state.seriesRequestId) {
          renderMeteogram();
        }
      }
    }
  } catch (_err) {
    if (requestId !== state.seriesRequestId) {
      return;
    }
    state.seriesData = null;
    state.seriesDiagnostics = null;
    renderMeteogram("Series unavailable");
  }
}

function renderMeteogram(errorText = "") {
  drawMeteogram({
    canvas: els.meteogramCanvas,
    pointEl: els.meteogramPoint,
    pinnedPoint: state.pinnedPoint,
    seriesData: state.seriesData,
    lead: state.lead,
    leadLabelOffsetHours: variableLeadDisplayOffsetHours(),
    errorText,
  });
}

function parseUrlState() {
  const p = new URLSearchParams(window.location.search);
  const parsed = {
    datasetId: p.get("model") || null,
    typeId: p.get("stat") || null,
    variableId: p.get("var") || null,
    init: p.get("run") || null,
    lead: p.get("lead") != null ? Number(p.get("lead")) : null,
    zoom: p.get("z") != null ? Number(p.get("z")) : null,
    centerLat: p.get("lat") != null ? Number(p.get("lat")) : null,
    centerLon: p.get("lon") != null ? Number(p.get("lon")) : null,
    pointLat: p.get("plat") != null ? Number(p.get("plat")) : null,
    pointLon: p.get("plon") != null ? Number(p.get("plon")) : null,
    speed: p.get("speed") || null,
  };
  return parsed;
}

function applyUrlState(urlState) {
  if (!urlState || !state.metadata || !state.metadata.datasets) {
    return;
  }
  const datasets = state.metadata.datasets;
  if (datasets.length === 0) {
    return;
  }

  const ds =
    (urlState.datasetId && datasets.find((d) => d.dataset_id === urlState.datasetId)) || datasets[0];
  state.datasetId = ds.dataset_id;
  els.dataset.value = state.datasetId;

  const types = ds.types || [{ type_id: "control", display_name: "Control" }];
  els.type.innerHTML = types.map((t) => `<option value="${t.type_id}">${t.display_name}</option>`).join("");
  state.typeId =
    (urlState.typeId && types.some((t) => t.type_id === urlState.typeId) && urlState.typeId) ||
    types[0].type_id;
  els.type.value = state.typeId;

  els.variable.innerHTML = ds.variables
    .map((v) => `<option value="${v.variable_id}">${v.display_name}</option>`)
    .join("");
  state.variableId =
    (urlState.variableId &&
      ds.variables.some((v) => v.variable_id === urlState.variableId) &&
      urlState.variableId) ||
    ds.variables[0].variable_id;
  els.variable.value = state.variableId;

  state.initToLeads = ds.init_to_leads || {};
  state.expectedInitToLeads = ds.expected_init_to_leads || {};
  state.leadHours = ds.lead_hours || [];

  const initTimes = ds.init_times || [];
  renderInitOptions(ds, urlState.init);
  if (urlState.init && initTimes.includes(urlState.init)) {
    state.init = urlState.init;
  } else {
    state.init = selectDefaultInit(ds.dataset_id, initTimes, state.initToLeads);
  }
  els.init.value = state.init;

  setLeadChoicesForCurrentInit(false);
  if (urlState.lead != null && Number.isFinite(urlState.lead) && state.leadHours.includes(urlState.lead)) {
    state.lead = urlState.lead;
    els.lead.value = String(state.leadHours.indexOf(state.lead));
    updateLeadLabel();
  }

  if (urlState.speed && Array.from(els.speedSelect.options).some((o) => o.value === urlState.speed)) {
    els.speedSelect.value = urlState.speed;
  }

  if (
    urlState.pointLat != null &&
    urlState.pointLon != null &&
    Number.isFinite(urlState.pointLat) &&
    Number.isFinite(urlState.pointLon)
  ) {
    state.pinnedPoint = {
      lat: urlState.pointLat,
      lon: urlState.pointLon,
    };
  }

  renderLegend();
  renderRefreshStatus();
}

function applyMapUrlState(urlState) {
  if (!urlState || !map.loaded()) {
    return;
  }
  const hasCenter =
    urlState.centerLat != null &&
    urlState.centerLon != null &&
    Number.isFinite(urlState.centerLat) &&
    Number.isFinite(urlState.centerLon);
  const hasZoom = urlState.zoom != null && Number.isFinite(urlState.zoom);
  if (hasCenter || hasZoom) {
    map.jumpTo({
      center: hasCenter ? [urlState.centerLon, urlState.centerLat] : map.getCenter(),
      zoom: hasZoom ? urlState.zoom : map.getZoom(),
    });
  }
}

function updateUrlState() {
  const params = new URLSearchParams();
  if (state.datasetId) params.set("model", state.datasetId);
  if (state.typeId) params.set("stat", state.typeId);
  if (state.variableId) params.set("var", state.variableId);
  if (state.init) params.set("run", state.init);
  if (Number.isFinite(state.lead)) params.set("lead", String(state.lead));
  if (els.speedSelect && els.speedSelect.value) params.set("speed", els.speedSelect.value);

  if (map && typeof map.getCenter === "function") {
    const center = map.getCenter();
    const zoom = map.getZoom();
    if (center && Number.isFinite(center.lat) && Number.isFinite(center.lng)) {
      params.set("lat", center.lat.toFixed(4));
      params.set("lon", center.lng.toFixed(4));
    }
    if (Number.isFinite(zoom)) {
      params.set("z", zoom.toFixed(2));
    }
  }

  if (state.pinnedPoint) {
    params.set("plat", state.pinnedPoint.lat.toFixed(4));
    params.set("plon", state.pinnedPoint.lon.toFixed(4));
  }

  const newUrl = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState(null, "", newUrl);
}

function unitForVariable(variableId) {
  const dataset = state.metadata.datasets.find((d) => d.dataset_id === state.datasetId);
  if (!dataset) {
    return "";
  }
  const variable = dataset.variables.find((v) => v.variable_id === variableId);
  if (!variable) {
    return "";
  }
  return variable.unit;
}

function variableLeadDisplayOffsetHours() {
  const dataset = state.metadata?.datasets?.find((d) => d.dataset_id === state.datasetId);
  const variable = dataset?.variables?.find((v) => v.variable_id === state.variableId);
  const offset = Number(variable?.lead_time_display_offset_hours ?? 0);
  return Number.isFinite(offset) ? offset : 0;
}

function leadForDisplay(leadHours) {
  return Number(leadHours) + variableLeadDisplayOffsetHours();
}

function formatInit(initStr) {
  const y = initStr.slice(0, 4);
  const m = initStr.slice(4, 6);
  const d = initStr.slice(6, 8);
  const h = initStr.slice(8, 10);
  return `${y}-${m}-${d} ${h}:00 UTC`;
}

function validTimeFromInitAndLead(initStr, leadHours) {
  const y = Number(initStr.slice(0, 4));
  const m = Number(initStr.slice(4, 6)) - 1;
  const d = Number(initStr.slice(6, 8));
  const h = Number(initStr.slice(8, 10));
  const dt = new Date(Date.UTC(y, m, d, h, 0, 0));
  dt.setUTCHours(dt.getUTCHours() + leadHours);
  return dt;
}

function formatSwissLocal(date) {
  const tz = "Europe/Zurich";
  const weekday = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    weekday: "short",
  })
    .format(date)
    .replace(",", "");
  const datePart = new Intl.DateTimeFormat("de-CH", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
  const timePart = new Intl.DateTimeFormat("de-CH", {
    timeZone: tz,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
  return `${weekday} ${datePart} ${timePart}`;
}

function formatLegendValue(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  const abs = Math.abs(value);
  let decimals = 0;
  if (abs < 1) {
    decimals = 2;
  } else if (abs < 10) {
    decimals = 1;
  }
  return value
    .toFixed(decimals)
    .replace(/\.0+$/, "")
    .replace(/(\.\d*[1-9])0+$/, "$1");
}

bootstrap().catch((err) => {
  console.error(err);
  if (els.catalogInfo) {
    els.catalogInfo.textContent = `Startup failed: ${err.message}`;
    els.catalogInfo.classList.add("loading");
  }
});
