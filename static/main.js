let fetchJson = null;
let drawMeteogram = null;
let seriesHasMissingValues = null;
let SERIES_TYPES = ["control", "p10", "p90"];
import { formatInit, validTimeFromInitAndLead, formatSwissLocal, formatLegendValue } from "/js/formatting.js";
import { filterLeadsForTimeOperator } from "/js/time_operator.js";
import { selectedOptionText } from "/js/ui_text.js";

const state = {
  metadata: null,
  datasetId: null,
  typeId: null,
  timeOperator: "none",
  variableId: null,
  init: null,
  lead: 0,
  leadHours: [],
  initToLeads: {},
  expectedInitToLeads: {},
  isAnimating: false,
  pinnedPoint: null,
  pinnedLocationName: "",
  pinnedLocationElevationM: null,
  pinnedModelElevationM: null,
  pinnedModelPoint: null,
  seriesData: null,
  seriesDiagnostics: null,
  seriesRequestId: 0,
};
const METADATA_REFRESH_MS = 20_000;
const METADATA_REFRESH_WHILE_RUNNING_MS = 3_000;

const els = {
  dataset: document.getElementById("datasetSelect"),
  type: document.getElementById("typeSelect"),
  timeOperator: document.getElementById("timeOperatorSelect"),
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
  openFullMeteogramBtn: document.getElementById("openFullMeteogramBtn"),
  fieldDebugInfo: document.getElementById("fieldDebugInfo"),
  mapSummaryLine1: document.getElementById("mapSummaryLine1"),
  mapSummaryLine2: document.getElementById("mapSummaryLine2"),
  mapSummaryLine3: document.getElementById("mapSummaryLine3"),
  mapSummaryLine4: document.getElementById("mapSummaryLine4"),
  mapSearchForm: document.getElementById("mapSearchForm"),
  mapSearchInput: document.getElementById("mapSearchInput"),
  mapSearchSuggestions: document.getElementById("mapSearchSuggestions"),
  layerToggleBtn: document.getElementById("layerToggleBtn"),
  layerPanel: document.getElementById("layerPanel"),
  layerBasemap: document.getElementById("layerBasemap"),
  geoLocateBtn: document.getElementById("geoLocateBtn"),
  zoomInBtn: document.getElementById("zoomInBtn"),
  zoomOutBtn: document.getElementById("zoomOutBtn"),
  sidebarToggleBtn: document.getElementById("sidebarToggleBtn"),
  meteogramFlowOverlay: document.getElementById("meteogramFlowOverlay"),
  meteogramFlowTitle: document.getElementById("meteogramFlowTitle"),
  meteogramFlowMessage: document.getElementById("meteogramFlowMessage"),
  meteogramFlowProgressFill: document.getElementById("meteogramFlowProgressFill"),
  meteogramFlowProgressLabel: document.getElementById("meteogramFlowProgressLabel"),
  meteogramFlowDownloadBtn: document.getElementById("meteogramFlowDownloadBtn"),
  meteogramFlowOpenBtn: document.getElementById("meteogramFlowOpenBtn"),
  meteogramFlowCloseBtn: document.getElementById("meteogramFlowCloseBtn"),
};

let map = null;
let fullMeteogramWindow = null;
const FULL_METEOGRAM_VARIABLES = ["clct", "tot_prec", "vmax_10m", "t_2m"];
const fullMeteogramSeriesMemo = new Map();
const FULL_METEOGRAM_MEMO_TTL_MS = 10 * 60 * 1000;
const FULL_METEOGRAM_WARMUP_TYPES = "control,median,p10,p90,min,max";
const FULL_METEOGRAM_WARMUP_POLL_MS = 1000;
const FULL_METEOGRAM_WARMUP_TIMEOUT_MS = 20 * 60 * 1000;
const FULL_METEOGRAM_WARMUP_STATUS_TIMEOUT_MS = 120 * 1000;
const FULL_METEOGRAM_WARMUP_MAX_TRANSIENT_ERRORS = 12;
const MODEL_ELEVATION_MEMO_TTL_MS = 20 * 60 * 1000;
const MODEL_ELEVATION_FAILURE_COOLDOWN_MS = 10 * 60 * 1000;
const modelElevationMemo = new Map();
const modelElevationFailureAt = new Map();

function initMap() {
  if (map) {
    return map;
  }
  if (!window.maplibregl) {
    throw new Error("Map library failed to load. Please reload and check internet access to unpkg.com.");
  }
  map = new maplibregl.Map({
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
    minZoom: 2,
    maxZoom: 12,
  });
  map.addControl(
    new maplibregl.AttributionControl({
      compact: true,
      customAttribution:
        '<a href="https://maplibre.org/" target="_blank" rel="noopener noreferrer">MapLibre</a> | Map: <a href="https://www.swisstopo.admin.ch/en" target="_blank" rel="noopener noreferrer">Swisstopo</a> | Data: <a href="https://www.meteoswiss.admin.ch/" target="_blank" rel="noopener noreferrer">MeteoSwiss</a>',
    })
  );
  map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "top-right");
  if (map.keyboard && typeof map.keyboard.disable === "function") {
    map.keyboard.disable();
  }
  return map;
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
const LOADING_OVERLAY_DELAY_MS = 350;
const LOADING_OVERLAY_DELAY_ANIM_MS = 900;
let loadingOverlayTimer = null;
let loadingOverlayPending = false;
let fieldUnavailable = false;
let fieldDebugProbeTimer = null;
let lastFieldDebugAtMs = 0;
const FIELD_DEBUG_MIN_INTERVAL_MS = 900;
let lastHover = null;
let weatherLayerRetryTimer = null;
let pendingWeatherForceRecreate = false;
let windVectorTimer = null;
let windVectorAbortController = null;
let windVectorRequestVersion = 0;
let windVectorWarmupUntil = 0;
let windVectorInFlight = false;
let windVectorInFlightKey = "";
const layerVisibility = {
  basemap: true,
};
let searchSuggestTimer = null;
let meteogramFlowContext = null;
let meteogramFlowBusy = false;

function sortedVariables(dataset) {
  const vars = Array.isArray(dataset?.variables) ? [...dataset.variables] : [];
  vars.sort((a, b) =>
    String(a?.display_name || "").localeCompare(String(b?.display_name || ""), undefined, { sensitivity: "base" })
  );
  return vars;
}

async function bootstrap() {
  await loadClientModules();
  const initialUrlState = parseUrlState();
  initMap();
  await refreshMetadata({ preserveSelection: false });
  applyUrlState(initialUrlState);
  bindEvents();

  map.on("load", () => {
    addOrReplaceWeatherLayer({ forceRecreateSource: true });
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
    scheduleWindVectorUpdate();
  });
  map.on("idle", () => {
    if (shouldShowWindVectors()) {
      scheduleWindVectorUpdate(0);
    }
  });
  setupAttributionInfoBehavior();
  bindMapOverlayControls();

  scheduleMetadataPoll();
}

async function loadClientModules() {
  if (fetchJson && drawMeteogram && typeof seriesHasMissingValues === "function") {
    return;
  }
  const apiModule = await import("/js/api.js");
  const meteogramModule = await import("/js/meteogram.js?v=20260225i");
  fetchJson = apiModule.fetchJson;
  drawMeteogram = meteogramModule.renderMeteogram;
  seriesHasMissingValues = meteogramModule.seriesHasMissingValues;
  SERIES_TYPES = Array.isArray(meteogramModule.SERIES_TYPES)
    ? meteogramModule.SERIES_TYPES
    : ["control", "p10", "p90"];
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
  const selectedVariables = sortedVariables(selectedDataset);
  const selectedTypes = selectedDataset.types || [{ type_id: "control", display_name: "Control" }];
  state.typeId = selectedTypes[0].type_id;
  els.type.innerHTML = selectedTypes
    .map((t) => `<option value="${t.type_id}">${t.display_name}</option>`)
    .join("");
  state.variableId = selectedVariables[0]?.variable_id || null;
  const timeOperators = metadata.time_operators || selectedDataset.time_operators || [{ time_operator: "none", display_name: "None" }];
  els.timeOperator.innerHTML = timeOperators
    .map((op) => `<option value="${op.time_operator}">${op.display_name}</option>`)
    .join("");
  state.timeOperator = timeOperators[0]?.time_operator || "none";
  els.timeOperator.value = state.timeOperator;
  state.initToLeads = selectedDataset.init_to_leads || {};
  state.expectedInitToLeads = selectedDataset.expected_init_to_leads || {};
  state.leadHours = selectedDataset.lead_hours || [];
  els.variable.innerHTML = selectedVariables
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
  renderMapSummary();
}

async function refreshMetadata({ preserveSelection }) {
  const metadata = await fetchJson("/api/metadata", { timeoutMs: 8000 });
  const previous = {
    datasetId: state.datasetId,
    typeId: state.typeId,
    timeOperator: state.timeOperator,
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
      const activeVariables = sortedVariables(activeDataset);
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

      const timeOperators =
        metadata.time_operators || activeDataset.time_operators || [{ time_operator: "none", display_name: "None" }];
      els.timeOperator.innerHTML = timeOperators
        .map((op) => `<option value="${op.time_operator}">${op.display_name}</option>`)
        .join("");
      if (timeOperators.some((op) => op.time_operator === previous.timeOperator)) {
        state.timeOperator = previous.timeOperator;
      } else {
        state.timeOperator = timeOperators[0]?.time_operator || "none";
      }
      els.timeOperator.value = state.timeOperator;

      els.variable.innerHTML = activeVariables
        .map((v) => `<option value="${v.variable_id}">${v.display_name}</option>`)
        .join("");
      state.initToLeads = activeDataset.init_to_leads || {};
      state.expectedInitToLeads = activeDataset.expected_init_to_leads || {};
      state.leadHours = activeDataset.lead_hours || [];

      if (activeVariables.some((v) => v.variable_id === previous.variableId)) {
        state.variableId = previous.variableId;
      } else if (activeVariables[0]) {
        state.variableId = activeVariables[0].variable_id;
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
      renderMapSummary();
    }
  }

  const previousKey = `${previous.datasetId}|${previous.typeId}|${previous.timeOperator}|${previous.variableId}|${previous.init}|${previous.lead}`;
  const currentKey = `${state.datasetId}|${state.typeId}|${state.timeOperator}|${state.variableId}|${state.init}|${state.lead}`;
  const shouldReloadTiles = !preserveSelection || previousKey !== currentKey;
  if (isMapStyleReady() && shouldReloadTiles) {
    addOrReplaceWeatherLayer({ forceRecreateSource: true });
  }
    if (state.pinnedPoint) {
      state.pinnedModelPoint = nearestModelGridPoint(state.datasetId, state.pinnedPoint.lat, state.pinnedPoint.lon);
      void resolvePinnedPointElevations();
    }
    if (state.pinnedPoint && (!preserveSelection || previousKey !== currentKey)) {
      loadSeriesForPinnedPoint();
    }
    updateUrlState();
    renderRefreshStatus();
    renderMapSummary();
}

function bindEvents() {
  els.dataset.addEventListener("change", () => {
    stopAnimation();
    state.datasetId = els.dataset.value;
    const dataset = state.metadata.datasets.find((d) => d.dataset_id === state.datasetId);
    const datasetVariables = sortedVariables(dataset);
    const datasetTypes = dataset.types || [{ type_id: "control", display_name: "Control" }];
    els.type.innerHTML = datasetTypes
      .map((t) => `<option value="${t.type_id}">${t.display_name}</option>`)
      .join("");
    state.typeId = datasetTypes[0].type_id;
    state.initToLeads = dataset.init_to_leads || {};
    state.expectedInitToLeads = dataset.expected_init_to_leads || {};
    state.leadHours = dataset.lead_hours || [];
    els.variable.innerHTML = datasetVariables
      .map((v) => `<option value="${v.variable_id}">${v.display_name}</option>`)
      .join("");
    state.variableId = datasetVariables[0]?.variable_id || null;
    const timeOperators =
      state.metadata.time_operators || dataset.time_operators || [{ time_operator: "none", display_name: "None" }];
    els.timeOperator.innerHTML = timeOperators
      .map((op) => `<option value="${op.time_operator}">${op.display_name}</option>`)
      .join("");
    state.timeOperator = timeOperators[0]?.time_operator || "none";
    els.timeOperator.value = state.timeOperator;
    const initTimes = dataset.init_times || [];
    state.init = selectDefaultInit(dataset.dataset_id, initTimes, state.initToLeads);
    renderInitOptions(dataset, state.init);
    els.init.value = state.init;
    setLeadChoicesForCurrentInit(true);
    renderLegend();
    renderRefreshStatus();
    addOrReplaceWeatherLayer({ forceRecreateSource: true });
    if (state.pinnedPoint) {
      state.pinnedModelPoint = nearestModelGridPoint(state.datasetId, state.pinnedPoint.lat, state.pinnedPoint.lon);
      void resolvePinnedPointElevations();
      loadSeriesForPinnedPoint();
    }
    updateUrlState();
  });

  els.type.addEventListener("change", () => {
    stopAnimation();
    state.typeId = els.type.value;
    addOrReplaceWeatherLayer({ forceRecreateSource: true });
    if (state.pinnedPoint) {
      loadSeriesForPinnedPoint();
    }
    updateUrlState();
  });

  els.timeOperator.addEventListener("change", () => {
    stopAnimation();
    state.timeOperator = els.timeOperator.value || "none";
    setLeadChoicesForCurrentInit(false);
    addOrReplaceWeatherLayer({ forceRecreateSource: true });
    if (state.pinnedPoint) {
      loadSeriesForPinnedPoint();
    }
    updateUrlState();
  });

  els.variable.addEventListener("change", () => {
    stopAnimation();
    state.variableId = els.variable.value;
    setLeadChoicesForCurrentInit(false);
    renderLegend();
    addOrReplaceWeatherLayer({ forceRecreateSource: true });
    if (shouldShowWindVectors()) {
      windVectorWarmupUntil = Date.now() + 10_000;
      scheduleWindVectorUpdate(0);
    }
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

  if (els.openFullMeteogramBtn) {
    els.openFullMeteogramBtn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      void openFullMeteogramPopup();
    });
  }
  if (els.meteogramFlowCloseBtn) {
    els.meteogramFlowCloseBtn.addEventListener("click", () => {
      hideMeteogramFlowOverlay();
    });
  }
  if (els.meteogramFlowOpenBtn) {
    els.meteogramFlowOpenBtn.addEventListener("click", () => {
      if (!meteogramFlowContext) {
        return;
      }
      hideMeteogramFlowOverlay();
      launchFullMeteogramWindow({ skipWarmup: true });
    });
  }
  if (els.meteogramFlowDownloadBtn) {
    els.meteogramFlowDownloadBtn.addEventListener("click", () => {
      void startMeteogramDownloadFlow();
    });
  }
}

function bindMapOverlayControls() {
  if (els.mapSearchForm) {
    els.mapSearchForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = String(els.mapSearchInput?.value || "").trim();
      if (!query || !map) {
        return;
      }
      try {
        const url =
          `https://api3.geo.admin.ch/rest/services/api/SearchServer?` +
          `searchText=${encodeURIComponent(query)}&type=locations&limit=10&sr=4326`;
        const resp = await fetch(url, { headers: { Accept: "application/json" } });
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const payload = await resp.json();
        const result = firstSwissTopoResult(payload);
        if (!result) {
          throw new Error("No matches");
        }
        const lat = Number(result.lat);
        const lon = Number(result.lon);
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
          throw new Error("Invalid coordinates");
        }
        setPinnedPointFromSelection(lat, lon, { label: result.label || query, easting: result.easting, northing: result.northing });
        map.flyTo({ center: [lon, lat], zoom: Math.max(8, map.getZoom()), essential: true });
        hideSearchSuggestions();
      } catch (_err) {
        if (els.mapSearchInput) {
          els.mapSearchInput.placeholder = "No place found";
          window.setTimeout(() => {
            if (els.mapSearchInput) {
              els.mapSearchInput.placeholder = "Location, coordinates, ...";
            }
          }, 1400);
        }
      }
    });
  }
  if (els.mapSearchInput) {
    els.mapSearchInput.addEventListener("input", () => {
      const query = String(els.mapSearchInput?.value || "").trim();
      if (searchSuggestTimer) {
        clearTimeout(searchSuggestTimer);
      }
      if (query.length < 2) {
        hideSearchSuggestions();
        return;
      }
      searchSuggestTimer = setTimeout(() => {
        loadSwissTopoSuggestions(query);
      }, 220);
    });
    els.mapSearchInput.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        hideSearchSuggestions();
      }
    });
  }

  if (els.layerToggleBtn && els.layerPanel) {
    els.layerToggleBtn.addEventListener("click", () => {
      els.layerPanel.hidden = !els.layerPanel.hidden;
    });
    document.addEventListener("click", (event) => {
      if (!els.layerPanel || !els.layerToggleBtn) {
        return;
      }
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }
      if (!els.layerPanel.hidden && !els.layerPanel.contains(target) && !els.layerToggleBtn.contains(target)) {
        els.layerPanel.hidden = true;
      }
    });
  }
  if (els.layerBasemap) {
    els.layerBasemap.addEventListener("change", () => {
      layerVisibility.basemap = !!els.layerBasemap?.checked;
      applyLayerVisibility();
    });
  }

  if (els.geoLocateBtn) {
    els.geoLocateBtn.addEventListener("click", () => {
      if (!navigator.geolocation || !map) {
        return;
      }
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const lat = Number(pos.coords.latitude);
          const lon = Number(pos.coords.longitude);
          if (Number.isFinite(lat) && Number.isFinite(lon)) {
            map.flyTo({ center: [lon, lat], zoom: Math.max(10, map.getZoom()), essential: true });
          }
        },
        () => {}
      );
    });
  }
  if (els.zoomInBtn) {
    els.zoomInBtn.addEventListener("click", () => map?.zoomIn({ duration: 180 }));
  }
  if (els.zoomOutBtn) {
    els.zoomOutBtn.addEventListener("click", () => map?.zoomOut({ duration: 180 }));
  }
  if (els.sidebarToggleBtn) {
    els.sidebarToggleBtn.addEventListener("click", () => {
      document.body.classList.toggle("sidebar-collapsed");
      window.setTimeout(() => map?.resize(), 120);
    });
  }
}

async function loadSwissTopoSuggestions(query) {
  if (!els.mapSearchSuggestions) {
    return;
  }
  try {
    const url =
      `https://api3.geo.admin.ch/rest/services/api/SearchServer?` +
      `searchText=${encodeURIComponent(query)}&type=locations&limit=8&sr=4326`;
    const resp = await fetch(url, { headers: { Accept: "application/json" } });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const payload = await resp.json();
    const results = Array.isArray(payload?.results) ? payload.results : [];
    if (results.length === 0) {
      hideSearchSuggestions();
      return;
    }
    els.mapSearchSuggestions.innerHTML = "";
    for (const item of results.slice(0, 6)) {
      const attrs = item?.attrs || {};
      const parsed = firstSwissTopoResult({ results: [item] });
      if (!parsed) {
        continue;
      }
      const labelRaw = String(attrs.label || attrs.detail || attrs.origin || "");
      const label = normalizeSwissTopoLabel(labelRaw) || `${parsed.lat.toFixed(4)}, ${parsed.lon.toFixed(4)}`;
      const row = document.createElement("div");
      row.className = "mapSuggestionItem";
      row.textContent = label;
      row.addEventListener("click", () => {
        if (els.mapSearchInput) {
          els.mapSearchInput.value = label;
        }
        setPinnedPointFromSelection(parsed.lat, parsed.lon, { label, easting: parsed.easting, northing: parsed.northing });
        map?.flyTo({ center: [parsed.lon, parsed.lat], zoom: Math.max(9, map.getZoom()), essential: true });
        hideSearchSuggestions();
      });
      els.mapSearchSuggestions.appendChild(row);
    }
    els.mapSearchSuggestions.hidden = els.mapSearchSuggestions.childElementCount === 0;
  } catch (_err) {
    hideSearchSuggestions();
  }
}

function hideSearchSuggestions() {
  if (!els.mapSearchSuggestions) {
    return;
  }
  els.mapSearchSuggestions.hidden = true;
  els.mapSearchSuggestions.innerHTML = "";
}

function firstSwissTopoResult(payload) {
  const results = Array.isArray(payload?.results) ? payload.results : [];
  for (const item of results) {
    const attrs = item?.attrs || {};
    const labelRaw = String(attrs.label || attrs.detail || attrs.origin || "");
    const label = normalizeSwissTopoLabel(labelRaw);
    const easting = Number(attrs.y);
    const northing = Number(attrs.x);
    // SearchServer with sr=4326 commonly returns lat/lon directly.
    const lat = Number(attrs.lat);
    const lon = Number(attrs.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      return {
        lat,
        lon,
        label: label || `${lat.toFixed(4)}, ${lon.toFixed(4)}`,
        easting: Number.isFinite(easting) ? easting : null,
        northing: Number.isFinite(northing) ? northing : null,
      };
    }
    // Fallback: bbox center if direct attrs are unavailable.
    const bbox = String(attrs.geom_st_box2d || "");
    const m = bbox.match(/BOX\(([-0-9.]+)\s+([-0-9.]+),([-0-9.]+)\s+([-0-9.]+)\)/i);
    if (m) {
      const minLon = Number(m[1]);
      const minLat = Number(m[2]);
      const maxLon = Number(m[3]);
      const maxLat = Number(m[4]);
      if ([minLon, minLat, maxLon, maxLat].every(Number.isFinite)) {
        const centerLat = (minLat + maxLat) * 0.5;
        const centerLon = (minLon + maxLon) * 0.5;
        return {
          lat: centerLat,
          lon: centerLon,
          label: label || `${centerLat.toFixed(4)}, ${centerLon.toFixed(4)}`,
          easting: Number.isFinite(easting) ? easting : null,
          northing: Number.isFinite(northing) ? northing : null,
        };
      }
    }
  }
  return null;
}

async function setPinnedPointFromSelection(lat, lon, meta = {}) {
  state.pinnedPoint = { lat: Number(lat), lon: Number(lon) };
  state.pinnedLocationName = String(meta.label || "").trim();
  state.pinnedLocationElevationM = null;
  state.pinnedModelElevationM = null;
  state.pinnedModelPoint = nearestModelGridPoint(state.datasetId, lat, lon);
  if (els.meteogramBlock && typeof els.meteogramBlock.open === "boolean") {
    els.meteogramBlock.open = true;
  }
  updatePinnedPointText("Loading...");
  loadSeriesForPinnedPoint();
  updateUrlState();
  void resolvePinnedPointElevations(meta);
}

function nearestModelGridPoint(datasetId, lat, lon) {
  const ds = state.metadata?.datasets?.find((d) => d.dataset_id === datasetId);
  const width = Number(ds?.target_grid_width);
  const height = Number(ds?.target_grid_height);
  const bounds = state.metadata?.bounds || null;
  if (!Number.isFinite(width) || !Number.isFinite(height) || !bounds) {
    return null;
  }
  const minLon = Number(bounds.min_lon);
  const maxLon = Number(bounds.max_lon);
  const minLat = Number(bounds.min_lat);
  const maxLat = Number(bounds.max_lat);
  if (![minLon, maxLon, minLat, maxLat].every(Number.isFinite)) {
    return null;
  }
  const lonFrac = (Number(lon) - minLon) / Math.max(1e-9, maxLon - minLon);
  const latFrac = (maxLat - Number(lat)) / Math.max(1e-9, maxLat - minLat);
  const x = Math.max(0, Math.min(Math.round(lonFrac * (width - 1)), width - 1));
  const y = Math.max(0, Math.min(Math.round(latFrac * (height - 1)), height - 1));
  const gridLon = minLon + (x / Math.max(1, width - 1)) * (maxLon - minLon);
  const gridLat = maxLat - (y / Math.max(1, height - 1)) * (maxLat - minLat);
  return { lat: gridLat, lon: gridLon };
}

function updatePinnedPointText(progressText = "") {
  if (!els.meteogramPoint) {
    return;
  }
  if (!state.pinnedPoint) {
    els.meteogramPoint.textContent = "Click map to pin a point";
    return;
  }
  const prefix = state.pinnedLocationName
    ? `${state.pinnedLocationName} (${state.pinnedPoint.lat.toFixed(3)}, ${state.pinnedPoint.lon.toFixed(3)})`
    : `${state.pinnedPoint.lat.toFixed(3)}, ${state.pinnedPoint.lon.toFixed(3)}`;
  const actual = Number.isFinite(state.pinnedLocationElevationM) ? `${Math.round(state.pinnedLocationElevationM)} masl` : "n/a masl";
  const model = Number.isFinite(state.pinnedModelElevationM) ? `${Math.round(state.pinnedModelElevationM)} masl` : "n/a masl";
  const elevText = ` (${actual}; ICON ${model})`;
  const progress = progressText ? ` - ${progressText}` : "";
  els.meteogramPoint.textContent = `${prefix}${elevText}${progress}`;
}

async function resolvePinnedPointElevations(meta = {}) {
  if (!state.pinnedPoint) {
    return;
  }
  const point = { ...state.pinnedPoint };
  const modelPoint = state.pinnedModelPoint ? { ...state.pinnedModelPoint } : null;
  const [actual, model] = await Promise.all([
    fetchSwissTopoHeight(point.lat, point.lon, meta.easting, meta.northing),
    modelPoint ? fetchModelElevation(state.datasetId, modelPoint.lat, modelPoint.lon) : Promise.resolve(null),
  ]);
  if (!state.pinnedPoint) {
    return;
  }
  if (Math.abs(state.pinnedPoint.lat - point.lat) > 1e-8 || Math.abs(state.pinnedPoint.lon - point.lon) > 1e-8) {
    return;
  }
  state.pinnedLocationElevationM = Number.isFinite(actual) ? Number(actual) : null;
  state.pinnedModelElevationM = Number.isFinite(model) ? Number(model) : null;
  updatePinnedPointText();
}

function normalizeSwissTopoLabel(labelRaw) {
  const clean = String(labelRaw || "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
  if (!clean) {
    return "";
  }
  return clean.replace(/\s+\(([A-Z]{2})\)\s*$/g, "").trim();
}

async function fetchModelElevation(datasetId, lat, lon) {
  if (!datasetId || !Number.isFinite(lat) || !Number.isFinite(lon)) {
    return null;
  }
  const now = Date.now();
  const key = `${datasetId}|${Math.round(Number(lat) * 1000) / 1000}|${Math.round(Number(lon) * 1000) / 1000}`;
  const memo = modelElevationMemo.get(key);
  if (memo && now - memo.ts <= MODEL_ELEVATION_MEMO_TTL_MS) {
    return memo.value;
  }
  const failedAt = modelElevationFailureAt.get(String(datasetId));
  if (Number.isFinite(failedAt) && now - failedAt <= MODEL_ELEVATION_FAILURE_COOLDOWN_MS) {
    return null;
  }
  try {
    const qs = new URLSearchParams({
      dataset_id: String(datasetId),
      lat: String(lat),
      lon: String(lon),
    });
    const resp = await fetch(`/api/model-elevation?${qs.toString()}`);
    if (!resp.ok) {
      modelElevationFailureAt.set(String(datasetId), now);
      return null;
    }
    const payload = await resp.json();
    const value = Number(payload?.model_elevation_m);
    if (Number.isFinite(value)) {
      modelElevationMemo.set(key, { ts: now, value });
      modelElevationFailureAt.delete(String(datasetId));
      return value;
    }
    return null;
  } catch (_err) {
    modelElevationFailureAt.set(String(datasetId), now);
    return null;
  }
}

async function fetchSwissTopoHeight(lat, lon, knownEasting = null, knownNorthing = null) {
  let easting = Number(knownEasting);
  let northing = Number(knownNorthing);
  if (!Number.isFinite(easting) || !Number.isFinite(northing)) {
    const converted = await wgs84ToLv95(lat, lon);
    if (!converted) {
      return null;
    }
    easting = converted.easting;
    northing = converted.northing;
  }
  const endpoints = [
    `https://api3.geo.admin.ch/rest/services/height?easting=${encodeURIComponent(easting)}&northing=${encodeURIComponent(northing)}&sr=2056`,
    `https://api3.geo.admin.ch/rest/services/height?easting=${encodeURIComponent(easting)}&northing=${encodeURIComponent(northing)}`,
  ];
  for (const url of endpoints) {
    try {
      const resp = await fetch(url, { headers: { Accept: "application/json,text/plain,*/*" } });
      if (!resp.ok) {
        continue;
      }
      const text = (await resp.text()).trim();
      if (!text) {
        continue;
      }
      try {
        const parsed = JSON.parse(text);
        const h = Number(parsed.height ?? parsed.altitude ?? parsed.h);
        if (Number.isFinite(h)) {
          return h;
        }
      } catch (_err) {
        const numeric = Number(text.replace(/[^0-9.+-]/g, ""));
        if (Number.isFinite(numeric)) {
          return numeric;
        }
      }
    } catch (_err) {
      // Best effort only.
    }
  }
  return null;
}

async function wgs84ToLv95(lat, lon) {
  const urls = [
    `https://geodesy.geo.admin.ch/reframe/wgs84tolv95?easting=${encodeURIComponent(lon)}&northing=${encodeURIComponent(lat)}&format=json`,
    `https://geodesy.geo.admin.ch/reframe/wgs84tolv95?easting=${encodeURIComponent(lat)}&northing=${encodeURIComponent(lon)}&format=json`,
  ];
  for (const url of urls) {
    try {
      const resp = await fetch(url, { headers: { Accept: "application/json" } });
      if (!resp.ok) {
        continue;
      }
      const payload = await resp.json();
      const easting = Number(payload.easting);
      const northing = Number(payload.northing);
      if (Number.isFinite(easting) && Number.isFinite(northing)) {
        return { easting, northing };
      }
    } catch (_err) {
      // Best effort only.
    }
  }
  return null;
}

function setupAttributionInfoBehavior() {
  const applyCompact = () => {
    const attrib = document.querySelector(".maplibregl-ctrl-attrib");
    if (!attrib) {
      return false;
    }
    attrib.classList.add("maplibregl-compact");
    attrib.classList.remove("maplibregl-compact-show");
    return true;
  };
  if (applyCompact()) {
    return;
  }
  let retries = 0;
  const timer = setInterval(() => {
    if (applyCompact() || retries >= 20) {
      clearInterval(timer);
    }
    retries += 1;
  }, 150);
}

function applyLayerVisibility() {
  if (!map) {
    return;
  }
  const setVis = (id, visible) => {
    if (!map.getLayer(id)) {
      return;
    }
    map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
  };
  setVis("swissgray", layerVisibility.basemap);
}

function updateLeadLabel() {
  if (!state.init) {
    els.leadText.textContent = "-";
    if (els.validTimeText) {
      els.validTimeText.textContent = "-";
    }
    renderMapSummary();
    return;
  }
  const displayLead = leadForDisplay(state.lead);
  const validDate = validTimeFromInitAndLead(state.init, displayLead);
  els.leadText.textContent = `+${displayLead} h`;
  if (els.validTimeText) {
    els.validTimeText.textContent = formatSwissLocal(validDate);
  }
  renderMapSummary();
}

function setLeadChoicesForCurrentInit(resetToFirst) {
  const rawLeadsForInit = state.initToLeads[state.init] || state.leadHours || [];
  const leadsForInit = filterLeadsForTimeOperator(rawLeadsForInit, state.timeOperator, leadForDisplay);
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

function isMapStyleReady() {
  if (!map) {
    return false;
  }
  if (typeof map.isStyleLoaded === "function") {
    return map.isStyleLoaded();
  }
  return typeof map.loaded === "function" ? map.loaded() : false;
}

function weatherTileUrl() {
  const nonce = tileUrlNonce;
  return `/api/tiles/${encodeURIComponent(state.datasetId)}/${encodeURIComponent(
    state.variableId
  )}/${encodeURIComponent(state.init)}/${state.lead}/{z}/{x}/{y}.png?type_id=${encodeURIComponent(
    state.typeId || "control"
  )}&time_operator=${encodeURIComponent(state.timeOperator || "none")}&_r=${nonce}`;
}

function addOrReplaceWeatherLayer({ forceRecreateSource = false } = {}) {
  const sourceId = "weather-source";
  const layerId = "weather-layer";

  if (!isMapStyleReady()) {
    pendingWeatherForceRecreate = pendingWeatherForceRecreate || forceRecreateSource;
    if (!weatherLayerRetryTimer) {
      weatherLayerRetryTimer = setTimeout(() => {
        weatherLayerRetryTimer = null;
        const pendingForce = pendingWeatherForceRecreate;
        pendingWeatherForceRecreate = false;
        addOrReplaceWeatherLayer({ forceRecreateSource: pendingForce });
      }, 120);
    }
    return;
  }
  if (weatherLayerRetryTimer) {
    clearTimeout(weatherLayerRetryTimer);
    weatherLayerRetryTimer = null;
  }
  pendingWeatherForceRecreate = false;
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
    deactivateWindVectors();
    return;
  }

  tileRequestVersion += 1;
  fieldUnavailable = false;
  if (forceRecreateSource) {
    tileUrlNonce += 1;
  }
  clearTileRetry();
  setTileLoadingVisible(true);
  maybeLoadFieldDebugInfo(tileRequestVersion);
  if (shouldShowWindVectors()) {
    // Avoid showing stale arrows while new type/lead vectors load.
    clearWindVectorLayer();
    windVectorWarmupUntil = Date.now() + 10_000;
  } else {
    deactivateWindVectors();
  }
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
    applyLayerVisibility();
    prefetchUpcomingLeads(PREFETCH_AHEAD_COUNT);
    scheduleWindVectorUpdate(0);
    return;
  }

  // Keep previous tiles on screen while new tiles load to avoid flicker.
  if (!forceRecreateSource && typeof source.setTiles === "function") {
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
    applyLayerVisibility();
  }
  applyLayerVisibility();
  prefetchUpcomingLeads(PREFETCH_AHEAD_COUNT);
  scheduleWindVectorUpdate(0);
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
    fieldUnavailable = false;
    tileRetryBackoffMs = 500;
    clearTileRetry();
    clearFieldDebugProbe();
    setTileLoadingVisible(false);
    maybeLoadFieldDebugInfo(tileRequestVersion);
    refreshHoverValueIfNeeded();
    scheduleWindVectorUpdate();
  } else {
    weatherSourceLoading = true;
    if (!fieldUnavailable) {
      setTileLoadingVisible(true);
    }
  }
}

function shouldShowWindVectors() {
  return state.variableId === "wind_speed_10m";
}

function ensureWindVectorLayer() {
  const sourceId = "wind-vector-source";
  const layerId = "wind-vector-layer";
  if (!map.getSource(sourceId)) {
    map.addSource(sourceId, {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] },
    });
  }
  if (!map.getLayer(layerId)) {
    map.addLayer({
      id: layerId,
      type: "line",
      source: sourceId,
      paint: {
        "line-color": "rgba(15, 23, 42, 0.85)",
        "line-width": ["interpolate", ["linear"], ["get", "speed"], 0, 1.3, 120, 3.2],
        "line-opacity": 0.9,
      },
      layout: {
        "line-cap": "round",
        "line-join": "round",
      },
    });
  }
  // Keep vectors above raster weather tiles across layer/source refreshes.
  if (map.getLayer(layerId)) {
    map.moveLayer(layerId);
  }
}

function clearWindVectorLayer() {
  const source = map.getSource("wind-vector-source");
  if (source && typeof source.setData === "function") {
    source.setData({ type: "FeatureCollection", features: [] });
  }
}

function destroyWindVectorLayer() {
  const layerId = "wind-vector-layer";
  const sourceId = "wind-vector-source";
  if (map.getLayer(layerId)) {
    map.removeLayer(layerId);
  }
  if (map.getSource(sourceId)) {
    map.removeSource(sourceId);
  }
}

function setWindVectorVisibility(visible) {
  const layer = map.getLayer("wind-vector-layer");
  if (!layer) {
    return;
  }
  map.setLayoutProperty("wind-vector-layer", "visibility", visible ? "visible" : "none");
}

function deactivateWindVectors() {
  // Invalidate any in-flight response so stale vectors can never be applied.
  windVectorRequestVersion += 1;
  if (windVectorAbortController) {
    windVectorAbortController.abort();
    windVectorAbortController = null;
  }
  if (windVectorTimer) {
    clearTimeout(windVectorTimer);
    windVectorTimer = null;
  }
  destroyWindVectorLayer();
}

function scheduleWindVectorUpdate(delayMs = 180) {
  if (windVectorTimer) {
    clearTimeout(windVectorTimer);
  }
  windVectorTimer = setTimeout(() => {
    windVectorTimer = null;
    updateWindVectors();
  }, delayMs);
}

function windRequestKey() {
  if (!map || typeof map.getBounds !== "function") {
    return "";
  }
  const b = map.getBounds();
  const r3 = (v) => Number(v).toFixed(3);
  return [
    state.datasetId,
    state.typeId || "control",
    state.timeOperator || "none",
    state.init,
    String(state.lead),
    Number(map.getZoom()).toFixed(2),
    r3(b.getSouth()),
    r3(b.getNorth()),
    r3(b.getWest()),
    r3(b.getEast()),
  ].join("|");
}

function buildWindVectorFeatures(vectors) {
  const features = [];
  for (const vec of vectors) {
    const speed = Number(vec.speed);
    const u = Number(vec.u);
    const v = Number(vec.v);
    if (!Number.isFinite(speed) || !Number.isFinite(u) || !Number.isFinite(v)) {
      continue;
    }
    const p0 = map.project([vec.lon, vec.lat]);
    const lenPx = Math.max(7, Math.min(24, 6 + speed * 0.07));
    const theta = Math.atan2(v, u);
    const dx = Math.cos(theta) * lenPx;
    const dy = -Math.sin(theta) * lenPx;
    const p1 = { x: p0.x + dx, y: p0.y + dy };
    const headLen = Math.max(3, lenPx * 0.35);
    const left = {
      x: p1.x + headLen * Math.cos(theta + Math.PI - 0.45),
      y: p1.y - headLen * Math.sin(theta + Math.PI - 0.45),
    };
    const right = {
      x: p1.x + headLen * Math.cos(theta + Math.PI + 0.45),
      y: p1.y - headLen * Math.sin(theta + Math.PI + 0.45),
    };
    const ll0 = map.unproject([p0.x, p0.y]);
    const ll1 = map.unproject([p1.x, p1.y]);
    const lll = map.unproject([left.x, left.y]);
    const llr = map.unproject([right.x, right.y]);
    features.push({
      type: "Feature",
      properties: { speed },
      geometry: { type: "LineString", coordinates: [[ll0.lng, ll0.lat], [ll1.lng, ll1.lat]] },
    });
    features.push({
      type: "Feature",
      properties: { speed },
      geometry: { type: "LineString", coordinates: [[ll1.lng, ll1.lat], [lll.lng, lll.lat]] },
    });
    features.push({
      type: "Feature",
      properties: { speed },
      geometry: { type: "LineString", coordinates: [[ll1.lng, ll1.lat], [llr.lng, llr.lat]] },
    });
  }
  return features;
}

async function updateWindVectors() {
  if (!map || typeof map.getBounds !== "function") {
    return;
  }
  if (typeof map.isStyleLoaded === "function" && !map.isStyleLoaded()) {
    scheduleWindVectorUpdate(150);
    return;
  }
  if (!shouldShowWindVectors() || !state.datasetId || !state.init || !Number.isFinite(state.lead)) {
    deactivateWindVectors();
    return;
  }
  ensureWindVectorLayer();
  setWindVectorVisibility(true);
  const requestKey = windRequestKey();
  if (windVectorInFlight && windVectorInFlightKey === requestKey) {
    return;
  }

  const bounds = map.getBounds();
  const requestVersion = ++windVectorRequestVersion;
  if (windVectorAbortController) {
    windVectorAbortController.abort();
  }
  windVectorAbortController = new AbortController();
  windVectorInFlight = true;
  windVectorInFlightKey = requestKey;
  const url = `/api/wind-vectors?dataset_id=${encodeURIComponent(state.datasetId)}&type_id=${encodeURIComponent(
    state.typeId || "control"
  )}&time_operator=${encodeURIComponent(state.timeOperator || "none")}&init=${encodeURIComponent(
    state.init
  )}&lead=${state.lead}&min_lat=${bounds.getSouth()}&max_lat=${bounds.getNorth()}&min_lon=${bounds.getWest()}&max_lon=${bounds.getEast()}&zoom=${map.getZoom()}`;
  try {
    const payload = await fetchJson(url, { signal: windVectorAbortController.signal, timeoutMs: 12000 });
    if (requestVersion !== windVectorRequestVersion) {
      return;
    }
    if (payload?.status === "loading") {
      scheduleWindVectorUpdate(500);
      return;
    }
    const source = map.getSource("wind-vector-source");
    if (!source || typeof source.setData !== "function") {
      return;
    }
    const features = buildWindVectorFeatures(payload?.vectors || []);
    source.setData({ type: "FeatureCollection", features });
    if (features.length === 0 && Date.now() < windVectorWarmupUntil && shouldShowWindVectors()) {
      scheduleWindVectorUpdate(500);
    }
  } catch (err) {
    if (err?.name === "AbortError") {
      return;
    }
    clearWindVectorLayer();
    if (shouldShowWindVectors()) {
      // Retry automatically when initial fetch races with tile/data loading.
      scheduleWindVectorUpdate(800);
    }
  } finally {
    if (windVectorInFlightKey === requestKey) {
      windVectorInFlight = false;
      windVectorInFlightKey = "";
    }
  }
}

function refreshHoverValueIfNeeded() {
  if (!lastHover || tooltip.style.display === "none" || state.isAnimating) {
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
    )}&lead=${state.lead}&time_operator=${encodeURIComponent(state.timeOperator || "none")}&lat=${lat}&lon=${lon}`;
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
  if (fieldUnavailable) {
    return;
  }
  const status = Number(event.error.status || event.error.statusCode || event.error?.response?.status);
  if (status === 503) {
    scheduleFieldDebugProbe(tileRequestVersion);
    scheduleTileRetry();
  }
}

function setTileLoadingVisible(visible) {
  if (fieldUnavailable && visible) {
    return;
  }
  if (loadingOverlayTimer) {
    clearTimeout(loadingOverlayTimer);
    loadingOverlayTimer = null;
  }
  if (visible) {
    loadingOverlayPending = true;
    const delayMs = state.isAnimating ? LOADING_OVERLAY_DELAY_ANIM_MS : LOADING_OVERLAY_DELAY_MS;
    loadingOverlayTimer = setTimeout(() => {
      loadingOverlayTimer = null;
      if (!loadingOverlayPending || fieldUnavailable) {
        return;
      }
      els.tileFetchStatus.textContent = "Loading...";
      els.tileFetchStatus.classList.add("loading");
    }, delayMs);
  } else {
    loadingOverlayPending = false;
    els.tileFetchStatus.textContent = "";
    els.tileFetchStatus.classList.remove("loading");
  }
}

function setTileUnavailable(reason = "Tiles unavailable") {
  fieldUnavailable = true;
  weatherSourceLoading = false;
  loadingOverlayPending = false;
  if (loadingOverlayTimer) {
    clearTimeout(loadingOverlayTimer);
    loadingOverlayTimer = null;
  }
  clearTileRetry();
  clearFieldDebugProbe();
  els.tileFetchStatus.textContent = reason;
  els.tileFetchStatus.classList.add("loading");
}

function scheduleTileRetry() {
  if (fieldUnavailable) {
    return;
  }
  if (tileRetryTimer) {
    return;
  }
  const requestVersion = tileRequestVersion;
  tileRetryTimer = setTimeout(() => {
    tileRetryTimer = null;
    if (requestVersion !== tileRequestVersion) {
      return;
    }
    if (fieldUnavailable) {
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

function scheduleFieldDebugProbe(requestVersion) {
  if (state.isAnimating) {
    return;
  }
  if (fieldDebugProbeTimer) {
    return;
  }
  fieldDebugProbeTimer = setTimeout(() => {
    fieldDebugProbeTimer = null;
    loadFieldDebugInfo(requestVersion);
  }, 250);
}

function clearFieldDebugProbe() {
  if (fieldDebugProbeTimer) {
    clearTimeout(fieldDebugProbeTimer);
    fieldDebugProbeTimer = null;
  }
}

function maybeLoadFieldDebugInfo(requestVersion, { force = false } = {}) {
  if (!state.isAnimating || force) {
    loadFieldDebugInfo(requestVersion);
    return;
  }
  const now = Date.now();
  if (now - lastFieldDebugAtMs < FIELD_DEBUG_MIN_INTERVAL_MS) {
    return;
  }
  lastFieldDebugAtMs = now;
  loadFieldDebugInfo(requestVersion);
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
      )}&variable_id=${encodeURIComponent(variableId)}&init=${encodeURIComponent(init)}&lead=${lead}&time_operator=${encodeURIComponent(
        state.timeOperator || "none"
      )}`,
      { timeoutMs: 4000 }
    );
    if (requestVersion !== tileRequestVersion) {
      return;
    }
    if (payload?.status === "loading" || !payload?.debug) {
      if (!fieldUnavailable) {
        els.fieldDebugInfo.textContent = "Source: loading...";
      }
      return;
    }
    if (payload?.status === "error") {
      const msg = String(payload?.debug?.message || "asset unavailable");
      els.fieldDebugInfo.textContent = `Source: unavailable (${msg})`;
      weatherSourceLoading = false;
      clearTileRetry();
      setTileUnavailable("Asset unavailable");
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
    if (msg.includes("503")) {
      els.fieldDebugInfo.textContent = "Source: loading...";
    } else {
      els.fieldDebugInfo.textContent = "Source: unavailable";
      setTileUnavailable("Source unavailable");
    }
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

function renderMapSummary() {
  if (!els.mapSummaryLine1 || !els.mapSummaryLine2 || !els.mapSummaryLine3 || !els.mapSummaryLine4) {
    return;
  }
  const dataset = state.metadata?.datasets?.find((d) => d.dataset_id === state.datasetId);
  const variable = dataset?.variables?.find((v) => v.variable_id === state.variableId);
  const variableName = variable?.display_name || selectedOptionText(els.variable, "-");
  const gribName = String(variable?.grib_name || variable?.variable_id || "-");
  const unitText = String(variable?.display_unit || variable?.unit || variable?.standard_unit || "").trim();
  const modelText = dataset?.display_name || selectedOptionText(els.dataset, "-");
  const forecastText = state.init ? formatInit(state.init).replace(":00 UTC", " UTC") : "-";
  const displayLead = leadForDisplay(state.lead);
  const validDate = state.init ? validTimeFromInitAndLead(state.init, displayLead) : null;
  const validText = validDate ? formatSwissLocal(validDate) : "-";
  const statisticText = selectedOptionText(els.type, "-");
  const timeOperatorText = selectedOptionText(els.timeOperator, "None");

  els.mapSummaryLine1.textContent = unitText
    ? `${variableName} (${gribName}, ${unitText})`
    : `${variableName} (${gribName})`;
  els.mapSummaryLine2.textContent = `${modelText} ${forecastText} +${displayLead}h`;
  els.mapSummaryLine3.textContent = `Valid time: ${validText}`;
  els.mapSummaryLine4.textContent = `${statisticText}, ${timeOperatorText}`;
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
  const key = `${state.datasetId}|${state.typeId}|${state.timeOperator}|${state.variableId}|${state.init}|${lead}`;
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
    )}&lead=${lead}&time_operator=${encodeURIComponent(state.timeOperator || "none")}`,
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
    if (weatherSourceLoading || state.isAnimating) {
      tooltip.textContent = "Loading...";
      return;
    }
    await requestHoverValue(lastHover.lat, lastHover.lon);
  }, HOVER_DEBOUNCE_MS);
}

function onMapClick(e) {
  void setPinnedPointFromSelection(Number(e.lngLat.lat), Number(e.lngLat.lng));
}

async function loadSeriesForPinnedPoint() {
  if (!state.pinnedPoint || !state.init) {
    renderMeteogram();
    return;
  }
  const { lat, lon } = state.pinnedPoint;
  const requestId = ++state.seriesRequestId;
  updatePinnedPointText("Loading...");
  const typeParam = SERIES_TYPES.join(",");
  const baseUrl = `/api/series?dataset_id=${encodeURIComponent(
    state.datasetId
  )}&variable_id=${encodeURIComponent(
    state.variableId
  )}&init=${encodeURIComponent(state.init)}&lat=${lat}&lon=${lon}&types=${encodeURIComponent(
    typeParam
  )}&time_operator=${encodeURIComponent(state.timeOperator || "none")}`;
  try {
    const cachedData = await fetchJson(`${baseUrl}&cached_only=true`, { timeoutMs: 10000 });
    if (requestId !== state.seriesRequestId) {
      return;
    }
    state.seriesData = cachedData;
    state.seriesDiagnostics = cachedData.diagnostics || null;
    updatePinnedPointText();
    renderMeteogram();

    if (seriesHasMissingValues(cachedData, SERIES_TYPES)) {
      try {
        const fullData = await fetchJson(`${baseUrl}&cached_only=false`, { timeoutMs: 60000 });
        if (requestId !== state.seriesRequestId) {
          return;
        }
        state.seriesData = fullData;
        state.seriesDiagnostics = fullData.diagnostics || null;
        updatePinnedPointText();
        renderMeteogram();
      } catch (_err) {
        // Keep cached partial result if full backfill fails.
        if (requestId === state.seriesRequestId) {
          updatePinnedPointText();
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
    updatePinnedPointText();
    renderMeteogram("Series unavailable");
  }
}

function renderMeteogram(errorText = "") {
  drawMeteogram({
    canvas: els.meteogramCanvas,
    pointEl: els.meteogramPoint,
    pinnedPoint: state.pinnedPoint,
    pointText: els.meteogramPoint ? els.meteogramPoint.textContent : "",
    seriesData: state.seriesData,
    lead: state.lead,
    leadLabelOffsetHours: variableLeadDisplayOffsetHours(),
    errorText,
  });
}

function currentDatasetMeta() {
  return state.metadata?.datasets?.find((d) => d.dataset_id === state.datasetId) || null;
}

function variableMeta(variableId) {
  const ds = currentDatasetMeta();
  return ds?.variables?.find((v) => v.variable_id === variableId) || null;
}

function initIsoLabel(initStr) {
  if (!initStr || initStr.length < 10) {
    return "-";
  }
  return `${initStr.slice(0, 4)}-${initStr.slice(4, 6)}-${initStr.slice(6, 8)} ${initStr.slice(8, 10)} UTC`;
}

function fullMeteogramPanelsForCurrentDataset() {
  const ds = currentDatasetMeta();
  if (!ds) {
    return [];
  }
  const available = new Set((ds.variables || []).map((v) => String(v.variable_id)));
  return FULL_METEOGRAM_VARIABLES.filter((id) => available.has(id));
}

function showMeteogramFlowOverlay() {
  if (!els.meteogramFlowOverlay) {
    return;
  }
  els.meteogramFlowOverlay.hidden = false;
}

function hideMeteogramFlowOverlay() {
  meteogramFlowBusy = false;
  if (els.meteogramFlowOverlay) {
    els.meteogramFlowOverlay.hidden = true;
  }
}

function setMeteogramFlowState({
  title = "Meteogram readiness",
  message = "",
  percent = null,
  downloadEnabled = false,
  openEnabled = false,
  closeEnabled = true,
} = {}) {
  if (els.meteogramFlowTitle) {
    els.meteogramFlowTitle.textContent = String(title || "Meteogram readiness");
  }
  if (els.meteogramFlowMessage) {
    els.meteogramFlowMessage.textContent = String(message || "");
  }
  if (els.meteogramFlowProgressFill) {
    if (Number.isFinite(percent)) {
      const p = Math.max(0, Math.min(100, Number(percent)));
      els.meteogramFlowProgressFill.style.width = `${p}%`;
    } else {
      els.meteogramFlowProgressFill.style.width = "10%";
    }
  }
  if (els.meteogramFlowProgressLabel) {
    els.meteogramFlowProgressLabel.textContent = Number.isFinite(percent)
      ? `${Math.round(Number(percent))}%`
      : "checking...";
  }
  if (els.meteogramFlowDownloadBtn) {
    els.meteogramFlowDownloadBtn.disabled = !downloadEnabled || meteogramFlowBusy;
  }
  if (els.meteogramFlowOpenBtn) {
    els.meteogramFlowOpenBtn.disabled = !openEnabled || meteogramFlowBusy;
  }
  if (els.meteogramFlowCloseBtn) {
    els.meteogramFlowCloseBtn.disabled = !closeEnabled || meteogramFlowBusy;
  }
}

function analyzeCachedSeriesAvailability(series) {
  const leads = Array.isArray(series?.lead_hours) ? series.lead_hours : [];
  const values = series?.values || {};
  const control = Array.isArray(values.control) ? values.control : [];
  const controlAny = control.some((v) => Number.isFinite(Number(v)));
  const controlComplete = leads.length > 0 && control.length === leads.length && control.every((v) => Number.isFinite(Number(v)));
  const ensembleTypes = ["median", "p10", "p90", "min", "max"];
  let ensembleComplete = true;
  for (const typeId of ensembleTypes) {
    const arr = Array.isArray(values[typeId]) ? values[typeId] : [];
    const complete = leads.length > 0 && arr.length === leads.length && arr.every((v) => Number.isFinite(Number(v)));
    if (!complete) {
      ensembleComplete = false;
      break;
    }
  }
  return {
    controlAny,
    controlComplete,
    ensembleComplete,
  };
}

async function evaluateCachedMeteogramAvailability(point, panels) {
  const panelData = panels.map((variableId) => ({ variableId, series: null, availability: null }));
  for (let i = 0; i < panels.length; i += 1) {
    const variableId = panels[i];
    const progressPct = Math.round((i / Math.max(1, panels.length)) * 100);
    setMeteogramFlowState({
      title: "Meteogram readiness",
      message: `Checking cached data (${i + 1}/${panels.length})...`,
      percent: progressPct,
      downloadEnabled: false,
      openEnabled: false,
      closeEnabled: true,
    });
    const series = await fetchFullMeteogramSeries(variableId, point.lat, point.lon, { mode: "cached_full_only" });
    const availability = analyzeCachedSeriesAvailability(series);
    panelData[i].series = series;
    panelData[i].availability = availability;
  }
  const ctrlAny = panelData.some((p) => p.availability?.controlAny);
  const ctrlComplete = panelData.length > 0 && panelData.every((p) => p.availability?.controlComplete);
  const ensembleComplete = panelData.length > 0 && panelData.every((p) => p.availability?.ensembleComplete);
  return {
    panelData,
    ctrlAny,
    ctrlComplete,
    ensembleComplete,
  };
}

async function runMeteogramWarmup(datasetId, init, panels, onProgress) {
  const varsParam = encodeURIComponent(panels.join(","));
  const typesParam = encodeURIComponent(FULL_METEOGRAM_WARMUP_TYPES);
  let status = await fetchJson(
    `/api/meteogram-warmup/start?dataset_id=${encodeURIComponent(datasetId)}&init=${encodeURIComponent(
      init
    )}&variables=${varsParam}&types=${typesParam}&time_operator=none`,
    { timeoutMs: 30000 }
  );
  let transientErrors = 0;
  const startedAt = Date.now();
  while (status && (status.status === "queued" || status.status === "running")) {
    if (typeof onProgress === "function") {
      onProgress(status, transientErrors);
    }
    if (Date.now() - startedAt > FULL_METEOGRAM_WARMUP_TIMEOUT_MS) {
      throw new Error("Timed out while warming meteogram cache");
    }
    await new Promise((resolve) => setTimeout(resolve, FULL_METEOGRAM_WARMUP_POLL_MS));
    try {
      status = await fetchJson(`/api/meteogram-warmup/status?job_id=${encodeURIComponent(status.job_id)}`, {
        timeoutMs: FULL_METEOGRAM_WARMUP_STATUS_TIMEOUT_MS,
      });
      transientErrors = 0;
    } catch (err) {
      transientErrors += 1;
      if (typeof onProgress === "function") {
        onProgress(null, transientErrors);
      }
      if (transientErrors >= FULL_METEOGRAM_WARMUP_MAX_TRANSIENT_ERRORS) {
        throw err;
      }
    }
  }
  if (typeof onProgress === "function") {
    onProgress(status, transientErrors);
  }
  return status;
}

function launchFullMeteogramWindow(options = {}) {
  fullMeteogramWindow = window.open("", "icon_full_meteogram", "width=1200,height=1700,resizable=yes,scrollbars=yes");
  if (!fullMeteogramWindow) {
    return;
  }
  renderFullMeteogramLoading(fullMeteogramWindow);
  loadAndRenderFullMeteogram(fullMeteogramWindow, options).catch((err) => {
    if (fullMeteogramWindow && !fullMeteogramWindow.closed) {
      fullMeteogramWindow.document.body.innerHTML = `<pre style="padding:16px;color:#a11;font:13px monospace;">${String(
        err?.message || err
      )}</pre>`;
    }
  });
}

async function openFullMeteogramPopup() {
  if (!state.pinnedPoint || !state.datasetId || !state.init || !fetchJson) {
    if (els.meteogramPoint) {
      els.meteogramPoint.textContent = "Click map to pin a point first";
    }
    return;
  }
  const ds = currentDatasetMeta();
  const panels = fullMeteogramPanelsForCurrentDataset();
  if (!ds || panels.length === 0) {
    if (els.meteogramPoint) {
      els.meteogramPoint.textContent = "No meteogram variables available";
    }
    return;
  }
  meteogramFlowContext = {
    datasetId: ds.dataset_id,
    init: state.init,
    point: { ...state.pinnedPoint },
    panels: [...panels],
  };
  showMeteogramFlowOverlay();
  meteogramFlowBusy = true;
  setMeteogramFlowState({
    title: "Meteogram readiness",
    message: "Checking cached data...",
    percent: null,
    downloadEnabled: false,
    openEnabled: false,
    closeEnabled: false,
  });
  try {
    const availability = await evaluateCachedMeteogramAvailability(meteogramFlowContext.point, meteogramFlowContext.panels);
    meteogramFlowContext.availability = availability;
    meteogramFlowBusy = false;
    if (!availability.ctrlAny) {
      setMeteogramFlowState({
        title: "Meteogram data missing",
        message: "CTRL and ensemble data are missing. Download is required before meteogram display.",
        percent: 0,
        downloadEnabled: true,
        openEnabled: false,
        closeEnabled: true,
      });
      return;
    }
    if (availability.ctrlComplete && availability.ensembleComplete) {
      setMeteogramFlowState({
        title: "Meteogram ready",
        message: "All data available. Opening full meteogram...",
        percent: 100,
        downloadEnabled: false,
        openEnabled: false,
        closeEnabled: false,
      });
      hideMeteogramFlowOverlay();
      launchFullMeteogramWindow({ skipWarmup: true });
      return;
    }
    setMeteogramFlowState({
      title: "Partial data available",
      message:
        "CTRL is available. Meteogram can be opened now with available ensemble data, or download missing ensemble data first.",
      percent: availability.ensembleComplete ? 100 : 55,
      downloadEnabled: true,
      openEnabled: true,
      closeEnabled: true,
    });
  } catch (err) {
    meteogramFlowBusy = false;
    setMeteogramFlowState({
      title: "Meteogram check failed",
      message: String(err?.message || err || "Unable to check meteogram readiness"),
      percent: 0,
      downloadEnabled: true,
      openEnabled: false,
      closeEnabled: true,
    });
  }
}

function renderFullMeteogramLoading(win) {
  win.document.open();
  win.document.write(
    "<!doctype html><html><head><meta charset='utf-8'><title>Full Meteogram</title>" +
      "<style>" +
      "body{margin:0;background:#e9e9e9;font-family:IBM Plex Sans,Segoe UI,sans-serif;position:relative;}" +
      "#loadOverlay{position:fixed;inset:0;display:flex;align-items:flex-start;justify-content:center;padding-top:72px;pointer-events:none;z-index:3;}" +
      "#loadCard{min-width:360px;max-width:520px;background:rgba(255,255,255,0.92);border:1px solid #cfd5dc;border-radius:12px;padding:12px 14px;box-shadow:0 4px 18px rgba(0,0,0,.1);}" +
      "#loadTop{display:flex;align-items:center;gap:8px;color:#5f6974;font-size:12px;}" +
      "#spin{width:11px;height:11px;border:2px solid #9aa3ad;border-right-color:transparent;border-radius:50%;animation:spin .8s linear infinite;flex:0 0 auto;}" +
      "#loadDetail{margin-top:5px;color:#6f7882;font-size:11px;}" +
      "#loadTrack{margin-top:8px;height:7px;background:#d9dfe6;border-radius:999px;overflow:hidden;}" +
      "#loadFill{height:100%;width:0%;background:#5c8bd8;transition:width .25s ease;}" +
      "#loadPct{margin-top:4px;color:#6f7882;font-size:11px;text-align:right;}" +
      "@keyframes spin{to{transform:rotate(360deg)}}" +
      "canvas{display:block;margin:0 auto;max-width:100%;height:auto;background:#e9e9e9;}" +
      "</style></head><body>" +
      "<div id='loadOverlay'><div id='loadCard'>" +
      "<div id='loadTop'><span id='spin'></span><span id='loadTxt'>Preparing meteogram...</span></div>" +
      "<div id='loadDetail'>Checking cache status...</div>" +
      "<div id='loadTrack'><div id='loadFill'></div></div>" +
      "<div id='loadPct'>0%</div>" +
      "</div></div>" +
      "<canvas id='fullMeteogramCanvas' width='920' height='980'></canvas>" +
      "</body></html>"
  );
  win.document.close();
}

function setFullMeteogramLoadingVisible(win, visible) {
  const overlay = win?.document?.getElementById("loadOverlay");
  if (!overlay) {
    return;
  }
  overlay.style.display = visible ? "flex" : "none";
}

function setFullMeteogramLoadingState(win, options = {}) {
  const doc = win?.document;
  if (!doc) {
    return;
  }
  const txt = doc.getElementById("loadTxt");
  const detail = doc.getElementById("loadDetail");
  const fill = doc.getElementById("loadFill");
  const pct = doc.getElementById("loadPct");
  const spin = doc.getElementById("spin");
  const title = String(options.title || "Loading...");
  const detailText = String(options.detail || "");
  const percent = Number.isFinite(options.percent) ? Math.max(0, Math.min(100, options.percent)) : null;
  const indeterminate = Boolean(options.indeterminate);
  if (txt) txt.textContent = title;
  if (detail) detail.textContent = detailText;
  if (fill) fill.style.width = percent == null ? "12%" : `${percent}%`;
  if (pct) pct.textContent = percent == null ? "" : `${Math.round(percent)}%`;
  if (spin) spin.style.display = indeterminate ? "inline-block" : "none";
}

async function waitForMeteogramWarmup(win, datasetId, init, panels) {
  setFullMeteogramLoadingState(win, {
    title: "Loading...",
    detail: "Checking cache status...",
    percent: 0,
    indeterminate: true,
  });
  let lastPercent = 0;
  const status = await runMeteogramWarmup(datasetId, init, panels, (payload, transientErrors) => {
    if (!win || win.closed) {
      return;
    }
    if (!payload) {
      setFullMeteogramLoadingState(win, {
        title: "Loading...",
        detail: `Waiting for status update... retry ${transientErrors}/${FULL_METEOGRAM_WARMUP_MAX_TRANSIENT_ERRORS}`,
        percent: lastPercent,
        indeterminate: true,
      });
      return;
    }
    const total = Math.max(0, Number(payload.total_tasks || 0));
    const completed = Math.max(0, Number(payload.completed_tasks || 0));
    const failed = Math.max(0, Number(payload.failed_tasks || 0));
    const percent = Number.isFinite(payload.percent_complete)
      ? Number(payload.percent_complete)
      : total > 0
      ? Math.round((completed / total) * 100)
      : 0;
    const clampedPercent = Math.max(lastPercent, Math.max(0, Math.min(100, percent)));
    lastPercent = clampedPercent;
    const detail = total > 0 ? `${completed}/${total} fields ready${failed > 0 ? ` (${failed} failed)` : ""}` : "Preparing warmup tasks...";
    setFullMeteogramLoadingState(win, {
      title: "Loading...",
      detail,
      percent: clampedPercent,
      indeterminate: false,
    });
  });
  if (status) {
    const total = Math.max(0, Number(status.total_tasks || 0));
    const completed = Math.max(0, Number(status.completed_tasks || 0));
    const failed = Math.max(0, Number(status.failed_tasks || 0));
    const percent = Number.isFinite(status.percent_complete) ? Number(status.percent_complete) : total > 0 ? 100 : 0;
    const detail =
      status.status === "partial"
        ? `${completed}/${total} fields ready (${failed} failed)`
        : total > 0
        ? `${completed}/${total} fields ready`
        : "Cache already warm";
    setFullMeteogramLoadingState(win, {
      title: "Loading...",
      detail,
      percent: Math.max(0, Math.min(100, percent)),
      indeterminate: false,
    });
  }
  return status;
}

async function startMeteogramDownloadFlow() {
  if (!meteogramFlowContext || meteogramFlowBusy) {
    return;
  }
  meteogramFlowBusy = true;
  setMeteogramFlowState({
    title: "Downloading meteogram data",
    message: "Starting download...",
    percent: 0,
    downloadEnabled: false,
    openEnabled: false,
    closeEnabled: false,
  });
  try {
    await runMeteogramWarmup(
      meteogramFlowContext.datasetId,
      meteogramFlowContext.init,
      meteogramFlowContext.panels,
      (payload, transientErrors) => {
        if (!payload) {
          setMeteogramFlowState({
            title: "Downloading meteogram data",
            message: `Waiting for status update... retry ${transientErrors}/${FULL_METEOGRAM_WARMUP_MAX_TRANSIENT_ERRORS}`,
            percent: null,
            downloadEnabled: false,
            openEnabled: false,
            closeEnabled: false,
          });
          return;
        }
        const total = Math.max(0, Number(payload.total_tasks || 0));
        const completed = Math.max(0, Number(payload.completed_tasks || 0));
        const failed = Math.max(0, Number(payload.failed_tasks || 0));
        const percent = Number.isFinite(payload.percent_complete)
          ? Number(payload.percent_complete)
          : total > 0
          ? Math.round((completed / total) * 100)
          : 0;
        const message =
          total > 0
            ? `${completed}/${total} fields ready${failed > 0 ? ` (${failed} failed)` : ""}`
            : "Preparing warmup tasks...";
        setMeteogramFlowState({
          title: "Downloading meteogram data",
          message,
          percent,
          downloadEnabled: false,
          openEnabled: false,
          closeEnabled: false,
        });
      }
    );
    const availability = await evaluateCachedMeteogramAvailability(meteogramFlowContext.point, meteogramFlowContext.panels);
    meteogramFlowContext.availability = availability;
    meteogramFlowBusy = false;
    if (!availability.ctrlAny) {
      setMeteogramFlowState({
        title: "Meteogram data still unavailable",
        message: "CTRL data is still missing after download. Please try again later.",
        percent: 0,
        downloadEnabled: true,
        openEnabled: false,
        closeEnabled: true,
      });
      return;
    }
    if (availability.ctrlComplete && availability.ensembleComplete) {
      setMeteogramFlowState({
        title: "Meteogram ready",
        message: "Download complete. Opening full meteogram...",
        percent: 100,
        downloadEnabled: false,
        openEnabled: false,
        closeEnabled: false,
      });
      hideMeteogramFlowOverlay();
      launchFullMeteogramWindow({ skipWarmup: true });
      return;
    }
    setMeteogramFlowState({
      title: "Partial meteogram ready",
      message: "CTRL is available; some ensemble values are still missing. Opening partial meteogram...",
      percent: 100,
      downloadEnabled: true,
      openEnabled: false,
      closeEnabled: true,
    });
    hideMeteogramFlowOverlay();
    launchFullMeteogramWindow({ skipWarmup: true });
  } catch (err) {
    meteogramFlowBusy = false;
    setMeteogramFlowState({
      title: "Download failed",
      message: String(err?.message || err || "Failed to download meteogram data"),
      percent: 0,
      downloadEnabled: true,
      openEnabled: false,
      closeEnabled: true,
    });
  }
}

async function loadAndRenderFullMeteogram(win, options = {}) {
  const ds = currentDatasetMeta();
  const point = state.pinnedPoint;
  if (!ds || !point) {
    return;
  }
  const available = new Set((ds.variables || []).map((v) => String(v.variable_id)));
  const panels = FULL_METEOGRAM_VARIABLES.filter((id) => available.has(id));
  if (panels.length === 0) {
    throw new Error("No full-meteogram variables available for this dataset");
  }

  const panelData = panels.map((variableId) => ({ variableId, series: null }));
  renderFullMeteogramWindow(win, ds, point, panelData);
  if (!options.skipWarmup) {
    await waitForMeteogramWarmup(win, ds.dataset_id, state.init, panels);
  }

  setFullMeteogramLoadingState(win, {
    title: "Loading...",
    detail: "Rendering meteogram from cache...",
    percent: 100,
    indeterminate: false,
  });
  const fullResults = await Promise.all(
    panels.map((variableId, idx) =>
      fetchFullMeteogramSeries(variableId, point.lat, point.lon, { mode: "cached_full_only" }).then((series) => ({
        idx,
        series,
      }))
    )
  );
  for (const result of fullResults) {
    panelData[result.idx].series = result.series;
  }
  renderFullMeteogramWindow(win, ds, point, panelData);
  setFullMeteogramLoadingVisible(win, false);
}

async function fetchFullMeteogramSeries(variableId, lat, lon, options = {}) {
  void options;
  const base = `/api/series?dataset_id=${encodeURIComponent(state.datasetId)}&variable_id=${encodeURIComponent(
    variableId
  )}&init=${encodeURIComponent(state.init)}&lat=${lat}&lon=${lon}&time_operator=none`;
  const fullTypes = encodeURIComponent(FULL_METEOGRAM_WARMUP_TYPES);
  const wantedTypes = ["control", "median", "p10", "p90", "min", "max"];
  const memoKey = fullMeteogramSeriesMemoKey(variableId, lat, lon);
  const now = Date.now();
  const memo = fullMeteogramSeriesMemo.get(memoKey);
  if (memo && now - memo.ts <= FULL_METEOGRAM_MEMO_TTL_MS && memo.payload && memo.isFull) {
    return memo.payload;
  }
  try {
    const cached = await fetchJson(`${base}&types=${fullTypes}&cached_only=true`, { timeoutMs: 20000 });
    if (cached) {
      const isFull = !seriesHasMissingValues(cached, wantedTypes);
      fullMeteogramSeriesMemo.set(memoKey, {
        ts: now,
        payload: cached,
        fullAttemptFailedAt: isFull ? 0 : now,
        isFull,
      });
    }
    return cached || memo?.payload || null;
  } catch (_err) {
    return memo?.payload || null;
  }
}

function fullMeteogramSeriesMemoKey(variableId, lat, lon) {
  const latBucket = Math.round(Number(lat) * 1000) / 1000;
  const lonBucket = Math.round(Number(lon) * 1000) / 1000;
  return `${state.datasetId}|${state.init}|${variableId}|${latBucket}|${lonBucket}`;
}

function renderFullMeteogramWindow(win, dataset, point, panelData) {
  if (!win || win.closed) {
    return;
  }
  const width = 920;
  const panelH = 258;
  const topPad = 182;
  const botPad = 72;
  const canvasH = topPad + panelData.length * panelH + botPad;
  const canvas = win.document.getElementById("fullMeteogramCanvas");
  if (!canvas) {
    return;
  }
  if (canvas.width !== width) {
    canvas.width = width;
  }
  if (canvas.height !== canvasH) {
    canvas.height = canvasH;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.fillStyle = "#e9e9e9";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const left = 52;
  const right = canvas.width - 40;
  ctx.fillStyle = "#111";
  ctx.font = "600 21px IBM Plex Sans, Segoe UI, sans-serif";
  ctx.fillText(`${dataset.display_name} Meteogram`, left, 48);
  ctx.font = "600 21px IBM Plex Sans, Segoe UI, sans-serif";
  const pointLabel = state.pinnedLocationName
    ? `${state.pinnedLocationName} (${point.lat.toFixed(3)}, ${point.lon.toFixed(3)})`
    : `${point.lat.toFixed(3)}, ${point.lon.toFixed(3)}`;
  const locationHeight = Number.isFinite(state.pinnedLocationElevationM) ? `${Math.round(state.pinnedLocationElevationM)} masl` : "n/a masl";
  const modelHeight = Number.isFinite(state.pinnedModelElevationM) ? `${Math.round(state.pinnedModelElevationM)} masl` : "n/a masl";
  ctx.fillText(`${pointLabel} (${locationHeight}; ICON ${modelHeight})`, left, 69);
  ctx.textAlign = "right";
  ctx.font = "600 21px IBM Plex Sans, Segoe UI, sans-serif";
  ctx.fillText(initIsoLabel(state.init), right, 48);
  ctx.textAlign = "left";

  // Legend
  const legendY = 136;
  const legendWidth = 660;
  const legendStart = Math.round((canvas.width - legendWidth) / 2);
  drawLegendChip(ctx, legendStart + 0, legendY, "#d22", 3, [], "Median");
  drawLegendBand(ctx, legendStart + 156, legendY - 9, "10%-90%");
  drawLegendChip(ctx, legendStart + 362, legendY, "#666", 2, [6, 6], "Min/Max");
  drawLegendChip(ctx, legendStart + 548, legendY, "#1248ff", 3, [], "CTRL");

  for (let i = 0; i < panelData.length; i += 1) {
    const top = topPad + i * panelH + 8;
    drawFullMeteogramPanel(ctx, {
      x0: left + 6,
      y0: top,
      width: right - left - 12,
      height: panelH - 62,
      variableId: panelData[i].variableId,
      variable: variableMeta(panelData[i].variableId),
      payload: panelData[i].series,
      leadLabelOffsetHours: variableLeadDisplayOffsetHours(),
      isBottomPanel: i === panelData.length - 1,
    });
  }
}

function drawLegendChip(ctx, x, y, color, width, dash, label) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 24, y);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = "#111";
  ctx.font = "500 19px IBM Plex Sans, Segoe UI, sans-serif";
  ctx.fillText(label, x + 36, y + 5);
}

function drawLegendBand(ctx, x, y, label) {
  ctx.fillStyle = "rgba(120,120,120,0.36)";
  ctx.fillRect(x, y, 48, 10);
  ctx.fillStyle = "#111";
  ctx.font = "500 19px IBM Plex Sans, Segoe UI, sans-serif";
  ctx.fillText(label, x + 58, y + 10);
}

function drawFullMeteogramPanel(ctx, { x0, y0, width, height, variableId, variable, payload, leadLabelOffsetHours, isBottomPanel }) {
  const titleMap = {
    clct: "Total Cloud Cover (%)",
    tot_prec: "Total Precipitation (mm/1h)",
    vmax_10m: "Max Wind Gust at 10m in Last 1 Hour (km/h)",
    t_2m: "2m Temperature (°C)",
  };
  const label = titleMap[variableId] || (variable ? `${variable.display_name} (${variable.unit || ""})` : variableId);
  ctx.fillStyle = "#111";
  ctx.font = "600 17px IBM Plex Sans, Segoe UI, sans-serif";
  ctx.fillText(label.trim(), x0, y0 - 6);

  ctx.strokeStyle = "#2fd24a";
  ctx.lineWidth = 2.4;
  ctx.strokeRect(x0, y0, width, height);

  if (!payload || !Array.isArray(payload.lead_hours) || payload.lead_hours.length === 0) {
    ctx.fillStyle = "#666";
    ctx.font = "500 20px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText("No data", x0 + 10, y0 + 28);
    return;
  }

  const leads = payload.lead_hours;
  const vals = payload.values || {};
  const control = (vals.control || []).map((v) => (v == null ? null : Number(v)));
  const median = (vals.median || []).map((v) => (v == null ? null : Number(v)));
  const minLine = (vals.min || []).map((v) => (v == null ? null : Number(v)));
  const maxLine = (vals.max || []).map((v) => (v == null ? null : Number(v)));
  const p10 = (vals.p10 || []).map((v) => (v == null ? null : Number(v)));
  const p90 = (vals.p90 || []).map((v) => (v == null ? null : Number(v)));

  const finite = [];
  [control, median, minLine, maxLine, p10, p90].forEach((arr) => {
    arr.forEach((v) => {
      if (Number.isFinite(v)) finite.push(v);
    });
  });
  if (finite.length === 0) {
    ctx.fillStyle = "#666";
    ctx.font = "500 20px IBM Plex Sans, Segoe UI, sans-serif";
    ctx.fillText("No valid values", x0 + 10, y0 + 28);
    return;
  }
  let minV = Math.min(...finite);
  let maxV = Math.max(...finite);
  ({ min: minV, max: maxV } = fullMeteogramScaleForVariable(variableId, minV, maxV));
  const safeSpan = maxV > minV ? maxV - minV : 1.0;
  const innerTop = y0 + 1;
  const innerBottom = y0 + height - 1;
  const innerHeight = Math.max(1, innerBottom - innerTop);
  const yScale = (v) => innerBottom - ((v - minV) / safeSpan) * innerHeight;
  const xScale = (idx) => x0 + (idx / Math.max(1, leads.length - 1)) * width;

  // green grid
  ctx.strokeStyle = "rgba(42,196,65,0.8)";
  ctx.lineWidth = 1.4;
  ctx.setLineDash([3, 5]);
  for (let i = 1; i <= 4; i += 1) {
    const y = y0 + (i / 5) * height;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x0 + width, y);
    ctx.stroke();
  }
  const dayBoundaryIndices = computeLocalDayBoundaryIndices(leads, leadLabelOffsetHours);
  for (const idx of dayBoundaryIndices) {
    const x = xScale(idx);
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y0 + height);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // p10-p90 band
  if (p10.length === leads.length && p90.length === leads.length) {
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < leads.length; i += 1) {
      if (!Number.isFinite(p10[i]) || !Number.isFinite(p90[i])) continue;
      const x = xScale(i);
      const y = yScale(p10[i]);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    for (let i = leads.length - 1; i >= 0; i -= 1) {
      if (!Number.isFinite(p10[i]) || !Number.isFinite(p90[i])) continue;
      ctx.lineTo(xScale(i), yScale(p90[i]));
    }
    if (started) {
      ctx.closePath();
      ctx.fillStyle = "rgba(120,120,120,0.32)";
      ctx.fill();
    }
  }

  const drawLine = (arr, color, widthPx, dash) => {
    if (!arr || arr.length === 0) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = widthPx;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    let started = false;
    const n = Math.min(arr.length, leads.length);
    for (let i = 0; i < n; i += 1) {
      if (!Number.isFinite(arr[i])) continue;
      const x = xScale(i);
      const y = yScale(arr[i]);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    if (started) ctx.stroke();
    ctx.restore();
  };

  drawLine(minLine, "#666", 1.6, [6, 5]);
  drawLine(maxLine, "#666", 1.6, [6, 5]);
  drawLine(median, "#d22", 2.8, []);
  drawLine(control, "#1248ff", 2.7, []);

  // y labels
  ctx.fillStyle = "#111";
  ctx.font = "500 15px IBM Plex Sans, Segoe UI, sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i += 1) {
    const frac = i / 4;
    const y = y0 + height - frac * height;
    const v = minV + frac * (maxV - minV);
    const labelV = formatFullMeteogramTick(variableId, v);
    ctx.strokeStyle = "rgba(42,196,65,0.9)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(x0 - 6, y);
    ctx.lineTo(x0, y);
    ctx.stroke();
    ctx.fillText(labelV, x0 - 10, y + 7);
  }
  ctx.textAlign = "left";

  // x-axis minor/major tick marks on every panel
  ctx.strokeStyle = "rgba(42,196,65,0.9)";
  ctx.lineWidth = 1.4;
  const dayBoundarySet = new Set(dayBoundaryIndices);
  for (let i = 0; i < leads.length; i += 1) {
    const x = xScale(i);
    const isMajor = dayBoundarySet.has(i);
    const tickH = isMajor ? 10 : 5;
    ctx.beginPath();
    ctx.moveTo(x, y0 + height);
    ctx.lineTo(x, y0 + height + tickH);
    ctx.stroke();
  }

  // bottom lead labels on final panel only
  if (isBottomPanel) {
    const dayEntries = [];
    const byDay = new Map();
    for (let i = 0; i < leads.length; i += 1) {
      const fcLead = Number(leads[i]) + Number(leadLabelOffsetHours || 0);
      const dt = validTimeFromInitAndLead(state.init, fcLead);
      const dayKey = new Intl.DateTimeFormat("en-CA", {
        timeZone: "Europe/Zurich",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
      }).format(dt);
      const prev = byDay.get(dayKey);
      if (!prev) {
        byDay.set(dayKey, { first: i, last: i, dt });
      } else {
        prev.last = i;
      }
    }
    for (const entry of byDay.values()) {
      dayEntries.push({ index: Math.round((entry.first + entry.last) / 2), dt: entry.dt });
    }
    const dayIndices = chooseTickIndicesLocal(dayEntries.length, 6);
    ctx.fillStyle = "#111";
    ctx.font = "700 17px IBM Plex Sans, Segoe UI, sans-serif";
    for (const di of dayIndices) {
      const item = dayEntries[di];
      if (!item) continue;
      const x = xScale(item.index);
      const txt = new Intl.DateTimeFormat("en-US", {
        weekday: "short",
        day: "2-digit",
        timeZone: "Europe/Zurich",
      })
        .format(item.dt)
        .replace(",", "")
        .toUpperCase();
      const tw = ctx.measureText(txt).width;
      ctx.fillText(txt, x - tw / 2, y0 + height + 34);
    }

    const midIdx = Math.floor(leads.length / 2);
    const midLead = Number(leads[midIdx] || 0) + Number(leadLabelOffsetHours || 0);
    const midDt = validTimeFromInitAndLead(state.init, midLead);
    const monthYear = new Intl.DateTimeFormat("en-US", {
      month: "long",
      year: "numeric",
      timeZone: "Europe/Zurich",
    })
      .format(midDt)
      .toUpperCase();
    ctx.font = "700 17px IBM Plex Sans, Segoe UI, sans-serif";
    const mw = ctx.measureText(monthYear).width;
    ctx.fillText(monthYear, x0 + width * 0.5 - mw * 0.5, y0 + height + 76);
  }
}

function computeLocalDayBoundaryIndices(leads, leadLabelOffsetHours) {
  const indices = [];
  let prevDayKey = null;
  for (let i = 0; i < leads.length; i += 1) {
    const fcLead = Number(leads[i]) + Number(leadLabelOffsetHours || 0);
    const dt = validTimeFromInitAndLead(state.init, fcLead);
    const dayKey = new Intl.DateTimeFormat("en-CA", {
      timeZone: "Europe/Zurich",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(dt);
    if (i > 0 && dayKey !== prevDayKey) {
      indices.push(i);
    }
    prevDayKey = dayKey;
  }
  return indices;
}

function chooseTickIndicesLocal(count, maxTicks) {
  if (count <= 0) return [];
  if (count <= maxTicks) {
    return Array.from({ length: count }, (_, i) => i);
  }
  const out = [0];
  const step = (count - 1) / (maxTicks - 1);
  for (let i = 1; i < maxTicks - 1; i += 1) {
    out.push(Math.round(i * step));
  }
  out.push(count - 1);
  return Array.from(new Set(out)).sort((a, b) => a - b);
}

function fullMeteogramScaleForVariable(variableId, minV, maxV) {
  if (!Number.isFinite(minV) || !Number.isFinite(maxV) || maxV <= minV) {
    return { min: 0, max: 1 };
  }
  if (variableId === "clct") {
    return { min: 0, max: 100 };
  }
  if (variableId === "tot_prec") {
    const maxNice = niceUpperBound(Math.max(1, maxV * 1.2));
    return { min: 0, max: maxNice };
  }
  if (variableId === "vmax_10m" || variableId === "wind_speed_10m") {
    const maxNice = niceUpperBound(Math.max(5, maxV * 1.1));
    return { min: 0, max: maxNice };
  }
  const span = maxV - minV;
  return { min: minV - span * 0.1, max: maxV + span * 0.1 };
}

function niceUpperBound(value) {
  const v = Math.max(1e-6, Number(value));
  const exp = 10 ** Math.floor(Math.log10(v));
  const r = v / exp;
  if (r <= 1) return 1 * exp;
  if (r <= 2) return 2 * exp;
  if (r <= 5) return 5 * exp;
  return 10 * exp;
}

function formatFullMeteogramTick(variableId, v) {
  const val = Number(v);
  if (!Number.isFinite(val)) return "-";
  if (variableId === "clct" || variableId === "vmax_10m" || variableId === "wind_speed_10m") {
    return String(Math.round(val));
  }
  if (variableId === "tot_prec") {
    return val >= 10 ? val.toFixed(0) : val.toFixed(1);
  }
  return Math.abs(val) >= 10 ? val.toFixed(0) : val.toFixed(1);
}

function parseUrlState() {
  const p = new URLSearchParams(window.location.search);
  const parsed = {
    datasetId: p.get("model") || null,
    typeId: p.get("stat") || null,
    timeOperator: p.get("op") || null,
    variableId: p.get("var") || null,
    init: p.get("run") || null,
    lead: p.get("lead") != null ? Number(p.get("lead")) : null,
    zoom: p.get("z") != null ? Number(p.get("z")) : null,
    centerLat: p.get("lat") != null ? Number(p.get("lat")) : null,
    centerLon: p.get("lon") != null ? Number(p.get("lon")) : null,
    pointLat: p.get("plat") != null ? Number(p.get("plat")) : null,
    pointLon: p.get("plon") != null ? Number(p.get("plon")) : null,
    pointLabel: p.get("plabel") || "",
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

  const timeOperators = state.metadata.time_operators || ds.time_operators || [{ time_operator: "none", display_name: "None" }];
  els.timeOperator.innerHTML = timeOperators
    .map((op) => `<option value="${op.time_operator}">${op.display_name}</option>`)
    .join("");
  state.timeOperator =
    (urlState.timeOperator &&
      timeOperators.some((op) => op.time_operator === urlState.timeOperator) &&
      urlState.timeOperator) ||
    (timeOperators[0]?.time_operator || "none");
  els.timeOperator.value = state.timeOperator;

  const dsVariables = sortedVariables(ds);
  els.variable.innerHTML = dsVariables
    .map((v) => `<option value="${v.variable_id}">${v.display_name}</option>`)
    .join("");
  state.variableId =
    (urlState.variableId &&
      dsVariables.some((v) => v.variable_id === urlState.variableId) &&
      urlState.variableId) ||
    dsVariables[0]?.variable_id ||
    null;
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
    state.pinnedLocationName = String(urlState.pointLabel || "");
    state.pinnedModelPoint = nearestModelGridPoint(state.datasetId, state.pinnedPoint.lat, state.pinnedPoint.lon);
    updatePinnedPointText();
    void resolvePinnedPointElevations();
  }

  renderLegend();
  renderRefreshStatus();
  renderMapSummary();
}

function applyMapUrlState(urlState) {
  if (!urlState || !isMapStyleReady()) {
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
  if (state.timeOperator && state.timeOperator !== "none") params.set("op", state.timeOperator);
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
    if (state.pinnedLocationName) {
      params.set("plabel", state.pinnedLocationName);
    }
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

bootstrap().catch((err) => {
  console.error(err);
  if (els.catalogInfo) {
    els.catalogInfo.textContent = `Startup failed: ${err.message}`;
    els.catalogInfo.classList.add("loading");
  }
});
