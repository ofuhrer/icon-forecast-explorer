/**
 * Full meteogram — constants and pure utility functions.
 *
 * The complex stateful rendering/warmup logic remains in ``main.js`` (see the
 * ``// ─── Section: Full Meteogram ───`` block there).  Only the portable,
 * side-effect-free pieces live here so they can be imported and tested
 * independently.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Variables shown as panels in the full-meteogram popup. */
export const FULL_METEOGRAM_VARIABLES = ["clct", "tot_prec", "vmax_10m", "t_2m"];

/** Types requested when warming the meteogram cache. */
export const FULL_METEOGRAM_WARMUP_TYPES = "control,median,p10,p90,min,max";

/** How often to poll the warmup-status endpoint (ms). */
export const FULL_METEOGRAM_WARMUP_POLL_MS = 1000;

/** Abort the entire warmup after this many ms. */
export const FULL_METEOGRAM_WARMUP_TIMEOUT_MS = 20 * 60 * 1000;

/** Abort a single status request after this many ms. */
export const FULL_METEOGRAM_WARMUP_STATUS_TIMEOUT_MS = 120 * 1000;

/** Give up after this many consecutive transient errors. */
export const FULL_METEOGRAM_WARMUP_MAX_TRANSIENT_ERRORS = 12;

/** Canvas width for the full-meteogram popup (px). */
export const FULL_METEOGRAM_CANVAS_WIDTH = 700;

/** Height of each meteogram panel row (px). */
export const FULL_METEOGRAM_PANEL_HEIGHT = 172;

/** Top padding inside the canvas (px). */
export const FULL_METEOGRAM_TOP_PAD = 130;

/** Bottom padding inside the canvas (px). */
export const FULL_METEOGRAM_BOTTOM_PAD = 64;

/** Height of the data plot area within a panel (px). */
export const FULL_METEOGRAM_PLOT_HEIGHT = 132;

/** TTL for memoised series data (ms). */
export const FULL_METEOGRAM_MEMO_TTL_MS = 10 * 60 * 1000;

// ---------------------------------------------------------------------------
// Abort / sleep utilities
// ---------------------------------------------------------------------------

/**
 * Create an ``AbortError`` compatible with the Fetch API.
 *
 * @param {string} [message]
 * @returns {Error}
 */
export function createAbortError(message = "Operation aborted") {
  const err = new Error(message);
  err.name = "AbortError";
  return err;
}

/**
 * Return a Promise that resolves after ``ms`` milliseconds, or rejects with an
 * ``AbortError`` if ``signal`` is aborted before the timer fires.
 *
 * @param {number} ms
 * @param {AbortSignal|null} [signal]
 * @returns {Promise<void>}
 */
export function sleepWithSignal(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(createAbortError());
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      if (signal) {
        signal.removeEventListener("abort", onAbort);
      }
      reject(createAbortError());
    };
    const timer = setTimeout(() => {
      if (signal) {
        signal.removeEventListener("abort", onAbort);
      }
      resolve();
    }, ms);
    if (signal) {
      signal.addEventListener("abort", onAbort, { once: true });
    }
  });
}

// ---------------------------------------------------------------------------
// Single-flight deduplication key
// ---------------------------------------------------------------------------

/**
 * Build a cache/deduplication key for a meteogram warmup request.
 *
 * @param {string} datasetId
 * @param {string} init
 * @param {string[]} panels  List of variable IDs.
 * @returns {string}
 */
export function warmupFlightKey(datasetId, init, panels) {
  return `${datasetId}|${init}|${panels.join(",")}`;
}

// ---------------------------------------------------------------------------
// Chart axis helpers
// ---------------------------------------------------------------------------

/**
 * Return indices in ``leads`` that mark the start of a new calendar day
 * (Europe/Zurich time zone).
 *
 * This 4-parameter version is exported for testability.  ``main.js`` retains a
 * 2-parameter wrapper (``computeLocalDayBoundaryIndices(leads, leadLabelOffsetHours)``)
 * that closes over ``state.init`` and the imported ``validTimeFromInitAndLead``
 * helper, so this export is **not** imported by ``main.js``.
 *
 * @param {number[]} leads  Lead hours.
 * @param {number} leadLabelOffsetHours  Display offset added to each lead.
 * @param {string} init  Current forecast init string (``YYYYMMDDhh``).
 * @param {Function} validTimeFromInitAndLead  Pure time helper from formatting.js.
 * @returns {number[]}
 */
export function computeLocalDayBoundaryIndices(leads, leadLabelOffsetHours, init, validTimeFromInitAndLead) {
  const indices = [];
  let prevDayKey = null;
  for (let i = 0; i < leads.length; i += 1) {
    const fcLead = Number(leads[i]) + Number(leadLabelOffsetHours || 0);
    const dt = validTimeFromInitAndLead(init, fcLead);
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

/**
 * Choose up to ``maxTicks`` evenly-spaced indices within ``[0, count)``.
 *
 * @param {number} count  Total number of data points.
 * @param {number} maxTicks  Maximum number of ticks to return.
 * @returns {number[]}
 */
export function chooseTickIndicesLocal(count, maxTicks) {
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

// ---------------------------------------------------------------------------
// Chart scale helpers
// ---------------------------------------------------------------------------

/**
 * Compute a sensible ``{ min, max }`` scale for the given variable.
 *
 * @param {string} variableId
 * @param {number} minV  Observed minimum value.
 * @param {number} maxV  Observed maximum value.
 * @returns {{ min: number, max: number }}
 */
export function fullMeteogramScaleForVariable(variableId, minV, maxV) {
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

/**
 * Round ``value`` up to the nearest "nice" number (1, 2, 5, 10, 20, …).
 *
 * @param {number} value
 * @returns {number}
 */
export function niceUpperBound(value) {
  const v = Math.max(1e-6, Number(value));
  const exp = 10 ** Math.floor(Math.log10(v));
  const r = v / exp;
  if (r <= 1) return 1 * exp;
  if (r <= 2) return 2 * exp;
  if (r <= 5) return 5 * exp;
  return 10 * exp;
}

// ---------------------------------------------------------------------------
// Tick label formatting
// ---------------------------------------------------------------------------

/**
 * Format a chart tick value for display in the full meteogram.
 *
 * @param {string} variableId
 * @param {number} v
 * @returns {string}
 */
export function formatFullMeteogramTick(variableId, v) {
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
