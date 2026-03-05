/**
 * Pure SwissTopo search helpers.
 *
 * These functions are stateless and operate only on their explicit arguments,
 * making them easy to unit-test in isolation.
 */

/**
 * Extract the first usable lat/lon result from a swisstopo SearchServer JSON
 * response payload.
 *
 * @param {object} payload - Raw JSON response from the swisstopo API.
 * @returns {{ lat: number, lon: number, label: string, easting: number|null, northing: number|null } | null}
 */
export function firstSwissTopoResult(payload) {
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

/**
 * Strip HTML tags and trailing two-letter canton abbreviations from a raw
 * swisstopo label string.
 *
 * @param {string} labelRaw - Raw label string from the API.
 * @returns {string}
 */
export function normalizeSwissTopoLabel(labelRaw) {
  const clean = String(labelRaw || "")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!clean) {
    return "";
  }
  return clean.replace(/\s+\(([A-Z]{2})\)\s*$/g, "").trim();
}
