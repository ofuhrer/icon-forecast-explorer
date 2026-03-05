/**
 * Wind vector feature builder.
 *
 * Converts raw ``/api/wind-vectors`` vector data into MapLibre/Mapbox
 * GeoJSON ``LineString`` features representing arrow glyphs.
 *
 * The function is extracted here so it can be tested independently of the
 * map instance; the caller passes the live ``map`` object so that pixel
 * projections remain accurate.
 */

/**
 * Build an array of GeoJSON LineString features representing wind arrows.
 *
 * Each source vector produces three line segments: the arrow shaft and two
 * head barbs.
 *
 * @param {Array<{speed: number, u: number, v: number, lat: number, lon: number}>} vectors
 *   Wind vector data returned by the API.
 * @param {import('maplibre-gl').Map} mapInstance
 *   Live MapLibre GL ``Map`` instance used for pixel ↔ coordinate projection.
 * @returns {Array<object>} GeoJSON Feature objects.
 */
export function buildWindVectorFeatures(vectors, mapInstance) {
  const features = [];
  for (const vec of vectors) {
    const speed = Number(vec.speed);
    const u = Number(vec.u);
    const v = Number(vec.v);
    if (!Number.isFinite(speed) || !Number.isFinite(u) || !Number.isFinite(v)) {
      continue;
    }
    const p0 = mapInstance.project([vec.lon, vec.lat]);
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
    const ll0 = mapInstance.unproject([p0.x, p0.y]);
    const ll1 = mapInstance.unproject([p1.x, p1.y]);
    const lll = mapInstance.unproject([left.x, left.y]);
    const llr = mapInstance.unproject([right.x, right.y]);
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
