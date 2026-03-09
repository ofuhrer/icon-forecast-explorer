/**
 * URL state parsing helper.
 *
 * Reads the current ``window.location.search`` query string and returns a
 * plain-object snapshot of all recognised parameters.  The function is pure
 * in the sense that it performs no side effects and does not mutate any
 * shared state.
 */

/**
 * Parse the current URL query string into a state snapshot object.
 *
 * @returns {{
 *   datasetId: string|null,
 *   typeId: string|null,
 *   timeOperator: string|null,
 *   variableId: string|null,
 *   levelKind: string|null,
 *   levelValue: string|null,
 *   init: string|null,
 *   lead: number|null,
 *   zoom: number|null,
 *   centerLat: number|null,
 *   centerLon: number|null,
 *   pointLat: number|null,
 *   pointLon: number|null,
 *   pointLabel: string,
 *   speed: string|null,
 * }}
 */
export function parseUrlState() {
  const p = new URLSearchParams(window.location.search);
  const parsed = {
    datasetId: p.get("model") || null,
    typeId: p.get("stat") || null,
    timeOperator: p.get("op") || null,
    variableId: p.get("var") || null,
    levelKind: p.get("lvlkind") || null,
    levelValue: p.get("lvl") || null,
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
