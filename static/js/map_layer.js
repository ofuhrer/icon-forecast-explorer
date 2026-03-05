/**
 * Weather tile layer management — section marker module.
 *
 * The tile-layer and field-debug logic (``anyDatasetRefreshing``,
 * ``isMapStyleReady``, ``reconcileWeatherLayerOnIdle``,
 * ``weatherTileUrl``, ``addOrReplaceWeatherLayer``, ``onMapDataLoading``,
 * ``onMapSourceData``, ``onMapError``, ``setTileLoadingVisible``,
 * ``setTileUnavailable``, ``scheduleTileRetry``, ``clearTileRetry``,
 * ``scheduleFieldDebugProbe``, ``clearFieldDebugProbe``,
 * ``maybeLoadFieldDebugInfo``, ``loadFieldDebugInfo``,
 * ``prefetchUpcomingLeads``, ``refreshHoverValueIfNeeded``,
 * ``requestHoverValue``) lives in ``main.js`` under the
 * ``// ─── Section: Map Layer ───`` block.  These functions share a large
 * amount of mutable module-level state (abort controllers, timers, request
 * versions, etc.) which makes them difficult to extract without a factory
 * pattern.
 *
 * This file exists as a placeholder so that the module reference in
 * ``static/index.html`` and ``AGENTS.md`` is accurate.  Nothing is exported
 * from here yet; future work can migrate these functions using a
 * ``createMapLayerManager(deps)`` factory pattern.
 */
