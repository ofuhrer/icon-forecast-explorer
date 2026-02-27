import unittest
from unittest.mock import patch

try:
    import app as app_module
except ModuleNotFoundError:
    app_module = None


class _FakeStore:
    def lead_hours_for_init(self, dataset_id, init):
        return [0, 1, 2]

    def get_cached_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control", time_operator="none"):
        if lead == 1 and type_id == "p10":
            raise RuntimeError("cached gap")
        if lead == 2:
            return None
        return 5.0 + lead

    def get_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control", time_operator="none"):
        if lead == 1 and type_id == "control":
            raise RuntimeError("full fetch failed")
        return 10.0 + lead


class _ErrorStore:
    def lead_hours_for_init(self, dataset_id, init):
        return [0, 1, 2]

    def get_cached_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control", time_operator="none"):
        raise RuntimeError(f"cached fail {type_id} {lead}")

    def get_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control", time_operator="none"):
        raise RuntimeError(f"full fail {type_id} {lead}")


class _NoLeadStore:
    def lead_hours_for_init(self, dataset_id, init):
        return []


class _CountingStore:
    def __init__(self):
        self.calls = 0

    def lead_hours_for_init(self, dataset_id, init):
        return [0, 1]

    def get_cached_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control", time_operator="none"):
        self.calls += 1
        return 1.0 + lead

    def get_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control", time_operator="none"):
        self.calls += 1
        return 2.0 + lead


class ApiSeriesTests(unittest.TestCase):
    def setUp(self):
        if app_module is not None:
            app_module._SERIES_CACHE.clear()
            app_module._SERIES_KEY_LOCKS.clear()

    @unittest.skipIf(app_module is None, "fastapi app dependencies not available in current interpreter")
    def test_series_reports_diagnostics_cached_only(self):
        with patch.object(app_module, "store", _FakeStore()):
            payload = app_module.series(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lat=47.0,
                lon=8.0,
                types="control,p10,p90",
                cached_only=True,
            )

        diagnostics = payload["diagnostics"]
        self.assertTrue(diagnostics["cached_only"])
        self.assertGreater(diagnostics["missing_counts"]["control"], 0)
        self.assertGreater(diagnostics["missing_counts"]["p10"], 0)
        self.assertGreaterEqual(diagnostics["error_count"], 1)
        self.assertIn("errors", diagnostics)

    @unittest.skipIf(app_module is None, "fastapi app dependencies not available in current interpreter")
    def test_series_reports_diagnostics_full_fetch(self):
        with patch.object(app_module, "store", _FakeStore()):
            payload = app_module.series(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lat=47.0,
                lon=8.0,
                types="control,p10",
                cached_only=False,
            )

        diagnostics = payload["diagnostics"]
        self.assertFalse(diagnostics["cached_only"])
        self.assertGreaterEqual(diagnostics["error_count"], 1)
        self.assertGreaterEqual(diagnostics["missing_counts"]["control"], 1)

    @unittest.skipIf(app_module is None, "fastapi app dependencies not available in current interpreter")
    def test_series_truncates_error_list(self):
        with patch.object(app_module, "store", _ErrorStore()):
            payload = app_module.series(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lat=47.0,
                lon=8.0,
                types="control,p10,p90",
                cached_only=True,
            )

        diagnostics = payload["diagnostics"]
        self.assertEqual(diagnostics["error_count"], 9)
        self.assertEqual(len(diagnostics["errors"]), 9)
        self.assertEqual(diagnostics["missing_counts"]["control"], 3)
        self.assertEqual(diagnostics["missing_counts"]["p10"], 3)
        self.assertEqual(diagnostics["missing_counts"]["p90"], 3)

    @unittest.skipIf(app_module is None, "fastapi app dependencies not available in current interpreter")
    def test_series_uses_default_type_when_types_empty(self):
        with patch.object(app_module, "store", _FakeStore()):
            payload = app_module.series(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lat=47.0,
                lon=8.0,
                types="   ",
                cached_only=True,
            )
        self.assertIn("control", payload["values"])
        self.assertEqual(list(payload["values"].keys()), ["control"])

    @unittest.skipIf(app_module is None, "fastapi app dependencies not available in current interpreter")
    def test_series_rejects_invalid_init_format(self):
        with patch.object(app_module, "store", _FakeStore()):
            with self.assertRaises(app_module.HTTPException) as ctx:
                app_module.series(
                    dataset_id="icon-ch1-eps-control",
                    variable_id="t_2m",
                    init="not-an-init",
                    lat=47.0,
                    lon=8.0,
                    types="control",
                    cached_only=True,
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Invalid init format", str(ctx.exception.detail))

    @unittest.skipIf(app_module is None, "fastapi app dependencies not available in current interpreter")
    def test_series_rejects_when_no_leads_available(self):
        with patch.object(app_module, "store", _NoLeadStore()):
            with self.assertRaises(app_module.HTTPException) as ctx:
                app_module.series(
                    dataset_id="icon-ch1-eps-control",
                    variable_id="t_2m",
                    init="2026022500",
                    lat=47.0,
                    lon=8.0,
                    types="control",
                    cached_only=True,
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("No leads available", str(ctx.exception.detail))

    @unittest.skipIf(app_module is None, "fastapi app dependencies not available in current interpreter")
    def test_series_cache_hit_on_repeated_request(self):
        counting_store = _CountingStore()
        with patch.object(app_module, "store", counting_store):
            payload1 = app_module.series(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lat=47.001,
                lon=8.001,
                types="control",
                cached_only=True,
            )
            payload2 = app_module.series(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lat=47.002,
                lon=8.002,
                types="control",
                cached_only=True,
            )
        self.assertFalse(payload1["diagnostics"]["cache_hit"])
        self.assertTrue(payload2["diagnostics"]["cache_hit"])
        self.assertEqual(counting_store.calls, 2)

    @unittest.skipIf(app_module is None, "fastapi app dependencies not available in current interpreter")
    def test_series_cache_key_includes_time_operator(self):
        counting_store = _CountingStore()
        with patch.object(app_module, "store", counting_store):
            payload1 = app_module.series(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lat=47.001,
                lon=8.001,
                types="control",
                cached_only=True,
                time_operator="none",
            )
            payload2 = app_module.series(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lat=47.001,
                lon=8.001,
                types="control",
                cached_only=True,
                time_operator="avg_3h",
            )
        self.assertFalse(payload1["diagnostics"]["cache_hit"])
        self.assertFalse(payload2["diagnostics"]["cache_hit"])
        self.assertEqual(counting_store.calls, 4)

    @unittest.skipIf(app_module is None, "fastapi app dependencies not available in current interpreter")
    def test_series_key_locks_are_pruned(self):
        with patch.object(app_module, "SERIES_CACHE_MAX_ENTRIES", 1):
            for i in range(70):
                key = ("d", "v", "i", ("control",), True, i, i, "none")
                app_module._series_key_lock(key)
            keep_key = ("d", "v", "i", ("control",), True, 0, 0, "none")
            app_module._series_cache_put(keep_key, {"ok": True})
        self.assertLessEqual(len(app_module._SERIES_KEY_LOCKS), 64)


if __name__ == "__main__":
    unittest.main()
