import unittest
from unittest.mock import patch

try:
    import app as app_module
except ModuleNotFoundError:
    app_module = None


class _FakeStore:
    def lead_hours_for_init(self, dataset_id, init):
        return [0, 1, 2]

    def get_cached_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control"):
        if lead == 1 and type_id == "p10":
            raise RuntimeError("cached gap")
        if lead == 2:
            return None
        return 5.0 + lead

    def get_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control"):
        if lead == 1 and type_id == "control":
            raise RuntimeError("full fetch failed")
        return 10.0 + lead


class ApiSeriesTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
