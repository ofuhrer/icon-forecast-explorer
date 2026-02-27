import unittest
from unittest.mock import patch
import json
from datetime import datetime, timedelta

import numpy as np

from weather_data import DatasetMeta, ForecastStore


class WeatherDataTests(unittest.TestCase):
    def test_cached_field_does_not_force_refresh_for_unknown_init(self):
        store = ForecastStore()
        with patch.object(store, "refresh_catalog", wraps=store.refresh_catalog) as mocked_refresh:
            result = store.get_cached_field(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init_str="1900010100",
                lead_hour=0,
                type_id="control",
            )
        self.assertIsNone(result)
        mocked_refresh.assert_not_called()

    def test_unit_normalization_temperature_and_wind(self):
        store = ForecastStore()
        temp_k = np.array([[273.15, 280.15]], dtype=np.float32)
        temp_c = store._normalize_variable_units(temp_k, "t_2m", "K")
        np.testing.assert_allclose(temp_c, np.array([[0.0, 7.0]], dtype=np.float32), atol=1e-3)

        wind_ms = np.array([[10.0]], dtype=np.float32)
        wind_kmh = store._normalize_variable_units(wind_ms, "wind_speed_10m", "m/s")
        np.testing.assert_allclose(wind_kmh, np.array([[36.0]], dtype=np.float32), atol=1e-3)

        precip_m = np.array([[0.002]], dtype=np.float32)
        precip_mm = store._normalize_variable_units(precip_m, "tot_prec", "m")
        np.testing.assert_allclose(precip_mm, np.array([[2.0]], dtype=np.float32), atol=1e-3)

        cloud_fraction = np.array([[0.25]], dtype=np.float32)
        cloud_percent = store._normalize_variable_units(cloud_fraction, "clct", "fraction")
        np.testing.assert_allclose(cloud_percent, np.array([[25.0]], dtype=np.float32), atol=1e-3)

        sunshine_seconds = np.array([[1800.0]], dtype=np.float32)
        sunshine_minutes = store._normalize_variable_units(sunshine_seconds, "dursun", "s")
        np.testing.assert_allclose(sunshine_minutes, np.array([[30.0]], dtype=np.float32), atol=1e-3)

        pressure_pa = np.array([[101325.0]], dtype=np.float32)
        pressure_hpa = store._normalize_variable_units(pressure_pa, "pres_sfc", "Pa")
        np.testing.assert_allclose(pressure_hpa, np.array([[1013.25]], dtype=np.float32), atol=1e-3)

        rain_m = np.array([[0.003]], dtype=np.float32)
        rain_mm = store._normalize_variable_units(rain_m, "rain_gsp", "m")
        np.testing.assert_allclose(rain_mm, np.array([[3.0]], dtype=np.float32), atol=1e-3)

    def test_extract_units_hint_uses_grib_units_and_standard_fallback(self):
        store = ForecastStore()

        class _FakeArrayWithGribUnits:
            attrs = {"GRIB_units": "s"}

        class _FakeArrayWithoutUnits:
            attrs = {}

        units_from_grib = store._extract_units_hint(_FakeArrayWithGribUnits(), "DURSUN")
        self.assertEqual(units_from_grib, "s")

        units_from_standard = store._extract_units_hint(_FakeArrayWithoutUnits(), "PS")
        self.assertEqual(units_from_standard, "Pa")

    def test_dursun_normalization_with_missing_units_uses_range_fallback(self):
        store = ForecastStore()
        # Represents a full hour in seconds but missing explicit unit metadata.
        sunshine_seconds = np.array([[3600.0]], dtype=np.float32)
        sunshine_minutes = store._normalize_variable_units(sunshine_seconds, "dursun", "")
        np.testing.assert_allclose(sunshine_minutes, np.array([[60.0]], dtype=np.float32), atol=1e-3)

    def test_merge_catalogs_keeps_cached_on_regression(self):
        store = ForecastStore()
        cfg = DatasetMeta(
            dataset_id="icon-ch1-eps-control",
            display_name="ICON-CH1-EPS",
            collection_id="c",
            ogd_collection="o",
            expected_members_total=11,
            fallback_cycle_hours=3,
            fallback_lead_hours=[0, 1, 2],
        )
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        run_hours = [0, 3, 6, 9, 12, 15, 18, 21]
        init_times = [(now - timedelta(hours=h)).strftime("%Y%m%d%H") for h in run_hours]
        cached = {
            "init_times": init_times,
            "lead_hours": [0, 1, 2],
            "init_to_leads": {init: [0, 1, 2] for init in init_times},
        }
        discovered = {
            "init_times": [init_times[0]],
            "lead_hours": [0, 1],
            "init_to_leads": {init_times[0]: [0, 1]},
        }

        merged = store._merge_catalogs(cfg, cached, discovered)
        self.assertEqual(merged, cached)

    def test_cached_field_returns_none_for_unknown_lead_in_known_init(self):
        store = ForecastStore()
        dataset_id = "icon-ch1-eps-control"
        init = store.init_times(dataset_id)[0]
        bad_lead = 9999

        with patch.object(store, "refresh_catalog", wraps=store.refresh_catalog) as mocked_refresh:
            result = store.get_cached_field(
                dataset_id=dataset_id,
                variable_id="t_2m",
                init_str=init,
                lead_hour=bad_lead,
                type_id="control",
            )
        self.assertIsNone(result)
        mocked_refresh.assert_not_called()

    def test_cached_field_rejects_invalid_variable(self):
        store = ForecastStore()
        init = store.init_times("icon-ch1-eps-control")[0]
        with self.assertRaises(ValueError):
            store.get_cached_field(
                dataset_id="icon-ch1-eps-control",
                variable_id="does_not_exist",
                init_str=init,
                lead_hour=0,
                type_id="control",
            )

    def test_cached_field_handles_corrupt_npz(self):
        store = ForecastStore()
        dataset_id = "icon-ch1-eps-control"
        init = store.init_times(dataset_id)[0]
        lead = 0
        path = store._field_cache_path(dataset_id, "control", "t_2m", init, lead)
        path.write_bytes(b"")
        result = store.get_cached_field(
            dataset_id=dataset_id,
            variable_id="t_2m",
            init_str=init,
            lead_hour=lead,
            type_id="control",
        )
        self.assertIsNone(result)
        self.assertFalse(path.exists())

    def test_fill_nan_with_neighbors_does_not_wrap_edges(self):
        grid = np.array([[np.nan, 1.0, np.nan, np.nan, 100.0]], dtype=np.float32)
        filled = ForecastStore._fill_nan_with_neighbors(grid)
        # Left edge may only be influenced by in-domain neighbors, not wrapped far edge values.
        self.assertLess(float(filled[0, 0]), 10.0)
        self.assertGreater(float(filled[0, 4]), 90.0)

    def test_load_catalog_cache_accepts_naive_timestamp(self):
        store = ForecastStore()
        cfg = store._dataset_config("icon-ch1-eps-control")
        path = store._catalog_cache_path(cfg.dataset_id)
        previous = path.read_text() if path.exists() else None
        payload = {
            "fetched_at": datetime.utcnow().isoformat(),
            "init_times": ["2026022500"],
            "lead_hours": [0, 1],
            "init_to_leads": {"2026022500": [0, 1]},
        }
        try:
            path.write_text(json.dumps(payload))
            loaded = store._load_catalog_cache(cfg)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["init_times"], ["2026022500"])
        finally:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(previous)

    def test_extract_display_lead_offset_from_end_step(self):
        class _FakeArray:
            attrs = {"endStep": 5}

        offset = ForecastStore._extract_display_lead_offset_hours(_FakeArray(), requested_lead_hour=4)
        self.assertEqual(offset, 1)

    def test_extract_display_lead_offset_from_step_range(self):
        class _FakeArray:
            attrs = {"stepRange": "4-5"}

        offset = ForecastStore._extract_display_lead_offset_hours(_FakeArray(), requested_lead_hour=4)
        self.assertEqual(offset, 1)

    def test_time_operator_window_is_trailing_and_uses_available_leads(self):
        store = ForecastStore()
        dataset_id = "icon-ch1-eps-control"
        init = store.init_times(dataset_id)[0]
        with patch.object(store, "lead_hours_for_init", return_value=[0, 1, 3, 4, 5]):
            window, kind = store._time_operator_window(dataset_id, init, 5, "avg_3h")
        self.assertEqual(kind, "avg")
        self.assertEqual(window, [3, 4, 5])

    def test_time_operator_field_averages_base_fields(self):
        store = ForecastStore()
        dataset_id = "icon-ch1-eps-control"
        init = store.init_times(dataset_id)[0]
        base0 = np.ones((2, 2), dtype=np.float32) * 2.0
        base1 = np.ones((2, 2), dtype=np.float32) * 4.0
        with patch.object(store, "lead_hours_for_init", return_value=[0, 1]):
            with patch.object(
                store,
                "get_field",
                side_effect=lambda dataset_id, variable_id, init_str, lead_hour, type_id="control", time_operator="none": (
                    base0 if lead_hour == 0 else base1
                ),
            ):
                field, info = store._compute_time_operated_field(
                    dataset_id=dataset_id,
                    variable_id="t_2m",
                    init_str=init,
                    lead_hour=1,
                    type_id="control",
                    time_operator="avg_3h",
                )
        np.testing.assert_allclose(field, np.ones((2, 2), dtype=np.float32) * 3.0)
        self.assertEqual(info["window_leads"], [0, 1])


if __name__ == "__main__":
    unittest.main()
