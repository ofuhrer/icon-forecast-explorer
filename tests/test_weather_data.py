import unittest
from unittest.mock import patch
import json
from datetime import datetime, timedelta
import threading

import numpy as np

from weather_data import DatasetMeta, ForecastStore
from weather_grib import fill_nan_with_neighbors, reduce_members


class WeatherDataTests(unittest.TestCase):
    def test_upper_air_level_selector_payload_uses_requested_levels(self):
        store = ForecastStore()

        selector = store.variable_level_selector_payload("temp_3d")

        self.assertTrue(selector["enabled"])
        self.assertEqual(selector["supported_kinds"], ["pressure", "altitude_msl"])
        self.assertEqual(selector["default_kind"], "altitude_msl")
        self.assertEqual(selector["levels"]["pressure"], ["850", "700", "600", "500", "300"])
        self.assertEqual(selector["levels"]["altitude_msl"], ["1500", "2500", "3000", "4000", "5500", "9000"])

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
        filled = fill_nan_with_neighbors(grid)
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
                side_effect=lambda dataset_id, variable_id, init_str, lead_hour, type_id="control", time_operator="none", **kwargs: (
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

    def test_sample_field_value_returns_none_outside_dataset_bounds(self):
        store = ForecastStore()
        field = np.arange(9, dtype=np.float32).reshape(3, 3)
        value = store._sample_field_value("icon-ch1-eps-control", field, lat=0.0, lon=0.0)
        self.assertIsNone(value)

    def test_background_prewarm_can_restart_after_stop(self):
        store = ForecastStore()
        ready = threading.Event()

        def _fake_loop():
            ready.set()
            store._prewarm_stop.wait(0.05)

        with (
            patch("weather_data.HOT_PREWARM_ENABLED", True),
            patch.object(store, "_prewarm_loop", side_effect=_fake_loop),
        ):
            store.start_background_prewarm()
            self.assertTrue(ready.wait(1.0))
            first_executor = store._background_fetch_executor
            store.stop_background_prewarm()

            ready.clear()
            store.start_background_prewarm()
            self.assertTrue(ready.wait(1.0))
            self.assertIsNotNone(store._background_fetch_executor)
            self.assertIsNot(first_executor, store._background_fetch_executor)
            store.stop_background_prewarm()

    def test_start_background_prewarm_always_queues_static_geometry_warm(self):
        store = ForecastStore()
        with (
            patch("weather_data.HOT_PREWARM_ENABLED", False),
            patch.object(store, "_queue_static_geometry_warm") as mocked_queue,
        ):
            store.start_background_prewarm()
        mocked_queue.assert_called_once()
        self.assertFalse(store._prewarm_started)
        store.stop_background_prewarm()

    def test_target_grid_shape_uses_configured_dimensions(self):
        store = ForecastStore()
        self.assertEqual(store.target_grid_shape("icon-ch1-eps-control"), (1759, 1219))
        self.assertEqual(store.target_grid_shape("icon-ch2-eps-control"), (864, 577))

    def test_grid_bounds_are_static_per_dataset(self):
        store = ForecastStore()
        before = store.grid_bounds("icon-ch1-eps-control")
        store._lock_grid_bounds(
            "icon-ch1-eps-control",
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )
        after = store.grid_bounds("icon-ch1-eps-control")
        self.assertEqual(before, after)
        self.assertEqual(
            before,
            {
                "min_lat": 42.4608,
                "max_lat": 50.1249,
                "min_lon": 0.6199,
                "max_lon": 16.6238,
            },
        )

    def test_ensemble_bundle_computed_once_for_multiple_stats(self):
        store = ForecastStore()
        dataset_id = "icon-ch1-eps-control"
        init = store.init_times(dataset_id)[0]
        lead = 0
        variable_id = "t_2m"
        fields_by_type = {
            "mean": np.full((2, 2), 1.0, dtype=np.float32),
            "median": np.full((2, 2), 2.0, dtype=np.float32),
            "p10": np.full((2, 2), 3.0, dtype=np.float32),
            "p90": np.full((2, 2), 4.0, dtype=np.float32),
            "min": np.full((2, 2), 5.0, dtype=np.float32),
            "max": np.full((2, 2), 6.0, dtype=np.float32),
        }
        debug_by_type = {key: {"mode": key, "source_files": ["dummy.grib2"]} for key in fields_by_type}
        control_field = np.full((2, 2), 7.0, dtype=np.float32)
        control_debug = {"mode": "control", "source_files": ["dummy.grib2"]}
        for type_id in fields_by_type:
            for suffix in (".npz", ".json"):
                store._field_cache_path(dataset_id, type_id, variable_id, init, lead).with_suffix(suffix).unlink(
                    missing_ok=True
                )
        for suffix in (".npz", ".json"):
            store._field_cache_path(dataset_id, "control", variable_id, init, lead).with_suffix(suffix).unlink(
                missing_ok=True
            )
        with patch.object(
            store,
            "_fetch_and_regrid_ensemble_bundle",
            return_value=(fields_by_type, debug_by_type, control_field, control_debug),
        ) as mocked_bundle:
            mean = store.get_field(dataset_id, variable_id, init, lead, type_id="mean")
            p10 = store.get_field(dataset_id, variable_id, init, lead, type_id="p10")
            control = store.get_field(dataset_id, variable_id, init, lead, type_id="control")
        np.testing.assert_allclose(mean, fields_by_type["mean"])
        np.testing.assert_allclose(p10, fields_by_type["p10"])
        np.testing.assert_allclose(control, control_field)
        mocked_bundle.assert_called_once()

    def test_safe_asset_urls_for_request_caches_request_lookup(self):
        store = ForecastStore()

        class _FakeRequest:
            collection = "collection"
            variable = "T_2M"
            reference_datetime = "2026-03-08T18:00:00Z"
            perturbed = True
            horizon = timedelta(hours=1)

        class _FakeOgdApi:
            def __init__(self):
                self.calls = 0

            def get_asset_urls(self, request):
                self.calls += 1
                return ["https://example.test/path/file.grib2"]

        fake_ogd = _FakeOgdApi()
        first = store._safe_asset_urls_for_request(fake_ogd, _FakeRequest())
        second = store._safe_asset_urls_for_request(fake_ogd, _FakeRequest())
        self.assertEqual(first, ["https://example.test/path/file.grib2"])
        self.assertEqual(second, ["https://example.test/path/file.grib2"])
        self.assertEqual(fake_ogd.calls, 1)

    def test_reduce_members_percentiles_handle_nans(self):
        members = np.array(
            [
                [[1.0, np.nan], [5.0, 1.0]],
                [[2.0, np.nan], [6.0, 2.0]],
                [[3.0, 9.0], [7.0, 3.0]],
                [[4.0, 11.0], [8.0, np.nan]],
            ],
            dtype=np.float32,
        )
        p10 = reduce_members(members, "p10")
        median = reduce_members(members, "median")
        p90 = reduce_members(members, "p90")
        np.testing.assert_allclose(p10, np.array([[1.3, 9.2], [5.3, 1.2]], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(median, np.array([[2.5, 10.0], [6.5, 2.0]], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(p90, np.array([[3.7, 10.8], [7.7, 2.8]], dtype=np.float32), atol=1e-5)

    def test_meteogram_warmup_reuses_completed_job_for_same_request(self):
        store = ForecastStore()
        init = store.init_times("icon-ch1-eps-control")[0]
        with (
            patch.object(store, "lead_hours_for_init", return_value=[0]),
            patch.object(
                store,
                "_compute_meteogram_warm_tasks",
                return_value=(1, 1, []),
            ) as mocked_compute,
        ):
            first = store.start_meteogram_warmup(
                dataset_id="icon-ch1-eps-control",
                init_str=init,
                variable_ids=["t_2m"],
                type_ids=["control"],
                time_operator="none",
            )
            second = store.start_meteogram_warmup(
                dataset_id="icon-ch1-eps-control",
                init_str=init,
                variable_ids=["t_2m"],
                type_ids=["control"],
                time_operator="none",
            )

        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(first["status"], "done")
        self.assertEqual(second["status"], "done")
        self.assertEqual(mocked_compute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
