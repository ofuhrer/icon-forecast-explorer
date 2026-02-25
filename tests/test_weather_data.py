import unittest
from unittest.mock import patch

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
        cached = {
            "init_times": [f"20260224{h:02d}" for h in (21, 18, 15, 12, 9, 6, 3, 0)],
            "lead_hours": [0, 1, 2],
            "init_to_leads": {f"20260224{h:02d}": [0, 1, 2] for h in (21, 18, 15, 12, 9, 6, 3, 0)},
        }
        discovered = {
            "init_times": ["2026022421"],
            "lead_hours": [0, 1],
            "init_to_leads": {"2026022421": [0, 1]},
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


if __name__ == "__main__":
    unittest.main()
