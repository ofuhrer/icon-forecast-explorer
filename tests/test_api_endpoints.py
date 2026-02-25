import unittest
from unittest.mock import patch

import numpy as np

try:
    import app as app_module
    from weather_data import DatasetMeta, VariableMeta
except ModuleNotFoundError:
    app_module = None
    DatasetMeta = None
    VariableMeta = None


if DatasetMeta is not None and VariableMeta is not None:
    _DATASET_METAS = [
        DatasetMeta(
            dataset_id="icon-ch1-eps-control",
            display_name="ICON-CH1-EPS",
            collection_id="x",
            ogd_collection="x",
            expected_members_total=11,
            fallback_cycle_hours=3,
            fallback_lead_hours=[0, 1],
        )
    ]
    _VARIABLE_METAS = [
        VariableMeta(
            variable_id="t_2m",
            display_name="2 m temperature",
            unit="°C",
            min_value=-20.0,
            max_value=40.0,
            ogd_variable="T_2M",
        )
    ]
else:
    _DATASET_METAS = []
    _VARIABLE_METAS = []


class _FakeStore:
    dataset_metas = _DATASET_METAS
    variable_metas = _VARIABLE_METAS

    def __init__(self) -> None:
        self.refresh_calls = []

    def start_background_prewarm(self):
        return None

    def stop_background_prewarm(self):
        return None

    def refresh_catalog(self, force=False, blocking=True, dataset_id=None):
        self.refresh_calls.append({"force": force, "blocking": blocking, "dataset_id": dataset_id})

    def init_times(self, dataset_id):
        return ["2026022500"]

    def init_to_leads(self, dataset_id):
        return {"2026022500": [0, 1]}

    def lead_hours_for_init(self, dataset_id, init):
        return [0, 1]

    def expected_lead_hours_for_init(self, dataset_id, init):
        return [0, 1]

    def refresh_status(self, dataset_id):
        return {"refreshing": False, "last_refreshed_at": "2026-02-25T00:00:00+00:00"}

    def variable_lead_display_offset_hours(self, variable_id):
        return 0

    def get_cached_field(self, dataset_id, variable_id, init, lead, type_id="control"):
        return np.full((380, 540), 5.0, dtype=np.float32)

    def get_field(self, dataset_id, variable_id, init, lead, type_id="control"):
        return np.full((380, 540), 5.0, dtype=np.float32)

    def get_cached_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control"):
        return 5.0

    def queue_field_fetch(self, dataset_id, variable_id, init, lead, type_id="control"):
        return True


class _EmptyMetaStore(_FakeStore):
    dataset_metas = []
    variable_metas = []

    def init_times(self, dataset_id):
        return []

    def init_to_leads(self, dataset_id):
        return {}


class _CacheMissStore(_FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self.get_field_calls = 0

    def get_cached_field(self, dataset_id, variable_id, init, lead, type_id="control"):
        return None

    def get_field(self, dataset_id, variable_id, init, lead, type_id="control"):
        self.get_field_calls += 1
        return np.full((380, 540), 5.0, dtype=np.float32)


class _ValueMissStore(_FakeStore):
    def get_cached_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control"):
        return None


@unittest.skipIf(app_module is None, "fastapi dependencies not available")
class ApiEndpointTests(unittest.TestCase):
    def test_metadata_triggers_non_blocking_refresh(self):
        fake_store = _FakeStore()
        with patch.object(app_module, "store", fake_store):
            payload = app_module.metadata()
        self.assertIn("datasets", payload)
        self.assertEqual(len(payload["datasets"]), 1)
        self.assertTrue(fake_store.refresh_calls)
        self.assertFalse(fake_store.refresh_calls[-1]["force"])
        self.assertFalse(fake_store.refresh_calls[-1]["blocking"])

    def test_metadata_handles_empty_catalog(self):
        fake_store = _EmptyMetaStore()
        with patch.object(app_module, "store", fake_store):
            payload = app_module.metadata()
        self.assertEqual(payload["datasets"], [])
        self.assertEqual(payload["init_times"], [])
        self.assertEqual(payload["lead_hours"], [])

    def test_tiles_endpoint_returns_png(self):
        fake_store = _FakeStore()
        with patch.object(app_module, "store", fake_store):
            response = app_module.tiles(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lead=0,
                z=7,
                x=67,
                y=45,
                type_id="control",
            )
        self.assertEqual(response.media_type, "image/png")
        self.assertTrue(response.body.startswith(b"\x89PNG"))

    def test_tiles_endpoint_sync_fetches_when_not_cached(self):
        fake_store = _CacheMissStore()
        with patch.object(app_module, "store", fake_store):
            response = app_module.tiles(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lead=0,
                z=7,
                x=67,
                y=45,
                type_id="control",
            )
        self.assertEqual(response.media_type, "image/png")
        self.assertEqual(fake_store.get_field_calls, 1)

    def test_value_endpoint_returns_503_when_not_cached(self):
        fake_store = _ValueMissStore()
        with patch.object(app_module, "store", fake_store):
            with self.assertRaises(app_module.HTTPException) as ctx:
                app_module.value(
                    dataset_id="icon-ch1-eps-control",
                    variable_id="t_2m",
                    init="2026022500",
                    lead=0,
                    lat=47.0,
                    lon=8.0,
                    type_id="control",
                )
        self.assertEqual(ctx.exception.status_code, 503)

    def test_prefetch_endpoint_returns_queue_hint(self):
        fake_store = _FakeStore()
        with patch.object(app_module, "store", fake_store):
            payload = app_module.prefetch(
                dataset_id="icon-ch1-eps-control",
                variable_id="t_2m",
                init="2026022500",
                lead=0,
                type_id="control",
            )
        self.assertEqual(payload, {"ok": True, "queued": True})


if __name__ == "__main__":
    unittest.main()
