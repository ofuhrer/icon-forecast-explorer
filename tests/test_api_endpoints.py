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
        self._grid_width = 540
        self._grid_height = 380
        self._warmup_payload = {
            "job_id": "mw-test-1",
            "dataset_id": "icon-ch1-eps-control",
            "init": "2026022500",
            "variables": ["t_2m"],
            "types": ["control"],
            "time_operator": "none",
            "status": "running",
            "total_tasks": 2,
            "completed_tasks": 1,
            "failed_tasks": 0,
            "remaining_tasks": 1,
            "percent_complete": 50,
            "ready": False,
            "partial": False,
            "errors": [],
        }

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

    def get_cached_field(self, dataset_id, variable_id, init, lead, type_id="control", time_operator="none"):
        return np.full((380, 540), 5.0, dtype=np.float32)

    def get_field(self, dataset_id, variable_id, init, lead, type_id="control", time_operator="none"):
        return np.full((380, 540), 5.0, dtype=np.float32)

    def get_cached_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control", time_operator="none"):
        return 5.0

    def queue_field_fetch(self, dataset_id, variable_id, init, lead, type_id="control", time_operator="none"):
        return True

    def get_cached_wind_vectors(self, dataset_id, init, lead, type_id="control", time_operator="none"):
        return (
            np.full((380, 540), 10.0, dtype=np.float32),
            np.full((380, 540), 0.0, dtype=np.float32),
        )

    def queue_wind_vector_fetch(self, dataset_id, init, lead, type_id="control", time_operator="none"):
        return True

    def get_field_failure(
        self, dataset_id, variable_id, init_str, lead_hour, type_id="control", time_operator="none"
    ):
        return None

    def start_meteogram_warmup(self, dataset_id, init_str, variable_ids, type_ids, time_operator="none"):
        _ = (dataset_id, init_str, variable_ids, type_ids, time_operator)
        return dict(self._warmup_payload)

    def get_meteogram_warmup(self, job_id):
        if str(job_id) != "mw-test-1":
            raise KeyError(str(job_id))
        return dict(self._warmup_payload)


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

    def get_cached_field(self, dataset_id, variable_id, init, lead, type_id="control", time_operator="none"):
        return None

    def get_field(self, dataset_id, variable_id, init, lead, type_id="control", time_operator="none"):
        self.get_field_calls += 1
        return np.full((380, 540), 5.0, dtype=np.float32)


class _ValueMissStore(_FakeStore):
    def get_cached_value(self, dataset_id, variable_id, init, lead, lat, lon, type_id="control", time_operator="none"):
        return None


class _ElevationErrorStore(_FakeStore):
    def get_model_elevation(self, dataset_id, lat, lon):
        _ = (dataset_id, lat, lon)
        raise RuntimeError("HSURF unavailable")


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

    def test_wind_vectors_endpoint_returns_vectors(self):
        fake_store = _FakeStore()
        with patch.object(app_module, "store", fake_store):
            payload = app_module.wind_vectors(
                dataset_id="icon-ch1-eps-control",
                type_id="control",
                init="2026022500",
                lead=0,
                min_lat=46.0,
                max_lat=47.0,
                min_lon=7.0,
                max_lon=8.0,
                zoom=7.0,
            )
        self.assertEqual(payload["status"], "ready")
        self.assertIn("vectors", payload)
        self.assertGreater(len(payload["vectors"]), 0)

    def test_model_elevation_returns_503_on_runtime_error(self):
        fake_store = _ElevationErrorStore()
        with patch.object(app_module, "store", fake_store):
            with self.assertRaises(app_module.HTTPException) as ctx:
                app_module.model_elevation(
                    dataset_id="icon-ch1-eps-control",
                    lat=47.0,
                    lon=8.0,
                )
        self.assertEqual(ctx.exception.status_code, 503)

    def test_meteogram_warmup_start_returns_payload(self):
        fake_store = _FakeStore()
        with patch.object(app_module, "store", fake_store):
            payload = app_module.meteogram_warmup_start(
                dataset_id="icon-ch1-eps-control",
                init="2026022500",
                variables="t_2m",
                types="control",
                time_operator="none",
            )
        self.assertEqual(payload["job_id"], "mw-test-1")
        self.assertEqual(payload["status"], "running")

    def test_meteogram_warmup_status_returns_payload(self):
        fake_store = _FakeStore()
        with patch.object(app_module, "store", fake_store):
            payload = app_module.meteogram_warmup_status(job_id="mw-test-1")
        self.assertEqual(payload["job_id"], "mw-test-1")
        self.assertEqual(payload["percent_complete"], 50)

    def test_meteogram_warmup_status_unknown_job_raises_404(self):
        fake_store = _FakeStore()
        with patch.object(app_module, "store", fake_store):
            with self.assertRaises(app_module.HTTPException) as ctx:
                app_module.meteogram_warmup_status(job_id="mw-unknown")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
