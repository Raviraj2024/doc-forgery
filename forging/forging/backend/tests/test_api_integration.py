from __future__ import annotations

from pathlib import Path


def test_upload_analysis_persistence_and_artifacts_round_trip(
    test_client,
    settings,
    sample_image_bytes: bytes,
) -> None:
    response = test_client.post(
        f"{settings.api_v1_prefix}/analyze",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
        data={
            "document_type": "invoice",
            "submitter_id": "integration-user",
            "tenant_id": "tenant-a",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_id"]
    assert payload["page_count"] == 1
    assert payload["device"] == "cpu"
    assert payload["pages"][0]["tampered_regions"]
    assert payload["forensic_layers"]

    analysis_id = payload["analysis_id"]

    get_response = test_client.get(f"{settings.api_v1_prefix}/analyze/{analysis_id}")
    assert get_response.status_code == 200
    assert get_response.json()["analysis_id"] == analysis_id

    history_response = test_client.get(f"{settings.api_v1_prefix}/analyze?page=1&page_size=10")
    assert history_response.status_code == 200
    assert history_response.json()["total"] == 1

    artifact_url = payload["pages"][0]["artifacts"]["overlay_url"]
    artifact_response = test_client.get(artifact_url)
    assert artifact_response.status_code == 200
    assert artifact_response.headers["content-type"] == "image/png"

    monitoring_response = test_client.get(f"{settings.api_v1_prefix}/devops/monitoring")
    assert monitoring_response.status_code == 200
    monitoring = monitoring_response.json()
    assert monitoring["total_analyses"] == 1
    assert monitoring["p95_processing_time_ms"] >= 0

    analysis_json = settings.outputs_dir / analysis_id / "analysis.json"
    assert analysis_json.exists()
    upload_path = next(settings.uploads_dir.glob(f"{analysis_id}_sample.png"))
    assert upload_path.exists()


def test_model_info_includes_serving_metadata(test_client, settings) -> None:
    response = test_client.get(f"{settings.api_v1_prefix}/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["checkpoint_path"] == str(settings.checkpoint_path)
    assert payload["checkpoint_sha256"] == "test-checkpoint"
    assert payload["calibration_loaded"] is False
    assert payload["calibration_profile_path"] == str(settings.calibration_profile_path)

