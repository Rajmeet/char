mod common;

use axum::http::StatusCode;
use transcribe_cactus::{STATUS_PATH, TranscribeReadiness, TranscribeReadinessState};

fn audio_wav_bytes() -> Vec<u8> {
    std::fs::read(hypr_data::english_1::AUDIO_PATH).expect("failed to read audio file")
}

use common::{
    invalid_model_path, start_server_with_model_path, start_test_server, wait_for_status,
};

#[ignore = "requires local cactus model files"]
#[test]
fn e2e_batch() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let (addr, shutdown_tx) = start_test_server(Default::default()).await;

        let wav_bytes = audio_wav_bytes();

        let url = format!(
            "http://{}/v1/listen?channels=1&sample_rate=16000&language=en",
            addr
        );
        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("content-type", "audio/wav")
            .body(wav_bytes)
            .send()
            .await
            .expect("request failed");

        assert_eq!(response.status(), 200);
        let v: serde_json::Value = response.json().await.expect("response is not JSON");

        let transcript = v
            .pointer("/results/channels/0/alternatives/0/transcript")
            .and_then(|t| t.as_str())
            .unwrap_or("");

        let transcript_lower = transcript.trim().to_lowercase();
        assert!(
            !transcript_lower.is_empty(),
            "expected non-empty transcript"
        );
        assert!(
            transcript_lower.contains("maybe")
                || transcript_lower.contains("this")
                || transcript_lower.contains("talking"),
            "transcript looks like a hallucination (got: {:?})",
            transcript_lower
        );
        assert!(
            v["metadata"]["duration"].as_f64().unwrap_or_default() > 0.0,
            "expected positive duration in metadata"
        );
        assert_eq!(v["metadata"]["channels"], 1);

        let _ = shutdown_tx.send(());
    });
}

#[tokio::test]
async fn invalid_model_path_returns_http_500_json_error() {
    let (addr, shutdown_tx) =
        start_server_with_model_path(invalid_model_path(), Default::default()).await;
    wait_for_status(addr, TranscribeReadinessState::Failed).await;

    let response = reqwest::Client::new()
        .post(format!(
            "http://{}/v1/listen?channels=1&sample_rate=16000&language=en",
            addr
        ))
        .header("content-type", "audio/wav")
        .body(audio_wav_bytes())
        .send()
        .await
        .expect("request failed");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: serde_json::Value = response.json().await.expect("response is not JSON");
    assert_eq!(body["error"], "model_load_failed");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("model file not found"),
        "unexpected detail: {body:?}"
    );

    let _ = shutdown_tx.send(());
}

#[tokio::test]
async fn invalid_model_path_returns_sse_error_event() {
    let (addr, shutdown_tx) =
        start_server_with_model_path(invalid_model_path(), Default::default()).await;
    wait_for_status(addr, TranscribeReadinessState::Failed).await;

    let response = reqwest::Client::new()
        .post(format!(
            "http://{}/v1/listen?channels=1&sample_rate=16000&language=en",
            addr
        ))
        .header("content-type", "audio/wav")
        .header("accept", "text/event-stream")
        .body(audio_wav_bytes())
        .send()
        .await
        .expect("request failed");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: serde_json::Value = response.json().await.expect("response is not JSON");
    assert_eq!(body["error"], "model_load_failed");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("model file not found"),
        "unexpected detail: {body:?}"
    );

    let _ = shutdown_tx.send(());
}

#[tokio::test]
async fn health_is_live_even_when_model_failed() {
    let (addr, shutdown_tx) =
        start_server_with_model_path(invalid_model_path(), Default::default()).await;
    wait_for_status(addr, TranscribeReadinessState::Failed).await;

    let health = reqwest::get(format!("http://{addr}/health"))
        .await
        .expect("health request failed");
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(health.text().await.unwrap(), "ok");

    let status = reqwest::get(format!("http://{addr}{STATUS_PATH}"))
        .await
        .expect("status request failed");
    assert_eq!(status.status(), StatusCode::OK);
    let body: TranscribeReadiness = status.json().await.expect("status is not JSON");
    assert_eq!(body.status, TranscribeReadinessState::Failed);
    assert!(
        body.error
            .as_deref()
            .unwrap_or_default()
            .contains("model file not found")
    );

    let _ = shutdown_tx.send(());
}
