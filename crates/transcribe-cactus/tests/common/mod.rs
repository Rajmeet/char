use std::net::SocketAddr;
use std::path::PathBuf;

use axum::http::StatusCode;
use transcribe_cactus::{
    CactusConfig, STATUS_PATH, TranscribeReadiness, TranscribeReadinessState, TranscribeService,
};

pub fn model_path() -> PathBuf {
    let path = std::env::var("CACTUS_STT_MODEL").unwrap_or_else(|_| {
        dirs::data_dir()
            .expect("could not find data dir")
            .join("com.hyprnote.dev/models/cactus/whisper-small-int8-apple")
            .to_string_lossy()
            .into_owned()
    });
    let path = PathBuf::from(path);
    assert!(path.exists(), "model not found: {}", path.display());
    path
}

#[allow(dead_code)]
pub fn invalid_model_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "transcribe-cactus-missing-model-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

pub async fn start_test_server(
    cactus_config: CactusConfig,
) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let server = start_server_with_model_path(model_path(), cactus_config).await;
    wait_for_status(server.0, TranscribeReadinessState::Ready).await;
    server
}

pub async fn start_server_with_model_path(
    model_path: PathBuf,
    cactus_config: CactusConfig,
) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let app = TranscribeService::builder()
        .model_path(model_path)
        .cactus_config(cactus_config)
        .build()
        .into_router(|err: String| async move { (StatusCode::INTERNAL_SERVER_ERROR, err) });

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    (addr, shutdown_tx)
}

pub async fn wait_for_status(addr: SocketAddr, expected: TranscribeReadinessState) {
    let client = reqwest::Client::new();

    for _ in 0..40 {
        let response = client
            .get(format!("http://{}{}", addr, STATUS_PATH))
            .send()
            .await
            .expect("status request failed");
        assert_eq!(response.status(), StatusCode::OK);
        let status: TranscribeReadiness = response.json().await.expect("status is not JSON");
        if status.status == expected {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    panic!("timed out waiting for readiness state {expected:?}");
}
