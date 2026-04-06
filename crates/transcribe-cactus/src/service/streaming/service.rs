use std::{
    future::Future,
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use axum::{
    Json,
    body::Body,
    extract::{FromRequestParts, ws::WebSocketUpgrade},
    http::{Request, Response, StatusCode, header},
    response::IntoResponse,
};
use hypr_model_manager::{ModelLoadState, ModelManager, ModelManagerBuilder};
use hypr_transcribe_core::json_error_response;
use tower::Service;

use hypr_ws_utils::ConnectionManager;
use owhisper_interface::ListenParams;

use super::super::batch;
use super::session;
use crate::CactusConfig;

type CactusModelManager = ModelManager<hypr_cactus::Model>;

const RETRY_AFTER_SECS: u64 = 5;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscribeReadinessState {
    Idle,
    Loading,
    Ready,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TranscribeReadiness {
    pub status: TranscribeReadinessState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_secs: Option<u64>,
}

#[derive(Clone)]
pub struct TranscribeReadinessHandle {
    manager: CactusModelManager,
}

impl TranscribeReadinessHandle {
    pub async fn snapshot(&self) -> Result<TranscribeReadiness, hypr_model_manager::Error> {
        let readiness = match self.manager.snapshot(None).await? {
            ModelLoadState::Idle => TranscribeReadiness {
                status: TranscribeReadinessState::Idle,
                error: None,
                retry_after_secs: Some(RETRY_AFTER_SECS),
            },
            ModelLoadState::Loading => TranscribeReadiness {
                status: TranscribeReadinessState::Loading,
                error: None,
                retry_after_secs: Some(RETRY_AFTER_SECS),
            },
            ModelLoadState::Ready => TranscribeReadiness {
                status: TranscribeReadinessState::Ready,
                error: None,
                retry_after_secs: None,
            },
            ModelLoadState::Failed { error } => TranscribeReadiness {
                status: TranscribeReadinessState::Failed,
                error: Some(error),
                retry_after_secs: None,
            },
        };

        Ok(readiness)
    }
}

#[derive(Clone)]
pub struct TranscribeService {
    model_path: PathBuf,
    manager: CactusModelManager,
    cactus_config: CactusConfig,
    connection_manager: ConnectionManager,
}

pub const LISTEN_PATH: &str = "/v1/listen";
pub const HEALTH_PATH: &str = "/health";
pub const STATUS_PATH: &str = "/status";

enum ReadyModel {
    Ready(Arc<hypr_cactus::Model>),
    Loading,
    Failed(String),
}

impl TranscribeService {
    pub fn builder() -> TranscribeServiceBuilder {
        TranscribeServiceBuilder::default()
    }

    pub fn readiness_handle(&self) -> TranscribeReadinessHandle {
        TranscribeReadinessHandle {
            manager: self.manager.clone(),
        }
    }

    pub fn into_router<F, Fut>(self, on_error: F) -> axum::Router
    where
        F: FnOnce(String) -> Fut + Clone + Send + Sync + 'static,
        Fut: std::future::Future<Output = (StatusCode, String)> + Send,
    {
        let readiness = self.readiness_handle();
        let svc = axum::error_handling::HandleError::new(self, on_error);
        axum::Router::new()
            .route(HEALTH_PATH, axum::routing::get(|| async { "ok" }))
            .route(
                STATUS_PATH,
                axum::routing::get(move || {
                    let readiness = readiness.clone();
                    async move {
                        match readiness.snapshot().await {
                            Ok(status) => Json(status).into_response(),
                            Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
                                .into_response(),
                        }
                    }
                }),
            )
            .route_service(LISTEN_PATH, svc)
    }

    async fn ready_model(&self) -> Result<ReadyModel, hypr_model_manager::Error> {
        if let Some(model) = self.manager.get_if_ready(None).await? {
            return Ok(ReadyModel::Ready(model));
        }

        match self.manager.snapshot(None).await? {
            ModelLoadState::Ready => Ok(self
                .manager
                .get_if_ready(None)
                .await?
                .map_or(ReadyModel::Loading, ReadyModel::Ready)),
            ModelLoadState::Idle | ModelLoadState::Loading => {
                let _ = self.manager.ensure_loading(None).await?;
                Ok(ReadyModel::Loading)
            }
            ModelLoadState::Failed { error } => {
                let _ = self.manager.ensure_loading(None).await?;
                Ok(ReadyModel::Failed(error))
            }
        }
    }
}

#[derive(Default)]
pub struct TranscribeServiceBuilder {
    model_path: Option<PathBuf>,
    cactus_config: CactusConfig,
    connection_manager: Option<ConnectionManager>,
}

impl TranscribeServiceBuilder {
    pub fn model_path(mut self, model_path: PathBuf) -> Self {
        self.model_path = Some(model_path);
        self
    }

    pub fn cactus_config(mut self, config: CactusConfig) -> Self {
        self.cactus_config = config;
        self
    }

    pub fn build(self) -> TranscribeService {
        crate::service::ensure_log_init();

        let model_path = self
            .model_path
            .expect("TranscribeServiceBuilder requires model_path");

        let manager = ModelManagerBuilder::default()
            .register("default", &model_path)
            .default_model("default")
            .build();

        let warmup_manager = manager.clone();
        tokio::spawn(async move {
            match warmup_manager.ensure_loading(None).await {
                Ok(true) | Ok(false) => tracing::info!("model warmup scheduled"),
                Err(error) => tracing::warn!(error = %error, "failed_to_schedule_model_warmup"),
            }
        });

        TranscribeService {
            model_path,
            manager,
            cactus_config: self.cactus_config,
            connection_manager: self.connection_manager.unwrap_or_default(),
        }
    }
}

impl Service<Request<Body>> for TranscribeService {
    type Response = Response<Body>;
    type Error = String;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let model_path = self.model_path.clone();
        let manager = self.manager.clone();
        let cactus_config = self.cactus_config.clone();
        let connection_manager = self.connection_manager.clone();
        let readiness = self.clone();

        Box::pin(async move {
            let is_ws = req
                .headers()
                .get("upgrade")
                .and_then(|v| v.to_str().ok())
                .map(|v| v.eq_ignore_ascii_case("websocket"))
                .unwrap_or(false);

            let query_string = req.uri().query().unwrap_or("");
            let params = match parse_listen_params(query_string) {
                Ok(p) => p,
                Err(e) => {
                    return Ok((StatusCode::BAD_REQUEST, e.to_string()).into_response());
                }
            };

            let model = match readiness.ready_model().await {
                Ok(ReadyModel::Ready(model)) => model,
                Ok(ReadyModel::Loading) => {
                    return Ok(not_ready_response(is_ws));
                }
                Ok(ReadyModel::Failed(error)) => {
                    return Ok(load_failed_response(is_ws, &error));
                }
                Err(error) => {
                    tracing::error!(error = %error, "failed_to_check_model_readiness");
                    return Ok(load_failed_response(is_ws, &error.to_string()));
                }
            };

            if is_ws {
                let metadata = crate::service::build_metadata(&model_path);
                let (mut parts, _body) = req.into_parts();
                let ws_upgrade = match WebSocketUpgrade::from_request_parts(&mut parts, &()).await {
                    Ok(ws) => ws,
                    Err(e) => {
                        return Ok((StatusCode::BAD_REQUEST, e.to_string()).into_response());
                    }
                };

                let guard = connection_manager.acquire_connection();

                Ok(ws_upgrade
                    .on_upgrade(move |socket| async move {
                        session::handle_websocket(
                            socket,
                            params,
                            model,
                            metadata,
                            cactus_config,
                            guard,
                            manager,
                        )
                        .await;
                    })
                    .into_response())
            } else {
                let content_type = req
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("application/octet-stream")
                    .to_string();

                let accept = req
                    .headers()
                    .get("accept")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();

                let body_bytes =
                    match axum::body::to_bytes(req.into_body(), 100 * 1024 * 1024).await {
                        Ok(b) => b,
                        Err(e) => {
                            return Ok((StatusCode::BAD_REQUEST, e.to_string()).into_response());
                        }
                    };

                if body_bytes.is_empty() {
                    return Ok((StatusCode::BAD_REQUEST, "request body is empty").into_response());
                }

                if accept.contains("text/event-stream") {
                    Ok(batch::handle_batch_sse(
                        body_bytes,
                        &content_type,
                        &params,
                        model,
                        &model_path,
                    )
                    .await)
                } else {
                    Ok(
                        batch::handle_batch(body_bytes, &content_type, &params, model, &model_path)
                            .await,
                    )
                }
            }
        })
    }
}

fn parse_listen_params(query: &str) -> Result<ListenParams, serde_html_form::de::Error> {
    serde_html_form::from_str(query)
}

fn not_ready_response(is_ws: bool) -> axum::response::Response {
    let mut response = if is_ws {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "model is warming up; retry later",
        )
            .into_response()
    } else {
        json_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_loading",
            "model is warming up; retry later",
        )
    };
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, header::HeaderValue::from_static("5"));
    response
}

fn load_failed_response(is_ws: bool, error: &str) -> axum::response::Response {
    if is_ws {
        let message = if error.starts_with("failed to load model: ") {
            error.to_string()
        } else {
            format!("failed to load model: {error}")
        };
        (StatusCode::INTERNAL_SERVER_ERROR, message).into_response()
    } else {
        json_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "model_load_failed",
            error.to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hypr_language::ISO639;

    #[test]
    fn parse_single_language() {
        let params = parse_listen_params("language=en").unwrap();
        assert_eq!(params.languages.len(), 1);
        assert_eq!(params.languages[0].iso639(), ISO639::En);
    }

    #[test]
    fn parse_multiple_languages() {
        let params = parse_listen_params("language=en&language=ko").unwrap();
        assert_eq!(params.languages.len(), 2);
        assert_eq!(params.languages[0].iso639(), ISO639::En);
        assert_eq!(params.languages[1].iso639(), ISO639::Ko);
    }

    #[test]
    fn parse_no_languages() {
        let params = parse_listen_params("").unwrap();
        assert!(params.languages.is_empty());
    }

    #[test]
    fn parse_with_keywords() {
        let params = parse_listen_params("language=en&keywords=hello&keywords=world").unwrap();
        assert_eq!(params.languages.len(), 1);
        assert_eq!(params.keywords, vec!["hello", "world"]);
    }

    #[test]
    fn defaults_channels_and_sample_rate_when_omitted() {
        let params = parse_listen_params("").unwrap();
        assert_eq!(params.channels, 1);
        assert_eq!(params.sample_rate, 16000);
    }

    #[test]
    fn not_ready_response_sets_retry_after() {
        let response = not_ready_response(false);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers().get(header::RETRY_AFTER).unwrap(), "5");
    }
}
