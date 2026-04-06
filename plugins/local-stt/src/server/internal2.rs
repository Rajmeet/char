use std::{
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
};

use axum::http::StatusCode;
use ractor::{Actor, ActorName, ActorProcessingErr, ActorRef, RpcReplyPort};
use tower_http::cors::{self, CorsLayer};

use super::{ServerInfo, ServerStatus};
use hypr_cactus_model::CactusSttModel;

pub enum Internal2STTMessage {
    GetHealth(RpcReplyPort<ServerInfo>),
    ServerError(String),
}

#[derive(Clone)]
pub struct Internal2STTArgs {
    pub model_type: CactusSttModel,
    pub model_cache_dir: PathBuf,
    pub cactus_config: hypr_transcribe_cactus::CactusConfig,
}

pub struct Internal2STTState {
    server_addr: SocketAddr,
    model: CactusSttModel,
    readiness: hypr_transcribe_cactus::TranscribeReadinessHandle,
    shutdown: tokio::sync::watch::Sender<()>,
    server_task: tokio::task::JoinHandle<()>,
}

pub struct Internal2STTActor;

impl Internal2STTActor {
    pub fn name() -> ActorName {
        "internal2_stt".into()
    }
}

#[ractor::async_trait]
impl Actor for Internal2STTActor {
    type Msg = Internal2STTMessage;
    type State = Internal2STTState;
    type Arguments = Internal2STTArgs;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let Internal2STTArgs {
            model_type,
            model_cache_dir,
            cactus_config,
        } = args;

        let model_path = model_cache_dir.join(model_type.dir_name());

        tracing::info!(model_path = %model_path.display(), "starting internal2 STT server");

        let service = hypr_transcribe_cactus::TranscribeService::builder()
            .model_path(model_path)
            .cactus_config(cactus_config)
            .build();
        let readiness = service.readiness_handle();

        let router = service
            .into_router(move |err: String| async move {
                let _ = myself.send_message(Internal2STTMessage::ServerError(err.clone()));
                (StatusCode::INTERNAL_SERVER_ERROR, err)
            })
            .layer(
                CorsLayer::new()
                    .allow_origin(cors::Any)
                    .allow_methods(cors::Any)
                    .allow_headers(cors::Any),
            );

        let listener =
            tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).await?;

        let server_addr = listener.local_addr()?;
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(());

        let server_task = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    shutdown_rx.changed().await.ok();
                })
                .await
                .unwrap();
        });

        Ok(Internal2STTState {
            server_addr,
            model: model_type,
            readiness,
            shutdown: shutdown_tx,
            server_task,
        })
    }

    async fn post_stop(
        &self,
        _myself: ActorRef<Self::Msg>,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        let _ = state.shutdown.send(());
        state.server_task.abort();
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            Internal2STTMessage::ServerError(e) => Err(e.into()),
            Internal2STTMessage::GetHealth(reply_port) => {
                let status = match state.readiness.snapshot().await {
                    Ok(readiness) => match readiness.status {
                        hypr_transcribe_cactus::TranscribeReadinessState::Idle
                        | hypr_transcribe_cactus::TranscribeReadinessState::Loading => {
                            ServerStatus::Loading
                        }
                        hypr_transcribe_cactus::TranscribeReadinessState::Ready => {
                            ServerStatus::Ready
                        }
                        hypr_transcribe_cactus::TranscribeReadinessState::Failed => {
                            ServerStatus::Unreachable
                        }
                    },
                    Err(error) => {
                        tracing::error!(error = %error, "failed_to_read_internal2_readiness");
                        ServerStatus::Unreachable
                    }
                };

                let info = ServerInfo {
                    url: Some(format!(
                        "http://{}{}",
                        state.server_addr,
                        hypr_transcribe_cactus::LISTEN_PATH,
                    )),
                    status,
                    model: Some(crate::LocalModel::Cactus(state.model.clone())),
                };

                if let Err(e) = reply_port.send(info) {
                    return Err(e.into());
                }

                Ok(())
            }
        }
    }
}
