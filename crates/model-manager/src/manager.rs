use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use tokio::sync::{Mutex, Notify, RwLock, watch};

use crate::builder::DropGuard;
use crate::{Error, ModelLoader};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelLoadState {
    Idle,
    Loading,
    Ready,
    Failed { error: String },
}

pub(crate) struct ActiveModel<M> {
    pub(crate) name: String,
    pub(crate) model: Arc<M>,
}

pub(crate) enum LoadState<M> {
    Idle,
    Loading { name: String },
    Ready(ActiveModel<M>),
    Failed { name: String, error: String },
}

pub(crate) struct ManagerState<M> {
    pub(crate) load_state: LoadState<M>,
    pub(crate) last_activity: Option<tokio::time::Instant>,
}

impl<M> Default for ManagerState<M> {
    fn default() -> Self {
        Self {
            load_state: LoadState::Idle,
            last_activity: None,
        }
    }
}

pub struct ModelManager<M: ModelLoader> {
    pub(crate) registry: Arc<RwLock<HashMap<String, PathBuf>>>,
    pub(crate) default_model: Arc<RwLock<Option<String>>>,
    pub(crate) state: Arc<Mutex<ManagerState<M>>>,
    pub(crate) load_notify: Arc<Notify>,
    pub(crate) inactivity_timeout: Duration,
    pub(crate) _drop_guard: Arc<DropGuard>,
}

impl<M: ModelLoader> Clone for ModelManager<M> {
    fn clone(&self) -> Self {
        Self {
            registry: Arc::clone(&self.registry),
            default_model: Arc::clone(&self.default_model),
            state: Arc::clone(&self.state),
            load_notify: Arc::clone(&self.load_notify),
            inactivity_timeout: self.inactivity_timeout,
            _drop_guard: Arc::clone(&self._drop_guard),
        }
    }
}

impl<M: ModelLoader> ModelManager<M> {
    pub fn builder() -> crate::ModelManagerBuilder<M> {
        crate::ModelManagerBuilder::default()
    }

    pub async fn register(&self, name: impl Into<String>, path: impl Into<PathBuf>) {
        let mut reg = self.registry.write().await;
        reg.insert(name.into(), path.into());
    }

    pub async fn unregister(&self, name: &str) {
        let mut reg = self.registry.write().await;
        reg.remove(name);

        let mut state = self.state.lock().await;
        if self.is_state_for_name(&state.load_state, name) {
            state.load_state = LoadState::Idle;
            state.last_activity = None;
            drop(state);
            self.load_notify.notify_waiters();
        }
    }

    pub async fn set_default(&self, name: impl Into<String>) {
        let mut default = self.default_model.write().await;
        *default = Some(name.into());
    }

    pub async fn get(&self, name: Option<&str>) -> Result<Arc<M>, Error> {
        let resolved = self.resolve_name(name).await?;
        let mut waited_for_load = false;

        loop {
            if let Some(model) = self.get_if_ready(Some(&resolved)).await? {
                return Ok(model);
            }

            match self.snapshot(Some(&resolved)).await? {
                ModelLoadState::Idle => {
                    self.ensure_loading(Some(&resolved)).await?;
                }
                ModelLoadState::Loading => {}
                ModelLoadState::Ready => continue,
                ModelLoadState::Failed { error } if waited_for_load => {
                    return Err(Error::StoredLoadFailure(error));
                }
                ModelLoadState::Failed { .. } => {
                    self.ensure_loading(Some(&resolved)).await?;
                }
            }

            let notified = self.load_notify.notified();
            match self.snapshot(Some(&resolved)).await? {
                ModelLoadState::Ready => continue,
                ModelLoadState::Failed { error } => {
                    return Err(Error::StoredLoadFailure(error));
                }
                ModelLoadState::Idle => continue,
                ModelLoadState::Loading => {
                    waited_for_load = true;
                    notified.await;
                }
            }
        }
    }

    pub async fn ensure_loading(&self, name: Option<&str>) -> Result<bool, Error> {
        let resolved = self.resolve_name(name).await?;
        let path = self.resolve_path(&resolved).await?;

        let should_spawn = {
            let mut state = self.state.lock().await;
            let _ = self.expire_if_inactive(&mut state);
            match self.snapshot_from_state(&state.load_state, &resolved) {
                ModelLoadState::Ready | ModelLoadState::Loading => false,
                ModelLoadState::Idle | ModelLoadState::Failed { .. } => {
                    state.load_state = LoadState::Loading {
                        name: resolved.clone(),
                    };
                    state.last_activity = None;
                    true
                }
            }
        };

        if !should_spawn {
            return Ok(false);
        }

        self.load_notify.notify_waiters();

        let state = Arc::clone(&self.state);
        let load_notify = Arc::clone(&self.load_notify);
        tokio::spawn(async move {
            let result = Self::load_model(path).await;
            let mut state = state.lock().await;
            let is_current = matches!(
                &state.load_state,
                LoadState::Loading { name } if name == &resolved
            );
            if !is_current {
                return;
            }

            match result {
                Ok(model) => {
                    state.load_state = LoadState::Ready(ActiveModel {
                        name: resolved,
                        model,
                    });
                    state.last_activity = Some(tokio::time::Instant::now());
                }
                Err(error) => {
                    state.load_state = LoadState::Failed {
                        name: resolved,
                        error: error.to_string(),
                    };
                    state.last_activity = None;
                }
            }

            drop(state);
            load_notify.notify_waiters();
        });

        Ok(true)
    }

    pub async fn snapshot(&self, name: Option<&str>) -> Result<ModelLoadState, Error> {
        let resolved = self.resolve_name(name).await?;
        let mut state = self.state.lock().await;
        let should_notify = self.expire_if_inactive(&mut state);
        let snapshot = self.snapshot_from_state(&state.load_state, &resolved);
        drop(state);

        if should_notify {
            self.load_notify.notify_waiters();
        }

        Ok(snapshot)
    }

    pub async fn get_if_ready(&self, name: Option<&str>) -> Result<Option<Arc<M>>, Error> {
        let resolved = self.resolve_name(name).await?;
        let mut state = self.state.lock().await;
        let should_notify = self.expire_if_inactive(&mut state);

        let model = match &state.load_state {
            LoadState::Ready(active) if active.name == resolved => Some(Arc::clone(&active.model)),
            _ => None,
        };

        if model.is_some() {
            state.last_activity = Some(tokio::time::Instant::now());
        }
        drop(state);

        if should_notify {
            self.load_notify.notify_waiters();
        }

        Ok(model)
    }

    pub async fn keep_alive(&self) {
        let mut state = self.state.lock().await;
        if matches!(state.load_state, LoadState::Ready(_)) {
            state.last_activity = Some(tokio::time::Instant::now());
        }
    }

    async fn resolve_name(&self, name: Option<&str>) -> Result<String, Error> {
        match name {
            Some(name) => Ok(name.to_string()),
            None => {
                let default = self.default_model.read().await;
                default.clone().ok_or(Error::NoDefaultModel)
            }
        }
    }

    async fn resolve_path(&self, name: &str) -> Result<PathBuf, Error> {
        let reg = self.registry.read().await;
        reg.get(name)
            .cloned()
            .ok_or_else(|| Error::ModelNotRegistered(name.to_string()))
    }

    async fn load_model(path: PathBuf) -> Result<Arc<M>, Error> {
        if !path.exists() {
            return Err(Error::ModelFileNotFound(path.display().to_string()));
        }

        let model = tokio::task::spawn_blocking(move || M::load(&path))
            .await
            .map_err(|_| Error::WorkerPanicked)?
            .map_err(|error| Error::Load(Box::new(error)))?;

        Ok(Arc::new(model))
    }

    fn snapshot_from_state(&self, state: &LoadState<M>, resolved: &str) -> ModelLoadState {
        match state {
            LoadState::Idle => ModelLoadState::Idle,
            LoadState::Loading { name } if name == resolved => ModelLoadState::Loading,
            LoadState::Ready(active) if active.name == resolved => ModelLoadState::Ready,
            LoadState::Failed { name, error } if name == resolved => ModelLoadState::Failed {
                error: error.clone(),
            },
            _ => ModelLoadState::Idle,
        }
    }

    fn is_state_for_name(&self, state: &LoadState<M>, name: &str) -> bool {
        match state {
            LoadState::Idle => false,
            LoadState::Loading { name: current } => current == name,
            LoadState::Ready(active) => active.name == name,
            LoadState::Failed { name: current, .. } => current == name,
        }
    }

    fn expire_if_inactive(&self, state: &mut ManagerState<M>) -> bool {
        let should_expire = matches!(state.load_state, LoadState::Ready(_))
            && state
                .last_activity
                .is_some_and(|t| t.elapsed() > self.inactivity_timeout);

        if should_expire {
            state.load_state = LoadState::Idle;
            state.last_activity = None;
        }

        should_expire
    }

    pub(crate) fn spawn_monitor(
        &self,
        check_interval: Duration,
        mut shutdown_rx: watch::Receiver<()>,
    ) {
        let state = Arc::clone(&self.state);
        let load_notify = Arc::clone(&self.load_notify);
        let inactivity_timeout = self.inactivity_timeout;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(check_interval);
            interval.tick().await;

            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => break,
                    _ = interval.tick() => {
                        let should_notify = {
                            let mut state = state.lock().await;
                            let should_expire = matches!(state.load_state, LoadState::Ready(_))
                                && state
                                    .last_activity
                                    .is_some_and(|t| t.elapsed() > inactivity_timeout);

                            if should_expire {
                                state.load_state = LoadState::Idle;
                                state.last_activity = None;
                            }

                            should_expire
                        };

                        if should_notify {
                            load_notify.notify_waiters();
                        }
                    }
                }
            }
        });
    }
}
