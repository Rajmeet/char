mod builder;
mod error;
mod loader;
mod manager;

pub use builder::ModelManagerBuilder;
pub use error::Error;
pub use loader::ModelLoader;
pub use manager::{ModelLoadState, ModelManager};

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use super::*;

    struct MockModel;
    struct SlowModel;
    struct FlakyModel;

    #[derive(Debug, thiserror::Error)]
    #[error("mock error")]
    struct MockError;

    static SLOW_LOADS: AtomicUsize = AtomicUsize::new(0);
    static FLAKY_LOADS: AtomicUsize = AtomicUsize::new(0);
    static FLAKY_SHOULD_FAIL: AtomicBool = AtomicBool::new(false);

    impl ModelLoader for MockModel {
        type Error = MockError;

        fn load(_path: &Path) -> Result<Self, Self::Error> {
            Ok(MockModel)
        }
    }

    impl ModelLoader for SlowModel {
        type Error = MockError;

        fn load(_path: &Path) -> Result<Self, Self::Error> {
            SLOW_LOADS.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(50));
            Ok(SlowModel)
        }
    }

    impl ModelLoader for FlakyModel {
        type Error = MockError;

        fn load(_path: &Path) -> Result<Self, Self::Error> {
            FLAKY_LOADS.fetch_add(1, Ordering::SeqCst);
            if FLAKY_SHOULD_FAIL.load(Ordering::SeqCst) {
                Err(MockError)
            } else {
                Ok(FlakyModel)
            }
        }
    }

    fn temp_model_path() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("model-manager-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{}.bin", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"").unwrap();
        path
    }

    fn build_manager(
        timeout: Duration,
        check_interval: Duration,
        models: &[(&str, std::path::PathBuf)],
    ) -> ModelManager<MockModel> {
        build_manager_for::<MockModel>(timeout, check_interval, models)
    }

    fn build_manager_for<M: ModelLoader>(
        timeout: Duration,
        check_interval: Duration,
        models: &[(&str, std::path::PathBuf)],
    ) -> ModelManager<M> {
        let mut builder = ModelManager::<M>::builder()
            .inactivity_timeout(timeout)
            .check_interval(check_interval);
        for (name, path) in models {
            builder = builder.register(*name, path.clone());
        }
        builder.build()
    }

    #[tokio::test(start_paused = true)]
    async fn idle_model_gets_evicted() {
        let path = temp_model_path();
        let mgr = build_manager(
            Duration::from_millis(100),
            Duration::from_millis(10),
            &[("a", path)],
        );

        let m1 = mgr.get(Some("a")).await.unwrap();
        let m2 = mgr.get(Some("a")).await.unwrap();
        assert!(Arc::ptr_eq(&m1, &m2));

        tokio::time::advance(Duration::from_millis(120)).await;
        tokio::task::yield_now().await;

        let m3 = mgr.get(Some("a")).await.unwrap();
        assert!(!Arc::ptr_eq(&m1, &m3));
    }

    #[tokio::test(start_paused = true)]
    async fn activity_prevents_eviction() {
        let path = temp_model_path();
        let mgr = build_manager(
            Duration::from_millis(100),
            Duration::from_millis(10),
            &[("a", path)],
        );

        let m1 = mgr.get(Some("a")).await.unwrap();

        for _ in 0..5 {
            tokio::time::advance(Duration::from_millis(50)).await;
            tokio::task::yield_now().await;

            let m = mgr.get(Some("a")).await.unwrap();
            assert!(Arc::ptr_eq(&m1, &m));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn access_near_timeout_resets_timer() {
        let path = temp_model_path();
        let mgr = build_manager(
            Duration::from_millis(100),
            Duration::from_millis(10),
            &[("a", path)],
        );

        let m1 = mgr.get(Some("a")).await.unwrap();

        tokio::time::advance(Duration::from_millis(90)).await;
        tokio::task::yield_now().await;

        let m2 = mgr.get(Some("a")).await.unwrap();
        assert!(Arc::ptr_eq(&m1, &m2));

        tokio::time::advance(Duration::from_millis(50)).await;
        tokio::task::yield_now().await;

        let m3 = mgr.get(Some("a")).await.unwrap();
        assert!(Arc::ptr_eq(&m1, &m3));
    }

    #[tokio::test(start_paused = true)]
    async fn access_after_timeout_before_monitor_tick_reloads() {
        let path = temp_model_path();
        let mgr = build_manager(
            Duration::from_millis(100),
            Duration::from_secs(60),
            &[("a", path)],
        );

        let m1 = mgr.get(Some("a")).await.unwrap();

        tokio::time::advance(Duration::from_millis(120)).await;
        tokio::task::yield_now().await;

        let m2 = mgr.get(Some("a")).await.unwrap();
        assert!(!Arc::ptr_eq(&m1, &m2));
    }

    #[tokio::test]
    async fn ensure_loading_starts_one_background_load() {
        SLOW_LOADS.store(0, Ordering::SeqCst);
        let path = temp_model_path();
        let mgr = build_manager_for::<SlowModel>(
            Duration::from_secs(60),
            Duration::from_secs(1),
            &[("a", path)],
        );

        let (a, b) = tokio::join!(mgr.ensure_loading(Some("a")), mgr.ensure_loading(Some("a")));
        assert!(a.unwrap() || b.unwrap());
        assert_eq!(
            mgr.snapshot(Some("a")).await.unwrap(),
            ModelLoadState::Loading
        );

        for _ in 0..20 {
            if mgr.snapshot(Some("a")).await.unwrap() == ModelLoadState::Ready {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        assert_eq!(
            mgr.snapshot(Some("a")).await.unwrap(),
            ModelLoadState::Ready
        );
        assert_eq!(SLOW_LOADS.load(Ordering::SeqCst), 1);
        assert!(mgr.get_if_ready(Some("a")).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn failed_load_can_be_retried() {
        FLAKY_LOADS.store(0, Ordering::SeqCst);
        FLAKY_SHOULD_FAIL.store(true, Ordering::SeqCst);
        let path = temp_model_path();
        let mgr = build_manager_for::<FlakyModel>(
            Duration::from_secs(60),
            Duration::from_secs(1),
            &[("a", path)],
        );

        assert!(mgr.ensure_loading(Some("a")).await.unwrap());
        for _ in 0..20 {
            if matches!(
                mgr.snapshot(Some("a")).await.unwrap(),
                ModelLoadState::Failed { .. }
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        assert!(matches!(
            mgr.snapshot(Some("a")).await.unwrap(),
            ModelLoadState::Failed { .. }
        ));

        FLAKY_SHOULD_FAIL.store(false, Ordering::SeqCst);
        assert!(mgr.ensure_loading(Some("a")).await.unwrap());
        for _ in 0..20 {
            if mgr.snapshot(Some("a")).await.unwrap() == ModelLoadState::Ready {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        assert_eq!(
            mgr.snapshot(Some("a")).await.unwrap(),
            ModelLoadState::Ready
        );
        assert_eq!(FLAKY_LOADS.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn snapshot_returns_idle_after_eviction() {
        let path = temp_model_path();
        let mgr = build_manager(
            Duration::from_millis(100),
            Duration::from_millis(10),
            &[("a", path)],
        );

        assert_eq!(mgr.snapshot(Some("a")).await.unwrap(), ModelLoadState::Idle);
        let _ = mgr.get(Some("a")).await.unwrap();
        assert_eq!(
            mgr.snapshot(Some("a")).await.unwrap(),
            ModelLoadState::Ready
        );

        tokio::time::advance(Duration::from_millis(120)).await;
        tokio::task::yield_now().await;

        assert_eq!(mgr.snapshot(Some("a")).await.unwrap(), ModelLoadState::Idle);
    }
}
