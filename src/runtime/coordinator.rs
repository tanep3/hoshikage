use super::{
    transition, RuntimeAction, RuntimeEndpoint, RuntimeFailure, RuntimeGeneration, RuntimePhase,
    RuntimeTransitionError, StopReason,
};
use crate::conversation::{ModelId, RequestId};
use crate::inference::LoadedRuntimeInfo;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Error)]
pub enum RuntimeAcquireError {
    #[error("runtime admission queue is full")]
    ServerBusy,
    #[error("runtime queue wait timed out")]
    QueueTimeout,
    #[error("runtime state lock is poisoned")]
    StatePoisoned,
    #[error(transparent)]
    Transition(#[from] RuntimeTransitionError),
}

struct RuntimeCoordinatorInner {
    phase: Mutex<RuntimePhase>,
    admission: Arc<Semaphore>,
    execution: Arc<Semaphore>,
    queue_timeout: Duration,
    next_generation: AtomicU64,
}

#[derive(Clone)]
pub struct RuntimeCoordinator {
    inner: Arc<RuntimeCoordinatorInner>,
}

impl RuntimeCoordinator {
    pub fn new(queue_capacity: usize, queue_timeout: Duration) -> Self {
        Self {
            inner: Arc::new(RuntimeCoordinatorInner {
                phase: Mutex::new(RuntimePhase::Cold),
                admission: Arc::new(Semaphore::new(queue_capacity)),
                execution: Arc::new(Semaphore::new(1)),
                queue_timeout,
                next_generation: AtomicU64::new(1),
            }),
        }
    }

    pub async fn prepare(
        &self,
        model: ModelId,
        request_id: RequestId,
    ) -> Result<RuntimePreparation, RuntimeAcquireError> {
        let queue_ticket = self
            .inner
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| RuntimeAcquireError::ServerBusy)?;
        let execution_permit = tokio::time::timeout(
            self.inner.queue_timeout,
            self.inner.execution.clone().acquire_owned(),
        )
        .await
        .map_err(|_| RuntimeAcquireError::QueueTimeout)?
        .map_err(|_| RuntimeAcquireError::StatePoisoned)?;
        drop(queue_ticket);

        let mut phase = self
            .inner
            .phase
            .lock()
            .map_err(|_| RuntimeAcquireError::StatePoisoned)?;
        let needs_start = match &*phase {
            RuntimePhase::Ready(ready)
                if ready.model == model && ready.active_request.is_none() =>
            {
                false
            }
            RuntimePhase::Ready(ready) if ready.active_request.is_none() => {
                *phase = transition(
                    &phase,
                    RuntimeAction::Stop {
                        reason: StopReason::ModelSwitch,
                    },
                )?
                .next;
                true
            }
            RuntimePhase::Cold | RuntimePhase::Failed(_) => true,
            _ => {
                return Err(RuntimeAcquireError::Transition(
                    RuntimeTransitionError::InvalidTransition {
                        phase: "active",
                        action: "prepare",
                    },
                ))
            }
        };
        let generation = if needs_start {
            let generation =
                RuntimeGeneration::new(self.inner.next_generation.fetch_add(1, Ordering::Relaxed));
            *phase = transition(
                &phase,
                RuntimeAction::Start {
                    generation,
                    model: model.clone(),
                },
            )?
            .next;
            generation
        } else {
            match &*phase {
                RuntimePhase::Ready(ready) => ready.generation,
                _ => unreachable!("ready phase checked above"),
            }
        };
        drop(phase);

        Ok(RuntimePreparation {
            request_id,
            model,
            generation,
            needs_start,
            coordinator: self.inner.clone(),
            completed: false,
            execution_permit: Some(execution_permit),
        })
    }

    pub fn phase(&self) -> Result<RuntimePhase, RuntimeAcquireError> {
        self.inner
            .phase
            .lock()
            .map(|phase| phase.clone())
            .map_err(|_| RuntimeAcquireError::StatePoisoned)
    }

    pub fn active_requests(&self) -> usize {
        self.phase()
            .ok()
            .and_then(|phase| match phase {
                RuntimePhase::Ready(ready) => Some(usize::from(ready.active_request.is_some())),
                _ => None,
            })
            .unwrap_or(0)
    }

    pub fn mark_unhealthy(
        &self,
        generation: RuntimeGeneration,
        reason: impl Into<String>,
    ) -> Result<(), RuntimeAcquireError> {
        let mut phase = self
            .inner
            .phase
            .lock()
            .map_err(|_| RuntimeAcquireError::StatePoisoned)?;
        *phase = transition(
            &phase,
            RuntimeAction::MarkUnhealthy {
                generation,
                failure: RuntimeFailure {
                    reason: reason.into(),
                },
            },
        )?
        .next;
        Ok(())
    }
}

pub struct RuntimePreparation {
    request_id: RequestId,
    model: ModelId,
    generation: RuntimeGeneration,
    needs_start: bool,
    coordinator: Arc<RuntimeCoordinatorInner>,
    completed: bool,
    execution_permit: Option<OwnedSemaphorePermit>,
}

impl RuntimePreparation {
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn generation(&self) -> RuntimeGeneration {
        self.generation
    }

    pub fn needs_start(&self) -> bool {
        self.needs_start
    }

    pub fn activate(
        mut self,
        endpoint: RuntimeEndpoint,
        loaded: LoadedRuntimeInfo,
    ) -> Result<RuntimeLease, RuntimeAcquireError> {
        let mut phase = self
            .coordinator
            .phase
            .lock()
            .map_err(|_| RuntimeAcquireError::StatePoisoned)?;
        if self.needs_start {
            *phase = transition(
                &phase,
                RuntimeAction::MarkReady {
                    generation: self.generation,
                    endpoint,
                    loaded,
                },
            )?
            .next;
        }
        *phase = transition(
            &phase,
            RuntimeAction::Acquire {
                request: self.request_id.clone(),
            },
        )?
        .next;
        let (endpoint, loaded) = match &*phase {
            RuntimePhase::Ready(ready) => (ready.endpoint.clone(), ready.loaded.clone()),
            _ => unreachable!("acquire transition produces ready state"),
        };
        drop(phase);
        self.completed = true;

        Ok(RuntimeLease {
            request_id: self.request_id.clone(),
            generation: self.generation,
            endpoint,
            loaded,
            coordinator: self.coordinator.clone(),
            released: false,
            execution_permit: self.execution_permit.take(),
        })
    }

    pub fn fail(mut self, reason: impl Into<String>) -> Result<(), RuntimeAcquireError> {
        let mut phase = self
            .coordinator
            .phase
            .lock()
            .map_err(|_| RuntimeAcquireError::StatePoisoned)?;
        *phase = transition(
            &phase,
            RuntimeAction::MarkUnhealthy {
                generation: self.generation,
                failure: RuntimeFailure {
                    reason: reason.into(),
                },
            },
        )?
        .next;
        self.completed = true;
        Ok(())
    }
}

impl Drop for RuntimePreparation {
    fn drop(&mut self) {
        if self.completed || !self.needs_start {
            return;
        }
        let Ok(mut phase) = self.coordinator.phase.lock() else {
            return;
        };
        if !matches!(
            &*phase,
            RuntimePhase::Starting(starting) if starting.generation == self.generation
        ) {
            return;
        }
        if let Ok(failed) = transition(
            &phase,
            RuntimeAction::MarkUnhealthy {
                generation: self.generation,
                failure: RuntimeFailure {
                    reason: "runtime preparation dropped before activation".to_string(),
                },
            },
        ) {
            *phase = failed.next;
        }
    }
}

pub struct RuntimeLease {
    request_id: RequestId,
    generation: RuntimeGeneration,
    endpoint: RuntimeEndpoint,
    loaded: LoadedRuntimeInfo,
    coordinator: Arc<RuntimeCoordinatorInner>,
    released: bool,
    execution_permit: Option<OwnedSemaphorePermit>,
}

impl RuntimeLease {
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn generation(&self) -> RuntimeGeneration {
        self.generation
    }

    pub fn endpoint(&self) -> &RuntimeEndpoint {
        &self.endpoint
    }

    pub fn loaded(&self) -> &LoadedRuntimeInfo {
        &self.loaded
    }

    pub fn finish(mut self) -> Result<(), RuntimeAcquireError> {
        self.release()
    }

    fn release(&mut self) -> Result<(), RuntimeAcquireError> {
        if self.released {
            return Ok(());
        }
        let mut phase = self
            .coordinator
            .phase
            .lock()
            .map_err(|_| RuntimeAcquireError::StatePoisoned)?;
        match &*phase {
            RuntimePhase::Failed(failed) if failed.generation == self.generation => {}
            _ => {
                *phase = transition(
                    &phase,
                    RuntimeAction::Release {
                        request: self.request_id.clone(),
                    },
                )?
                .next;
            }
        }
        self.released = true;
        drop(phase);
        self.execution_permit.take();
        Ok(())
    }
}

impl Drop for RuntimeLease {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            tracing::error!(error = %error, "Failed to release runtime lease");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SpeculationMode;
    use std::path::PathBuf;

    fn loaded() -> LoadedRuntimeInfo {
        LoadedRuntimeInfo {
            main_model: PathBuf::from("model.gguf"),
            mmproj: None,
            draft_model: None,
            n_ctx: 16384,
            n_gpu_layers: 0,
            vision_supported: false,
            vision_marker: None,
            speculation_enabled: false,
            speculation_mode: SpeculationMode::Off,
            speculation_fallback_reason: None,
        }
    }

    #[tokio::test]
    async fn lease_holds_execution_permit_until_stream_owner_drops_it() {
        let coordinator = RuntimeCoordinator::new(2, Duration::from_secs(1));
        let first = coordinator
            .prepare(
                ModelId::new("model-a").unwrap(),
                RequestId::new("request-1").unwrap(),
            )
            .await
            .unwrap()
            .activate(
                RuntimeEndpoint::new("http://127.0.0.1:13030").unwrap(),
                loaded(),
            )
            .unwrap();

        let second_coordinator = coordinator.clone();
        let second = tokio::spawn(async move {
            second_coordinator
                .prepare(
                    ModelId::new("model-b").unwrap(),
                    RequestId::new("request-2").unwrap(),
                )
                .await
        });

        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!second.is_finished());
        assert_eq!(coordinator.active_requests(), 1);

        drop(first);
        let preparation = second.await.unwrap().unwrap();
        assert!(preparation.needs_start());
        assert_eq!(preparation.model().as_str(), "model-b");
    }

    #[tokio::test]
    async fn bounded_admission_rejects_requests_beyond_queue_capacity() {
        let coordinator = RuntimeCoordinator::new(1, Duration::from_secs(1));
        let first = coordinator
            .prepare(
                ModelId::new("model-a").unwrap(),
                RequestId::new("request-1").unwrap(),
            )
            .await
            .unwrap()
            .activate(
                RuntimeEndpoint::new("http://127.0.0.1:13030").unwrap(),
                loaded(),
            )
            .unwrap();

        let queued_coordinator = coordinator.clone();
        let queued = tokio::spawn(async move {
            queued_coordinator
                .prepare(
                    ModelId::new("model-a").unwrap(),
                    RequestId::new("request-2").unwrap(),
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(25)).await;

        let rejected = coordinator
            .prepare(
                ModelId::new("model-a").unwrap(),
                RequestId::new("request-3").unwrap(),
            )
            .await;
        assert!(matches!(rejected, Err(RuntimeAcquireError::ServerBusy)));

        drop(first);
        drop(queued.await.unwrap());
    }
}
