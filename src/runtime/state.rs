use crate::conversation::{ModelId, RequestId};
use crate::inference::LoadedRuntimeInfo;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeGeneration(u64);

impl RuntimeGeneration {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeEndpoint(String);

impl RuntimeEndpoint {
    pub fn new(value: impl Into<String>) -> Result<Self, RuntimeTransitionError> {
        let value = value.into();
        if value.is_empty() {
            return Err(RuntimeTransitionError::InvalidEndpoint);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartingState {
    pub generation: RuntimeGeneration,
    pub model: ModelId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyState {
    pub generation: RuntimeGeneration,
    pub model: ModelId,
    pub endpoint: RuntimeEndpoint,
    pub loaded: LoadedRuntimeInfo,
    pub active_request: Option<RequestId>,
    pub last_access: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedState {
    pub generation: RuntimeGeneration,
    pub model: ModelId,
    pub failure: RuntimeFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimePhase {
    Cold,
    Starting(StartingState),
    Ready(ReadyState),
    Failed(FailedState),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeFailure {
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    ModelSwitch,
    IdleTimeout,
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeAction {
    Start {
        generation: RuntimeGeneration,
        model: ModelId,
    },
    MarkReady {
        generation: RuntimeGeneration,
        endpoint: RuntimeEndpoint,
        loaded: LoadedRuntimeInfo,
    },
    Acquire {
        request: RequestId,
    },
    Release {
        request: RequestId,
    },
    MarkUnhealthy {
        generation: RuntimeGeneration,
        failure: RuntimeFailure,
    },
    Stop {
        reason: StopReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeEffect {
    Spawn {
        generation: RuntimeGeneration,
        model: ModelId,
    },
    WaitUntilReady {
        generation: RuntimeGeneration,
    },
    StopProcess {
        generation: RuntimeGeneration,
    },
    ReleaseMaterializedBundle {
        generation: RuntimeGeneration,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeTransition {
    pub next: RuntimePhase,
    pub effects: Vec<RuntimeEffect>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RuntimeTransitionError {
    #[error("runtime transition is not allowed: {action} from {phase}")]
    InvalidTransition {
        phase: &'static str,
        action: &'static str,
    },
    #[error("runtime generation does not match")]
    StaleGeneration,
    #[error("runtime request does not own the active lease")]
    RequestMismatch,
    #[error("runtime endpoint must not be empty")]
    InvalidEndpoint,
}

fn phase_name(phase: &RuntimePhase) -> &'static str {
    match phase {
        RuntimePhase::Cold => "cold",
        RuntimePhase::Starting(_) => "starting",
        RuntimePhase::Ready(_) => "ready",
        RuntimePhase::Failed(_) => "failed",
    }
}

fn invalid(phase: &RuntimePhase, action: &'static str) -> RuntimeTransitionError {
    RuntimeTransitionError::InvalidTransition {
        phase: phase_name(phase),
        action,
    }
}

pub fn transition(
    current: &RuntimePhase,
    action: RuntimeAction,
) -> Result<RuntimeTransition, RuntimeTransitionError> {
    match (current, action) {
        (
            RuntimePhase::Cold | RuntimePhase::Failed(_),
            RuntimeAction::Start { generation, model },
        ) => Ok(RuntimeTransition {
            next: RuntimePhase::Starting(StartingState {
                generation,
                model: model.clone(),
            }),
            effects: vec![
                RuntimeEffect::Spawn { generation, model },
                RuntimeEffect::WaitUntilReady { generation },
            ],
        }),
        (
            RuntimePhase::Starting(starting),
            RuntimeAction::MarkReady {
                generation,
                endpoint,
                loaded,
            },
        ) => {
            if starting.generation != generation {
                return Err(RuntimeTransitionError::StaleGeneration);
            }
            Ok(RuntimeTransition {
                next: RuntimePhase::Ready(ReadyState {
                    generation,
                    model: starting.model.clone(),
                    endpoint,
                    loaded,
                    active_request: None,
                    last_access: Instant::now(),
                }),
                effects: Vec::new(),
            })
        }
        (RuntimePhase::Ready(ready), RuntimeAction::Acquire { request }) => {
            if ready.active_request.is_some() {
                return Err(invalid(current, "acquire"));
            }
            let mut next = ready.clone();
            next.active_request = Some(request);
            next.last_access = Instant::now();
            Ok(RuntimeTransition {
                next: RuntimePhase::Ready(next),
                effects: Vec::new(),
            })
        }
        (RuntimePhase::Ready(ready), RuntimeAction::Release { request }) => {
            if ready.active_request.as_ref() != Some(&request) {
                return Err(RuntimeTransitionError::RequestMismatch);
            }
            let mut next = ready.clone();
            next.active_request = None;
            next.last_access = Instant::now();
            Ok(RuntimeTransition {
                next: RuntimePhase::Ready(next),
                effects: Vec::new(),
            })
        }
        (
            RuntimePhase::Starting(starting),
            RuntimeAction::MarkUnhealthy {
                generation,
                failure,
            },
        ) => {
            if starting.generation != generation {
                return Err(RuntimeTransitionError::StaleGeneration);
            }
            Ok(RuntimeTransition {
                next: RuntimePhase::Failed(FailedState {
                    generation,
                    model: starting.model.clone(),
                    failure,
                }),
                effects: vec![RuntimeEffect::StopProcess { generation }],
            })
        }
        (
            RuntimePhase::Ready(ready),
            RuntimeAction::MarkUnhealthy {
                generation,
                failure,
            },
        ) => {
            if ready.generation != generation {
                return Err(RuntimeTransitionError::StaleGeneration);
            }
            Ok(RuntimeTransition {
                next: RuntimePhase::Failed(FailedState {
                    generation,
                    model: ready.model.clone(),
                    failure,
                }),
                effects: vec![RuntimeEffect::StopProcess { generation }],
            })
        }
        (RuntimePhase::Ready(ready), RuntimeAction::Stop { .. }) => {
            if ready.active_request.is_some() {
                return Err(invalid(current, "stop"));
            }
            Ok(RuntimeTransition {
                next: RuntimePhase::Cold,
                effects: vec![
                    RuntimeEffect::StopProcess {
                        generation: ready.generation,
                    },
                    RuntimeEffect::ReleaseMaterializedBundle {
                        generation: ready.generation,
                    },
                ],
            })
        }
        (RuntimePhase::Failed(failed), RuntimeAction::Stop { .. }) => Ok(RuntimeTransition {
            next: RuntimePhase::Cold,
            effects: vec![
                RuntimeEffect::StopProcess {
                    generation: failed.generation,
                },
                RuntimeEffect::ReleaseMaterializedBundle {
                    generation: failed.generation,
                },
            ],
        }),
        (RuntimePhase::Cold, RuntimeAction::Stop { .. }) => Ok(RuntimeTransition {
            next: RuntimePhase::Cold,
            effects: Vec::new(),
        }),
        (_, RuntimeAction::Start { .. }) => Err(invalid(current, "start")),
        (_, RuntimeAction::MarkReady { .. }) => Err(invalid(current, "mark_ready")),
        (_, RuntimeAction::Acquire { .. }) => Err(invalid(current, "acquire")),
        (_, RuntimeAction::Release { .. }) => Err(invalid(current, "release")),
        (_, RuntimeAction::MarkUnhealthy { .. }) => Err(invalid(current, "mark_unhealthy")),
        (_, RuntimeAction::Stop { .. }) => Err(invalid(current, "stop")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SpeculationMode;
    use std::path::PathBuf;

    fn model(name: &str) -> ModelId {
        ModelId::new(name).unwrap()
    }

    fn request(name: &str) -> RequestId {
        RequestId::new(name).unwrap()
    }

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

    fn ready(active_request: Option<RequestId>) -> RuntimePhase {
        RuntimePhase::Ready(ReadyState {
            generation: RuntimeGeneration::new(1),
            model: model("model-a"),
            endpoint: RuntimeEndpoint::new("http://127.0.0.1:13030").unwrap(),
            loaded: loaded(),
            active_request,
            last_access: Instant::now(),
        })
    }

    #[test]
    fn cold_start_ready_acquire_release_is_explicit() {
        let started = transition(
            &RuntimePhase::Cold,
            RuntimeAction::Start {
                generation: RuntimeGeneration::new(1),
                model: model("model-a"),
            },
        )
        .unwrap();
        let ready = transition(
            &started.next,
            RuntimeAction::MarkReady {
                generation: RuntimeGeneration::new(1),
                endpoint: RuntimeEndpoint::new("http://127.0.0.1:13030").unwrap(),
                loaded: loaded(),
            },
        )
        .unwrap();
        let acquired = transition(
            &ready.next,
            RuntimeAction::Acquire {
                request: request("request-1"),
            },
        )
        .unwrap();
        let released = transition(
            &acquired.next,
            RuntimeAction::Release {
                request: request("request-1"),
            },
        )
        .unwrap();

        assert!(matches!(
            released.next,
            RuntimePhase::Ready(ReadyState {
                active_request: None,
                ..
            })
        ));
    }

    #[test]
    fn active_runtime_cannot_stop_or_release_another_request() {
        let active = ready(Some(request("request-1")));
        assert!(matches!(
            transition(
                &active,
                RuntimeAction::Stop {
                    reason: StopReason::ModelSwitch
                }
            ),
            Err(RuntimeTransitionError::InvalidTransition { .. })
        ));
        assert_eq!(
            transition(
                &active,
                RuntimeAction::Release {
                    request: request("request-2")
                }
            )
            .unwrap_err(),
            RuntimeTransitionError::RequestMismatch
        );
    }

    #[test]
    fn stale_generation_cannot_mark_runtime_ready_or_failed() {
        let starting = RuntimePhase::Starting(StartingState {
            generation: RuntimeGeneration::new(2),
            model: model("model-a"),
        });
        assert_eq!(
            transition(
                &starting,
                RuntimeAction::MarkReady {
                    generation: RuntimeGeneration::new(1),
                    endpoint: RuntimeEndpoint::new("http://127.0.0.1:13030").unwrap(),
                    loaded: loaded(),
                }
            )
            .unwrap_err(),
            RuntimeTransitionError::StaleGeneration
        );
    }
}
