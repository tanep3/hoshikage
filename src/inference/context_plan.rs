#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextAccuracy {
    Exact,
    Conservative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextPlan {
    pub input_tokens: u32,
    pub reserved_output_tokens: u32,
    pub context_window: u32,
    pub accuracy: ContextAccuracy,
}

impl ContextPlan {
    pub fn validate(self) -> crate::Result<Self> {
        if self
            .input_tokens
            .saturating_add(self.reserved_output_tokens)
            > self.context_window
        {
            return Err(crate::error::HoshikageError::ContextLengthExceeded);
        }
        Ok(self)
    }

    pub fn conservative_from_request_bytes(
        request_bytes: usize,
        reserved_output_tokens: u32,
        context_window: u32,
    ) -> Self {
        Self {
            input_tokens: u32::try_from(request_bytes)
                .unwrap_or(u32::MAX)
                .saturating_add(1024),
            reserved_output_tokens,
            context_window,
            accuracy: ContextAccuracy::Conservative,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_plan_rejects_reserved_output_beyond_context() {
        let error = ContextPlan {
            input_tokens: 15_000,
            reserved_output_tokens: 2_000,
            context_window: 16_384,
            accuracy: ContextAccuracy::Exact,
        }
        .validate()
        .err()
        .unwrap();

        assert!(matches!(
            error,
            crate::error::HoshikageError::ContextLengthExceeded
        ));
    }
}
