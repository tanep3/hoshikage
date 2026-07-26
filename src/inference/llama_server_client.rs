use crate::runtime::RuntimeEndpoint;

#[derive(Clone)]
pub struct LlamaServerClient {
    client: reqwest::Client,
}

impl Default for LlamaServerClient {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaServerClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub async fn chat_completions(
        &self,
        endpoint: &RuntimeEndpoint,
        body: &serde_json::Value,
    ) -> reqwest::Result<reqwest::Response> {
        self.client
            .post(Self::chat_completions_url(endpoint))
            .json(body)
            .send()
            .await
    }

    fn chat_completions_url(endpoint: &RuntimeEndpoint) -> String {
        format!("{}/v1/chat/completions", endpoint.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_chat_completions_url_from_runtime_endpoint() {
        let endpoint = RuntimeEndpoint::new("http://127.0.0.1:13030").unwrap();

        assert_eq!(
            LlamaServerClient::chat_completions_url(&endpoint),
            "http://127.0.0.1:13030/v1/chat/completions"
        );
    }
}
