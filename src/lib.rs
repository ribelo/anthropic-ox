use messages::MessagesRequestBuilder;
use serde::Deserialize;
use thiserror::Error;

pub mod messages;

const BASE_URL: &str = "https://api.anthropic.com";

cfg_if::cfg_if! {
    if #[cfg(feature = "leaky-bucket")] {
        use derivative::Derivative;
        use std::sync::Arc;
        pub use leaky_bucket::RateLimiter;
    }
}

#[derive(Derivative)]
#[derivative(Default)]
pub struct AnthropicBuilder {
    api_key: Option<String>,
    #[derivative(Default(value = r#""2023-06-01".to_string()"#))]
    version: String,
    client: Option<reqwest::Client>,
    #[cfg(feature = "leaky-bucket")]
    leaky_bucket: Option<RateLimiter>,
}

#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct Anthropic {
    api_key: String,
    version: String,
    client: reqwest::Client,
    #[derivative(Debug = "ignore")]
    #[cfg(feature = "leaky-bucket")]
    leaky_bucket: Option<Arc<RateLimiter>>,
}

#[derive(Debug, Error)]
pub enum ClientBuilderError {
    #[error("API key not set")]
    ApiKeyNotSet,
}

impl Anthropic {
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::default()
    }

    pub fn messages(&self) -> MessagesRequestBuilder {
        MessagesRequestBuilder {
            client: Some(self.clone()),
            ..Default::default()
        }
    }
}

impl AnthropicBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn api_key<T: ToString>(mut self, api_key: T) -> AnthropicBuilder {
        self.api_key = Some(api_key.to_string());
        self
    }

    pub fn client(mut self, client: &reqwest::Client) -> AnthropicBuilder {
        self.client = Some(client.clone());
        self
    }

    #[cfg(feature = "leaky-bucket")]
    pub fn limiter(mut self, leaky_bucket: RateLimiter) -> AnthropicBuilder {
        self.leaky_bucket = Some(leaky_bucket);
        self
    }

    pub fn build(self) -> Result<Anthropic, ClientBuilderError> {
        let Some(api_key) = self.api_key else {
            return Err(ClientBuilderError::ApiKeyNotSet);
        };

        let client = self.client.unwrap_or_default();

        #[cfg(feature = "leaky-bucket")]
        let leaky_bucket = self.leaky_bucket.map(Arc::new);

        Ok(Anthropic {
            api_key,
            version: self.version,
            client,
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorDetail {
    message: String,
    #[serde(default)]
    param: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

#[derive(Debug, Error)]
pub enum ApiRequestError {
    #[error("HTTP request error: {0}")]
    Request(#[from] reqwest::Error),

    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("EventSource error: {0}")]
    EventSource(#[from] reqwest_eventsource::Error),

    #[error("Invalid request: {message}")]
    InvalidRequest {
        message: String,
        param: Option<String>,
        code: Option<String>,
        status: reqwest::StatusCode,
    },

    #[error("Unexpected API response: {0}")]
    UnexpectedResponse(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Deserialization failed: {0}")]
    Deserialization(String),

    #[error("Unknown event type received: {0}")]
    UnknownEventType(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Serialization failed: {0}")]
    Serialization(String),
}
