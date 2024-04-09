use messages::MessagesRequestBuilder;
use serde::Deserialize;
use thiserror::Error;

pub mod messages;
pub mod tools;
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
pub enum AnthropicBuilderError {
    #[error("API key not set")]
    ApiKeyNotSet,
}

impl Anthropic {
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::default()
    }

    pub fn messages(&self) -> MessagesRequestBuilder {
        MessagesRequestBuilder {
            anthropic: Some(self.clone()),
            ..Default::default()
        }
    }
}

impl AnthropicBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn api_key(mut self, api_key: String) -> AnthropicBuilder {
        self.api_key = Some(api_key);
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

    pub fn build(self) -> Result<Anthropic, AnthropicBuilderError> {
        let Some(api_key) = self.api_key else {
            return Err(AnthropicBuilderError::ApiKeyNotSet);
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
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),
    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),
    #[error(transparent)]
    EventSourceError(#[from] reqwest_eventsource::Error),

    #[error("Invalid request error: {message}")]
    InvalidRequestError {
        message: String,
        param: Option<String>,
        code: Option<String>,
    },
    #[error("Unexpected response from API: {response}")]
    UnexpectedResponse { response: String },
}
