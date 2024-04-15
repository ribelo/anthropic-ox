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
pub struct ClientBuilder {
    api_key: Option<String>,
    #[derivative(Default(value = r#""2023-06-01".to_string()"#))]
    version: String,
    client: Option<reqwest::Client>,
    #[cfg(feature = "leaky-bucket")]
    leaky_bucket: Option<RateLimiter>,
}

#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct Client {
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

impl Client {
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    pub fn messages(&self) -> MessagesRequestBuilder {
        MessagesRequestBuilder {
            client: Some(self.clone()),
            ..Default::default()
        }
    }
}

impl ClientBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn api_key<T: ToString>(mut self, api_key: T) -> ClientBuilder {
        self.api_key = Some(api_key.to_string());
        self
    }

    pub fn client(mut self, client: &reqwest::Client) -> ClientBuilder {
        self.client = Some(client.clone());
        self
    }

    #[cfg(feature = "leaky-bucket")]
    pub fn limiter(mut self, leaky_bucket: RateLimiter) -> ClientBuilder {
        self.leaky_bucket = Some(leaky_bucket);
        self
    }

    pub fn build(self) -> Result<Client, ClientBuilderError> {
        let Some(api_key) = self.api_key else {
            return Err(ClientBuilderError::ApiKeyNotSet);
        };

        let client = self.client.unwrap_or_default();

        #[cfg(feature = "leaky-bucket")]
        let leaky_bucket = self.leaky_bucket.map(Arc::new);

        Ok(Client {
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
