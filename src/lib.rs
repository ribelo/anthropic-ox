use messages::MessagesRequestBuilder;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod agent;
pub mod messages;

const BASE_URL: &str = "https://api.anthropic.com";

#[derive(Debug, PartialEq, PartialOrd, strum::EnumString, strum::Display)]
pub enum Model {
    #[strum(to_string = "claude-3-haiku-20240307")]
    Claude3Haiku,
    #[strum(to_string = "claude-3-sonnet-20240229")]
    Claude3Sonnet,
    #[strum(to_string = "claude-3-opus-20240229")]
    Claude3Opus,
    #[strum(to_string = "claude-3-5-sonnet-20240620")]
    Claude35Sonnet,
}

#[cfg(feature = "leaky-bucket")]
use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;

pub struct AnthropicBuilder {
    api_key: Option<String>,
    version: String,
    client: Option<reqwest::Client>,
    #[cfg(feature = "leaky-bucket")]
    leaky_bucket: Option<RateLimiter>,
}

impl Default for AnthropicBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            version: "2023-06-01".to_string(),
            client: None,
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket: None,
        }
    }
}

#[derive(Clone)]
pub struct Anthropic {
    api_key: String,
    version: String,
    client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    leaky_bucket: Option<Arc<RateLimiter>>,
}

impl std::fmt::Debug for Anthropic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Anthropic")
            .field("api_key", &"[REDACTED]")
            .field("version", &self.version)
            .field("client", &self.client)
            .finish()
    }
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

    pub fn with_api_key<T: ToString>(mut self, api_key: T) -> AnthropicBuilder {
        self.api_key = Some(api_key.to_string());
        self
    }

    pub fn with_client(mut self, client: &reqwest::Client) -> AnthropicBuilder {
        self.client = Some(client.clone());
        self
    }

    #[cfg(feature = "leaky-bucket")]
    pub fn with_limiter(mut self, leaky_bucket: RateLimiter) -> AnthropicBuilder {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorInfo {
    pub r#type: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorDetail {
    message: String,
    #[serde(default)]
    param: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    r#type: String,
    error: ApiErrorDetail,
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Unknown event type: {0}")]
    UnknownEventType(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Invalid request: {message}")]
    InvalidRequest {
        message: String,
        param: Option<String>,
        code: Option<String>,
    },

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("API error: {0}")]
    Generic(String),

    #[error("API overloaded: {0}")]
    Overloaded(String),

    #[error("HTTP request error: {0}")]
    Request(#[from] reqwest::Error),

    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Unexpected API response: {0}")]
    UnexpectedResponse(String),

    #[error("Serialization failed: {0}")]
    Serialization(String),

    #[error("Invalid event data: {0}")]
    InvalidEventData(String),
}

impl From<ErrorInfo> for ApiError {
    fn from(error: ErrorInfo) -> Self {
        match error.r#type.as_str() {
            "invalid_request_error" => ApiError::InvalidRequest {
                message: error.message,
                param: None,
                code: None,
            },
            "authentication_error" => ApiError::Authentication(error.message),
            "permission_error" => ApiError::PermissionDenied(error.message),
            "not_found_error" => ApiError::NotFound(error.message),
            "rate_limit_error" => ApiError::RateLimit,
            "api_error" => ApiError::Generic(error.message),
            "overloaded_error" => ApiError::Overloaded(error.message),
            _ => ApiError::UnexpectedResponse(format!("Unknown error type: {}", error.r#type)),
        }
    }
}
