pub mod message;
pub mod tools;

use std::{borrow::Cow, fmt, ops::Deref, sync::Arc};

use derivative::Derivative;
use reqwest_eventsource::{self, Event, RequestBuilderExt};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio_stream::{wrappers::LinesStream, Stream, StreamExt};

use crate::{ApiRequestError, Client, ErrorResponse, BASE_URL};

use self::message::{Message, Messages};

const API_URL: &str = "v1/messages";

#[derive(Debug, Clone, Serialize)]
pub struct MessagesRequest {
    pub messages: Messages,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip)]
    pub anthropic: Client,
}

#[derive(Debug, Default)]
pub struct MessagesRequestBuilder {
    pub(crate) messages: Option<Messages>,
    pub(crate) model: Option<String>,
    pub(crate) system: Option<String>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) stop_sequences: Option<Vec<String>>,
    pub(crate) stream: Option<bool>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<i32>,
    pub(crate) client: Option<Client>,
}

#[derive(Debug, Error)]
pub enum MessagesRequestBuilderError {
    #[error("Messages not set")]
    MessagesNotSet,
    #[error("Model not set")]
    ModelNotSet,
    #[error("Max tokens not set")]
    MaxTokensNotSet,
    #[error("Client not set")]
    ClientNotSet,
}

impl MessagesRequestBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn messages<T: Into<Messages>>(mut self, messages: T) -> Self {
        self.messages = Some(messages.into());
        self
    }

    pub fn add_message<T: Into<Message>>(mut self, message: T) -> Self {
        if let Some(ref mut messages) = self.messages {
            messages.push_message(message);
        } else {
            self.messages = Some(Messages::from(message.into()));
        }
        self
    }

    pub fn model<T: AsRef<str>>(mut self, model: T) -> Self {
        self.model = Some(model.as_ref().to_string());
        self
    }

    pub fn system<T: AsRef<str>>(mut self, system: T) -> Self {
        self.system = Some(system.as_ref().to_string());
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    pub fn stream(mut self) -> Self {
        self.stream = Some(true);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn anthropic(mut self, anthropic: Client) -> Self {
        self.client = Some(anthropic);
        self
    }

    pub fn build(self) -> Result<MessagesRequest, MessagesRequestBuilderError> {
        Ok(MessagesRequest {
            messages: self
                .messages
                .ok_or(MessagesRequestBuilderError::MessagesNotSet)?,
            model: self.model.ok_or(MessagesRequestBuilderError::ModelNotSet)?,
            system: self.system,
            max_tokens: self
                .max_tokens
                .ok_or(MessagesRequestBuilderError::MaxTokensNotSet)?,
            stop_sequences: self.stop_sequences,
            stream: self.stream,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            anthropic: self
                .client
                .ok_or(MessagesRequestBuilderError::ClientNotSet)?,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Content {
    pub r#type: String,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<Content>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MessagesStreamDelta {
    pub r#type: String,
    pub text: String,
}

impl fmt::Display for MessagesStreamDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContentBlockDelta {
    pub r#type: String,
    pub index: u32,
    pub delta: MessagesStreamDelta,
}

impl MessagesRequest {
    pub fn push_message<T: Into<Message>>(&mut self, message: T) {
        self.messages.push_message(message);
    }
    pub async fn send(&self) -> Result<MessagesResponse, ApiRequestError> {
        let url = format!("{}/{}", BASE_URL, API_URL);
        let body = serde_json::to_value(self).unwrap();
        let req = self
            .anthropic
            .client
            .post(&url)
            .header("x-api-key", &self.anthropic.api_key)
            .header("anthropic-version", &self.anthropic.version)
            .json(&body);
        let res = req.send().await?;
        if res.status().is_success() {
            let data: MessagesResponse = res.json().await?;
            Ok(data)
        } else {
            let error_response: ErrorResponse = res.json().await?;
            Err(ApiRequestError::InvalidRequestError {
                message: error_response.error.message,
                param: error_response.error.param,
                code: error_response.error.code,
            })
        }
    }

    pub async fn stream(&self) -> impl Stream<Item = MessagesStreamDelta> {
        let url = format!("{}/{}", BASE_URL, API_URL);
        let mut body = serde_json::to_value(self).unwrap();
        body["stream"] = serde_json::Value::Bool(true);
        let mut es = self
            .anthropic
            .client
            .post(url)
            .header("x-api-key", &self.anthropic.api_key)
            .header("content-type", "application/json")
            .header("anthropic-version", &self.anthropic.version)
            .header("anthropic-beta", "messages-2023-12-15")
            .json(&body)
            .eventsource()
            .unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Message(msg)) => match msg.event.as_str() {
                        "message_start" => {}
                        "message_stop" => {
                            es.close();
                            break;
                        }
                        "content_block_start" => {}
                        "ping" => {}
                        "content_block_delta" => {
                            if let Ok(block) = serde_json::from_str::<ContentBlockDelta>(&msg.data)
                            {
                                tx.send(block.delta).unwrap();
                            } else {
                                es.close();
                                eprintln!("error: {:?}", msg.data);
                                break;
                            }
                        }
                        "content_block_stop" => {}
                        "message_delta" => {}
                        other => {
                            eprintln!("Nieznany typ zdarzenia: {}", other);
                            es.close();
                            break;
                        }
                    },
                    Err(err) => {
                        eprintln!("err: {:#?}", err);
                        break;
                    }
                    _ => {}
                }
            }
        });
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }
}

#[cfg(test)]
mod test {
    use tokio_stream::StreamExt;

    use crate::{
        messages::{message::Messages, Message},
        ClientBuilder,
    };

    #[tokio::test]
    async fn test_messages_stream_request_builder() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let anthropic = ClientBuilder::default()
            .api_key(api_key)
            .client(&client)
            .build()
            .unwrap();
        let mut res = anthropic
            .messages()
            .model("claude-3-sonnet-20240229")
            .max_tokens(512)
            .messages(Message::user(String::from("Hi, I'm John.")))
            .build()
            .unwrap()
            .stream()
            .await;
        // dbg!(res);
        while let Some(res) = res.next().await {
            print!("{}", res);
        }
    }
    #[tokio::test]
    async fn test_messages_request_builder() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let anthropic = ClientBuilder::default()
            .api_key(api_key)
            .client(&client)
            .build()
            .unwrap();
        let mut res = anthropic
            .messages()
            .model("claude-3-sonnet-20240229")
            .max_tokens(512)
            .messages(Message::user(String::from("Hi, I'm John.")))
            .build()
            .unwrap()
            .send()
            .await;
        dbg!(res);
    }
}
