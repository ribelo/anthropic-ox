pub mod message;
pub mod tools;

use std::fmt;

use reqwest_eventsource::{self, Event, EventSource, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_stream::{Stream, StreamExt};
use tools::ToolUse;

use crate::{ApiRequestError, Client, ErrorResponse, BASE_URL};

use self::{
    message::{Message, Messages, MultimodalContent, Role},
    tools::{Tool, Tools},
};

const API_URL: &str = "v1/messages";

#[derive(Debug, Clone, Serialize)]
pub struct MessagesRequest {
    pub messages: Messages,
    #[serde(skip_serializing_if = "Tools::is_empty")]
    pub tools: Tools,
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
    pub client: Client,
}

#[derive(Debug, Default)]
pub struct MessagesRequestBuilder {
    pub(crate) messages: Option<Messages>,
    pub(crate) tools: Option<Tools>,
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

    pub fn with_messages<T: Into<Messages>>(mut self, messages: T) -> Self {
        self.messages = Some(messages.into());
        self
    }

    pub fn add_message<T: Into<Message>>(mut self, message: T) -> Self {
        self.messages
            .get_or_insert_with(Messages::default)
            .add_message(message);
        self
    }

    pub fn with_tools(mut self, tools: Tools) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn add_tool(mut self, tool: impl Into<Tool>) -> Self {
        self.tools
            .get_or_insert_with(Tools::default)
            .add_tool(tool.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    pub fn enable_stream(mut self) -> Self {
        self.stream = Some(true);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    pub fn build(self) -> Result<MessagesRequest, MessagesRequestBuilderError> {
        Ok(MessagesRequest {
            messages: self
                .messages
                .ok_or(MessagesRequestBuilderError::MessagesNotSet)?,
            tools: self.tools.unwrap_or_default(),
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
            client: self
                .client
                .ok_or(MessagesRequestBuilderError::ClientNotSet)?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessagesResponse {
    pub id: String,
    pub r#type: String,
    pub role: Role,
    pub content: Vec<MultimodalContent>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

impl MessagesResponse {
    pub fn text_content(&self) -> Vec<&str> {
        self.content
            .iter()
            .filter_map(|content| {
                if let MultimodalContent::Text(text) = content {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
    pub fn tool_uses(&self) -> impl Iterator<Item = &ToolUse> {
        self.content.iter().filter_map(|content| {
            if let MultimodalContent::ToolUse(tool_use) = content {
                Some(tool_use)
            } else {
                None
            }
        })
    }

    pub fn tool_uses_mut(&mut self) -> impl Iterator<Item = &mut ToolUse> {
        self.content.iter_mut().filter_map(|content| {
            if let MultimodalContent::ToolUse(tool_use) = content {
                Some(tool_use)
            } else {
                None
            }
        })
    }
}

impl From<MessagesResponse> for Message {
    fn from(resp: MessagesResponse) -> Self {
        match resp.role {
            Role::User => Message::user(resp.content),
            Role::Assistant => Message::assistant(resp.content),
        }
    }
}

impl From<MessagesResponse> for Vec<MultimodalContent> {
    fn from(resp: MessagesResponse) -> Self {
        resp.content
    }
}

impl std::fmt::Display for MessagesResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MessagesResponse {{ id: {}, type: {}, role: {:?}, model: {}, content: [{}] }}",
            self.id,
            self.r#type,
            self.role,
            self.model,
            self.text_content().join(", ")
        )
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessagesStreamDelta {
    pub r#type: String,
    pub text: String,
}

impl fmt::Display for MessagesStreamDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContentBlockDelta {
    pub r#type: String,
    pub index: u32,
    pub delta: MessagesStreamDelta,
}

impl MessagesRequest {
    pub fn add_message<T: Into<Message>>(&mut self, message: T) {
        self.messages.add_message(message);
    }

    pub fn with_message<T: Into<Message>>(mut self, message: T) -> Self {
        self.messages.add_message(message);
        self
    }

    pub async fn send(&self) -> Result<MessagesResponse, ApiRequestError> {
        let url = format!("{}/{}", BASE_URL, API_URL);
        let body = serde_json::to_value(self)
            .map_err(|e| ApiRequestError::Serialization(e.to_string()))?;

        let response = self
            .client
            .client
            .post(&url)
            .header("x-api-key", &self.client.api_key)
            .header("anthropic-version", &self.client.version)
            .header("anthropic-beta", "tools-2024-04-04")
            .json(&body)
            .send()
            .await?;

        match response.status().as_u16() {
            200 | 201 => response
                .json()
                .await
                .map_err(|e| ApiRequestError::Deserialization(e.to_string())),
            429 => Err(ApiRequestError::RateLimit),
            _ => {
                let status = response.status();
                let error_response: ErrorResponse = response
                    .json()
                    .await
                    .map_err(|e| ApiRequestError::Deserialization(e.to_string()))?;
                Err(ApiRequestError::InvalidRequest {
                    message: error_response.error.message,
                    param: error_response.error.param,
                    code: error_response.error.code,
                    status,
                })
            }
        }
    }

    pub async fn stream(&self) -> impl Stream<Item = Result<MessagesStreamDelta, ApiRequestError>> {
        let url = format!("{}/{}", BASE_URL, API_URL);
        let mut body = serde_json::to_value(self)
            .map_err(|e| ApiRequestError::Serialization(e.to_string()))
            .expect("Failed to serialize request");
        body["stream"] = serde_json::Value::Bool(true);

        let es = self
            .client
            .client
            .post(url)
            .header("x-api-key", &self.client.api_key)
            .header("content-type", "application/json")
            .header("anthropic-version", &self.client.version)
            .header("anthropic-beta", "tools-2024-04-04")
            .json(&body)
            .eventsource()
            .expect("Failed to create EventSource");

        tokio_stream::wrappers::ReceiverStream::new(self.process_event_stream(es))
    }

    fn process_event_stream(
        &self,
        mut es: EventSource,
    ) -> tokio::sync::mpsc::Receiver<Result<MessagesStreamDelta, ApiRequestError>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        tokio::spawn(async move {
            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Open) => {}
                    Ok(Event::Message(msg)) => match msg.event.as_str() {
                        "message_start"
                        | "content_block_start"
                        | "ping"
                        | "content_block_stop"
                        | "message_delta" => {}
                        "message_stop" => break,
                        "content_block_delta" => {
                            let result = serde_json::from_str::<ContentBlockDelta>(&msg.data)
                                .map(|block| block.delta)
                                .map_err(|e| ApiRequestError::Deserialization(e.to_string()));

                            if tx.send(result).await.is_err() {
                                break;
                            }
                        }
                        other => {
                            if tx
                                .send(Err(ApiRequestError::UnknownEventType(other.to_string())))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                    },
                    Err(e) => {
                        if tx
                            .send(Err(ApiRequestError::Stream(e.to_string())))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        });

        rx
    }
}

#[cfg(test)]
mod test {
    use core::panic;
    use std::{
        collections::HashMap,
        sync::{atomic::AtomicBool, Arc},
    };

    use message::UserMessage;
    use schemars::JsonSchema;
    use serde_json::json;
    use test::message::{MultimodalContent, Text};
    use tools::{ToolContext, ToolError};

    use crate::ClientBuilder;

    use super::*;

    fn empty_handler(
        _input: serde_json::Value,
        _cx: &ToolContext,
    ) -> Result<serde_json::Value, ToolError> {
        Ok(json!({}))
    }

    #[test]
    fn test_messages_request_builder_messages() {
        let messages = Messages::from(Message::user(vec!["Hello"]));
        let builder = MessagesRequestBuilder::new().with_messages(messages.clone());
        assert_eq!(builder.messages, Some(messages));
    }

    #[test]
    fn test_messages_request_builder_add_message() {
        let message = Message::user(vec!["Hello"]);
        let builder = MessagesRequestBuilder::new().add_message(message.clone());
        assert_eq!(builder.messages, Some(Messages::from(message)));
    }

    #[test]
    fn test_messages_request_builder_tools() {
        let tools = Tools::from(vec![Tool::builder()
            .name("tool1")
            .handler(empty_handler)
            .build()
            .unwrap()]);
        let builder = MessagesRequestBuilder::new().with_tools(tools.clone());
        assert!(builder.tools.unwrap().get_tool("tool1").is_some());
    }

    #[test]
    fn test_messages_request_builder_add_tool() {
        let tool = Tool::builder()
            .name("tool1")
            .handler(empty_handler)
            .build()
            .unwrap();
        let builder = MessagesRequestBuilder::new().add_tool(tool);
        assert!(builder.tools.unwrap().get_tool("tool1").is_some());
    }

    #[test]
    fn test_messages_request_builder_model() {
        let model = "";
        let builder = MessagesRequestBuilder::new().with_model(model);
        assert_eq!(builder.model, Some(model.to_string()));
    }

    #[test]
    fn test_messages_request_builder_system() {
        let system = "You are a helpful assistant";
        let builder = MessagesRequestBuilder::new().with_system(system);
        assert_eq!(builder.system, Some(system.to_string()));
    }

    #[test]
    fn test_messages_request_builder_max_tokens() {
        let max_tokens = 100;
        let builder = MessagesRequestBuilder::new().with_max_tokens(max_tokens);
        assert_eq!(builder.max_tokens, Some(max_tokens));
    }

    #[test]
    fn test_messages_request_builder_stop_sequences() {
        let stop_sequences = vec!["stop1".to_string(), "stop2".to_string()];
        let builder = MessagesRequestBuilder::new().with_stop_sequences(stop_sequences.clone());
        assert_eq!(builder.stop_sequences, Some(stop_sequences));
    }

    #[test]
    fn test_messages_request_builder_stream() {
        let builder = MessagesRequestBuilder::new().enable_stream();
        assert_eq!(builder.stream, Some(true));
    }

    #[test]
    fn test_messages_request_builder_temperature() {
        let temperature = 0.5;
        let builder = MessagesRequestBuilder::new().with_temperature(temperature);
        assert_eq!(builder.temperature, Some(temperature));
    }

    #[test]
    fn test_messages_request_builder_top_p() {
        let top_p = 0.8;
        let builder = MessagesRequestBuilder::new().with_top_p(top_p);
        assert_eq!(builder.top_p, Some(top_p));
    }

    #[test]
    fn test_messages_request_builder_top_k() {
        let top_k = 50;
        let builder = MessagesRequestBuilder::new().with_top_k(top_k);
        assert_eq!(builder.top_k, Some(top_k));
    }

    #[test]
    fn test_messages_request_builder_client() {
        let client = ClientBuilder::default().api_key("api_key").build().unwrap();
        let builder = MessagesRequestBuilder::new().with_client(client.clone());
        assert!(builder.client.is_some());
    }

    #[test]
    fn test_messages_request_builder_build_success() {
        let messages = Messages::from("Hello");
        let model = "claude-3-sonnet-20240229";
        let max_tokens = 100;
        let client = ClientBuilder::default().api_key("api_key").build().unwrap();

        let request = MessagesRequestBuilder::new()
            .with_messages(messages.clone())
            .with_model(model)
            .with_max_tokens(max_tokens)
            .with_client(client.clone())
            .build()
            .unwrap();

        assert_eq!(request.messages, messages);
        assert_eq!(request.model, model.to_string());
        assert_eq!(request.max_tokens, max_tokens);
    }

    #[test]
    fn test_messages_request_builder_build_messages_not_set() {
        let model = "claude-3-sonnet-20240229";
        let max_tokens = 100;
        let client = ClientBuilder::default().api_key("api_key").build().unwrap();

        let result = MessagesRequestBuilder::new()
            .with_model(model)
            .with_max_tokens(max_tokens)
            .with_client(client)
            .build();

        assert!(matches!(
            result,
            Err(MessagesRequestBuilderError::MessagesNotSet)
        ));
    }

    #[test]
    fn test_messages_request_builder_build_model_not_set() {
        let messages = Messages::from(UserMessage::from("Hello"));
        let max_tokens = 100;
        let client = ClientBuilder::default().api_key("api_key").build().unwrap();

        let result = MessagesRequestBuilder::new()
            .with_messages(messages)
            .with_max_tokens(max_tokens)
            .with_client(client)
            .build();

        assert!(matches!(
            result,
            Err(MessagesRequestBuilderError::ModelNotSet)
        ));
    }

    #[test]
    fn test_messages_request_builder_build_max_tokens_not_set() {
        let messages = Messages::from(UserMessage::from("Hello"));
        let model = "claude-3-sonnet-20240229";
        let client = ClientBuilder::default().api_key("api_key").build().unwrap();

        let result = MessagesRequestBuilder::new()
            .with_messages(messages)
            .with_model(model)
            .with_client(client)
            .build();

        assert!(matches!(
            result,
            Err(MessagesRequestBuilderError::MaxTokensNotSet)
        ));
    }

    #[test]
    fn test_messages_request_builder_build_client_not_set() {
        let messages = Messages::from(UserMessage::from("Hello"));
        let model = "claude-3-sonnet-20240229";
        let max_tokens = 100;

        let result = MessagesRequestBuilder::new()
            .with_messages(messages)
            .with_model(model)
            .with_max_tokens(max_tokens)
            .build();

        assert!(matches!(
            result,
            Err(MessagesRequestBuilderError::ClientNotSet)
        ));
    }

    #[test]
    fn test_stop_reason_serialization() {
        let end_turn = StopReason::EndTurn;
        let max_tokens = StopReason::MaxTokens;
        let stop_sequence = StopReason::StopSequence;

        assert_eq!(serde_json::to_string(&end_turn).unwrap(), r#""end_turn""#);
        assert_eq!(
            serde_json::to_string(&max_tokens).unwrap(),
            r#""max_tokens""#
        );
        assert_eq!(
            serde_json::to_string(&stop_sequence).unwrap(),
            r#""stop_sequence""#
        );
    }

    #[test]
    fn test_stop_reason_deserialization() {
        let end_turn_json = r#""end_turn""#;
        let max_tokens_json = r#""max_tokens""#;
        let stop_sequence_json = r#""stop_sequence""#;

        assert_eq!(
            serde_json::from_str::<StopReason>(end_turn_json).unwrap(),
            StopReason::EndTurn
        );
        assert_eq!(
            serde_json::from_str::<StopReason>(max_tokens_json).unwrap(),
            StopReason::MaxTokens
        );
        assert_eq!(
            serde_json::from_str::<StopReason>(stop_sequence_json).unwrap(),
            StopReason::StopSequence
        );
    }

    #[test]
    fn test_usage_serialization() {
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 20,
        };

        assert_eq!(
            serde_json::to_string(&usage).unwrap(),
            r#"{"input_tokens":10,"output_tokens":20}"#
        );
    }

    #[test]
    fn test_usage_deserialization() {
        let usage_json = r#"{"input_tokens":10,"output_tokens":20}"#;
        let expected_usage = Usage {
            input_tokens: 10,
            output_tokens: 20,
        };

        assert_eq!(
            serde_json::from_str::<Usage>(usage_json).unwrap(),
            expected_usage
        );
    }

    #[test]
    fn test_messages_stream_delta_display() {
        let delta = MessagesStreamDelta {
            r#type: "type".to_string(),
            text: "hello".to_string(),
        };

        assert_eq!(format!("{}", delta), "hello");
    }

    #[test]
    fn test_content_block_delta_serialization() {
        let delta = ContentBlockDelta {
            r#type: "type".to_string(),
            index: 1,
            delta: MessagesStreamDelta {
                r#type: "type".to_string(),
                text: "hello".to_string(),
            },
        };

        let expected_json = r#"{"type":"type","index":1,"delta":{"type":"type","text":"hello"}}"#;
        assert_eq!(serde_json::to_string(&delta).unwrap(), expected_json);
    }

    #[test]
    fn test_content_block_delta_deserialization() {
        let delta_json = r#"{"type":"type","index":1,"delta":{"type":"type","text":"hello"}}"#;
        let expected_delta = ContentBlockDelta {
            r#type: "type".to_string(),
            index: 1,
            delta: MessagesStreamDelta {
                r#type: "type".to_string(),
                text: "hello".to_string(),
            },
        };

        assert_eq!(
            serde_json::from_str::<ContentBlockDelta>(delta_json).unwrap(),
            expected_delta
        );
    }

    #[test]
    fn test_messages_request_push_message() {
        let client = ClientBuilder::default().api_key("api_key").build().unwrap();
        let mut request = MessagesRequest {
            messages: Messages::default(),
            tools: Tools::new(),
            model: "model".to_string(),
            system: None,
            max_tokens: 10,
            stop_sequences: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            client,
        };

        request.add_message(UserMessage::from("Hello"));

        assert_eq!(request.messages.len(), 1);
    }

    #[test]
    fn test_messages_request_add_message() {
        let client = ClientBuilder::default().api_key("api_key").build().unwrap();
        let request = MessagesRequest {
            messages: Messages::default(),
            tools: Tools::new(),
            model: "model".to_string(),
            system: None,
            max_tokens: 10,
            stop_sequences: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            client,
        };

        let new_request = request.with_message(UserMessage::from("Hello"));

        assert_eq!(new_request.messages.len(), 1);
    }

    #[test]
    fn test_messages_response_deserialization() {
        let json = r#"{
            "id":"msg_015UCYG7heogFUS81jXr4z45",
            "type":"message",
            "role":"assistant",
            "model":"claude-3-sonnet-20240229",
            "stop_sequence":null,
            "usage":{"input_tokens":13,"output_tokens":32},
            "content":[{
                "type":"text",
                "text":"Hello John, it's nice to meet you! I'm Claude, an AI assistant created by Anthropic. How are you doing today?"
            }],
            "stop_reason":"end_turn"
        }"#;

        let response: MessagesResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.id, "msg_015UCYG7heogFUS81jXr4z45");
        assert_eq!(response.r#type, "message");
        assert_eq!(response.role, Role::Assistant);
        assert_eq!(response.model, "claude-3-sonnet-20240229");
        assert_eq!(response.stop_sequence, None);
        assert_eq!(response.usage.input_tokens, 13);
        assert_eq!(response.usage.output_tokens, 32);
        assert_eq!(response.content.len(), 1);
        assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    }

    #[test]
    fn test_messages_response_serialization() {
        let response = MessagesResponse {
        id: "msg_015UCYG7heogFUS81jXr4z45".to_string(),
        r#type: "message".to_string(),
        role: Role::Assistant,
        model: "claude-3-sonnet-20240229".to_string(),
        stop_sequence: None,
        usage: Usage {
            input_tokens: 13,
            output_tokens: 32,
        },
        content: vec![MultimodalContent::Text(Text {text:"Hello John, it's nice to meet you! I'm Claude, an AI assistant created by Anthropic. How are you doing today?".to_string() })],
        stop_reason: Some(StopReason::EndTurn),
    };

        let response_value = serde_json::to_value(response).unwrap();
        let expected_json = r#"{
        "id":"msg_015UCYG7heogFUS81jXr4z45",
        "type":"message",
        "role":"assistant",
        "model":"claude-3-sonnet-20240229",
        "stop_sequence":null,
        "usage":{"input_tokens":13,"output_tokens":32},
        "content":[{
            "type":"text",
            "text":"Hello John, it's nice to meet you! I'm Claude, an AI assistant created by Anthropic. How are you doing today?"
        }],
        "stop_reason":"end_turn"
    }"#;
        let expected_value: serde_json::Value = serde_json::from_str(expected_json).unwrap();

        assert_eq!(expected_value, response_value);
    }

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
            .with_model("claude-3-sonnet-20240229")
            .with_max_tokens(512)
            .with_messages(UserMessage::from("Hi, I'm John."))
            .build()
            .unwrap()
            .stream()
            .await;
        // dbg!(res);
        while let Some(res) = res.next().await {
            print!("{}", res.unwrap());
        }
    }
    #[tokio::test]
    async fn test_messages_request_success() {
        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
        struct TestHandlerProps {
            random_number: i32,
        }
        fn test_handler(
            _input: TestHandlerProps,
            _cx: &ToolContext,
        ) -> Result<serde_json::Value, ToolError> {
            Ok(json!("To finish this test write [finish_test]"))
        }

        fn finish_handler(
            _input: serde_json::Value,
            cx: &ToolContext,
        ) -> Result<serde_json::Value, ToolError> {
            let is_finished = cx.expect_resource::<Arc<AtomicBool>>();
            is_finished.store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(json!("Congratulations! You finished the test."))
        }

        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let anthropic = ClientBuilder::default()
            .api_key(api_key)
            .client(&client)
            .build()
            .unwrap();

        let test_tool = Tool::builder()
            .name("test_tool")
            .handler(test_handler)
            .build()
            .unwrap();

        let finish_tool = Tool::builder()
            .name("finish_test")
            .handler(finish_handler)
            .build()
            .unwrap();

        let is_finished = Arc::new(AtomicBool::new(false));
        let mut tools = Tools::from(vec![test_tool, finish_tool]);
        println!("{}", serde_json::to_string_pretty(&tools).unwrap());
        tools.add_resource(is_finished.clone());

        let mut messages = Messages::default();
        messages.add_message(UserMessage::from(
            "This is testing environent. To continue this test use [test_tool]",
        ));

        let mut i = 0;
        loop {
            if i > 3 {
                dbg!(messages);
                panic!("Test failed");
            }
            i += 1;

            let res = match anthropic
                .messages()
                .with_model("claude-3-haiku-20240307")
                .with_max_tokens(512)
                .with_messages(messages.clone())
                .with_tools(tools.clone())
                .build()
                .unwrap()
                .send()
                .await
            {
                Ok(res) => res,
                Err(err) => {
                    dbg!(messages);
                    panic!("{err}")
                }
            };

            messages.add_message(res.clone());

            let tools_used = res.tool_uses();
            let mut content = Vec::<MultimodalContent>::new();
            for tool in tools_used {
                if let Ok(result) = tools.invoke(tool.clone()) {
                    content.push(result.into());
                }
            }
            if !content.is_empty() {
                content.push("Here you have the result.".into());
                messages.add_message(Message::user(content));
            }
            if is_finished.load(std::sync::atomic::Ordering::Relaxed) {
                println!("Test passed");
                break;
            }
        }
    }
}
