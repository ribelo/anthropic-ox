pub mod message;
pub mod tool;

use std::pin::Pin;

use futures::{Stream, StreamExt};
use message::Text;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tool::{Tool, ToolBox, ToolResult, ToolUse, ToolUseBuilder};

use crate::{Anthropic, ApiError, ApiErrorDetail, ErrorInfo, BASE_URL};

use self::message::{Message, Messages, MultimodalContent, Role};

const API_URL: &str = "v1/messages";

#[derive(Debug, Clone, Serialize)]
pub struct MessagesRequest {
    pub messages: Messages,
    #[serde(skip_serializing_if = "ToolBox::is_empty")]
    pub tools: ToolBox,
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
    pub anthropic: Anthropic,
}

#[derive(Debug, Default)]
pub struct MessagesRequestBuilder {
    pub(crate) messages: Option<Messages>,
    pub(crate) tools: Option<ToolBox>,
    pub(crate) model: Option<String>,
    pub(crate) system: Option<String>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) stop_sequences: Option<Vec<String>>,
    pub(crate) stream: Option<bool>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<i32>,
    pub(crate) client: Option<Anthropic>,
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

    pub fn with_message<T: Into<Message>>(mut self, message: T) -> Self {
        self.messages
            .get_or_insert_with(Messages::default)
            .add_message(message);
        self
    }

    pub fn with_tools(mut self, tools: ToolBox) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn add_tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.get_or_insert_with(ToolBox::default).add(tool);
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

    pub fn with_client(mut self, client: Anthropic) -> Self {
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
            anthropic: self
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

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
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

    pub fn tool_uses_owned(self) -> impl Iterator<Item = ToolUse> {
        self.content.into_iter().filter_map(|content| {
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

    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|content| matches!(content, MultimodalContent::ToolUse(_)))
    }

    pub async fn invoke_tools(&self, tools: &ToolBox) -> Vec<ToolResult> {
        let tool_uses: Vec<_> = self.tool_uses().cloned().collect();
        let mut results = Vec::with_capacity(tool_uses.len());

        for tool_use in tool_uses {
            let result = tools.invoke(tool_use).await;
            results.push(result);
        }

        results
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

#[derive(Debug, Default)]
pub struct MessagesResponseBuilder {
    id: Option<String>,
    r#type: Option<String>,
    role: Option<Role>,
    content: Vec<MultimodalContent>,
    model: Option<String>,
    stop_reason: Option<StopReason>,
    stop_sequence: Option<String>,
    usage: Usage,
    text_builder: Option<Text>,
    tool_use_builder: Option<ToolUseBuilder>,
}

#[derive(Debug, Error)]
pub enum MessagesResponseBuilderError {
    #[error("Missing id")]
    MissingId,
    #[error("Missing type")]
    MissingType,
    #[error("Missing role")]
    MissingRole,
    #[error("Missing model")]
    MissingModel,
}

impl MessagesResponseBuilder {
    pub fn build(self) -> Result<MessagesResponse, MessagesResponseBuilderError> {
        let id = self.id.ok_or(MessagesResponseBuilderError::MissingId)?;
        let r#type = self
            .r#type
            .ok_or(MessagesResponseBuilderError::MissingType)?;
        let role = self.role.ok_or(MessagesResponseBuilderError::MissingRole)?;
        let model = self
            .model
            .ok_or(MessagesResponseBuilderError::MissingModel)?;

        Ok(MessagesResponse {
            id,
            r#type,
            role,
            content: self.content,
            model,
            stop_reason: self.stop_reason,
            stop_sequence: self.stop_sequence,
            usage: self.usage,
        })
    }

    pub fn push_event(&mut self, event: EventData) {
        match event {
            EventData::MessageStart { message } => {
                self.id = Some(message.id);
                self.r#type = Some(message.r#type);
                self.role = Some(message.role);
                self.model = Some(message.model);
                self.content = message.content;
                self.stop_reason = message.stop_reason;
                self.stop_sequence = message.stop_sequence;
                self.usage = message.usage;
            }
            EventData::ContentBlockStart { content_block, .. } => match content_block {
                ContentBlock::Text { .. } => {
                    self.text_builder.replace(Text::default());
                }
                ContentBlock::ToolUse { id, name, .. } => {
                    self.tool_use_builder.replace(ToolUseBuilder::new(id, name));
                }
            },
            EventData::ContentBlockDelta { delta, .. } => {
                match &delta {
                    ContentBlockDelta::TextDelta { text } => {
                        if let Some(x) = self.text_builder.as_mut() {
                            x.push_str(text);
                        }
                    }
                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                        if let Some(x) = self.tool_use_builder.as_mut() {
                            x.push_str(partial_json);
                        }
                    }
                };
            }
            EventData::ContentBlockStop { .. } => {
                if let Some(text_builder) = self.text_builder.take() {
                    let content = MultimodalContent::Text(text_builder);
                    self.content.push(content);
                } else if let Some(tool_use_builder) = self.tool_use_builder.take() {
                    let tool_use = tool_use_builder.build().unwrap();
                    let content = MultimodalContent::ToolUse(tool_use);
                    self.content.push(content);
                } else {
                    unreachable!()
                }
            }
            EventData::MessageDelta { usage, .. } => {
                if let Some(usage) = usage {
                    self.usage.input_tokens = usage.input_tokens.or(self.usage.input_tokens);
                    self.usage.output_tokens = usage.output_tokens.or(self.usage.output_tokens);
                }
            }
            EventData::MessageStop | EventData::Ping => {}
            // TODO:
            EventData::Error { .. } => {}
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamMessage {
    pub id: String,
    pub r#type: String,
    pub role: Role,
    pub content: Vec<MultimodalContent>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventData {
    MessageStart {
        message: StreamMessage,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: Option<Usage>,
    },
    MessageStop,
    Ping,
    Error {
        error: ErrorInfo,
    },
}

pub type EventStream = Pin<Box<dyn Stream<Item = Result<EventData, ApiError>> + Send>>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ApiErrorDetail,
}

impl From<ErrorResponse> for ApiError {
    fn from(response: ErrorResponse) -> Self {
        ApiError::InvalidRequest {
            message: response.error.message,
            param: response.error.param,
            code: response.error.code,
        }
    }
}

impl MessagesRequest {
    pub fn add_message<T: Into<Message>>(&mut self, message: T) {
        self.messages.add_message(message);
    }

    pub fn with_message<T: Into<Message>>(mut self, message: T) -> Self {
        self.messages.add_message(message);
        self
    }

    pub async fn send(&self) -> Result<MessagesResponse, ApiError> {
        let url = format!("{}/{}", BASE_URL, API_URL);
        let body =
            serde_json::to_value(self).map_err(|e| ApiError::Serialization(e.to_string()))?;

        let response = self
            .anthropic
            .client
            .post(&url)
            .header("x-api-key", &self.anthropic.api_key)
            .header("anthropic-version", &self.anthropic.version)
            .header("anthropic-beta", "tools-2024-04-04")
            .json(&body)
            .send()
            .await?;

        match response.status().as_u16() {
            200 | 201 => response
                .json()
                .await
                .map_err(|e| ApiError::Deserialization(e.to_string())),
            429 => Err(ApiError::RateLimit),
            _ => {
                let error_response: ErrorResponse = response
                    .json()
                    .await
                    .map_err(|e| ApiError::Deserialization(e.to_string()))?;
                Err(ApiError::InvalidRequest {
                    message: error_response.error.message,
                    param: error_response.error.param,
                    code: error_response.error.code,
                })
            }
        }
    }

    pub async fn stream(&self) -> impl Stream<Item = Result<EventData, ApiError>> {
        let url = format!("{BASE_URL}/{API_URL}");
        let mut body = serde_json::to_value(self).expect("Failed to serialize request");
        body["stream"] = serde_json::Value::Bool(true);

        let stream = self
            .anthropic
            .client
            .post(url)
            .header("x-api-key", &self.anthropic.api_key)
            .header("content-type", "application/json")
            .header("anthropic-version", &self.anthropic.version)
            .json(&body)
            .send()
            .await
            .unwrap()
            .bytes_stream();

        let filtered_stream = stream.filter_map(|chunk| async move {
            match chunk {
                Ok(bytes) => {
                    let data = String::from_utf8(bytes.to_vec()).unwrap();
                    match data.as_str() {
                        "" => None,
                        s if s.starts_with("data: ") => {
                            let json_data = s.trim_start_matches("data: ");
                            Some(
                                serde_json::from_str::<EventData>(json_data)
                                    .map_err(|e| ApiError::Deserialization(e.to_string())),
                            )
                        }
                        _ => Some(Err(ApiError::InvalidEventData(data))),
                    }
                }
                Err(e) => Some(Err(ApiError::Stream(e.to_string()))),
            }
        });

        Box::pin(filtered_stream)
    }
}

#[cfg(test)]
mod test {
    use core::panic;
    use std::sync::{atomic::AtomicBool, Arc};

    use async_trait::async_trait;
    use message::UserMessage;
    use schemars::JsonSchema;
    use serde_json::{json, Value};
    use tap::{self, Pipe, Tap};
    use test::message::{MultimodalContent, Text};

    use crate::AnthropicBuilder;

    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    #[cfg(not(target_arch = "wasm32"))]
    fn get_api_key() -> String {
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set")
    }

    #[cfg(target_arch = "wasm32")]
    fn get_api_key() -> String {
        std::env!("ANTHROPIC_API_KEY").to_string()
    }

    #[derive(Clone)]
    pub struct EmptyTool;

    #[async_trait]
    impl Tool for EmptyTool {
        type Input = ();

        type Output = ();

        type Error = String;

        fn name(&self) -> String {
            "empty_tool".to_string()
        }

        async fn invoke(&self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
            dbg!("EmptyTool");
            Ok(())
        }
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_messages() {
        let messages = Messages::from(Message::user(vec!["Hello"]));
        let builder = MessagesRequestBuilder::new().with_messages(messages.clone());
        assert_eq!(builder.messages, Some(messages));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_add_message() {
        let message = Message::user(vec!["Hello"]);
        let builder = MessagesRequestBuilder::new().with_message(message.clone());
        assert_eq!(builder.messages, Some(Messages::from(message)));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_tools() {
        let tools = ToolBox::from(vec![EmptyTool]);
        let builder = MessagesRequestBuilder::new().with_tools(tools.clone());
        assert!(builder.tools.unwrap().get("empty_tool").is_some());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_add_tool() {
        let builder = MessagesRequestBuilder::new().add_tool(EmptyTool);
        assert!(builder.tools.unwrap().get("empty_tool").is_some());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_model() {
        let model = "";
        let builder = MessagesRequestBuilder::new().with_model(model);
        assert_eq!(builder.model, Some(model.to_string()));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_system() {
        let system = "You are a helpful assistant";
        let builder = MessagesRequestBuilder::new().with_system(system);
        assert_eq!(builder.system, Some(system.to_string()));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_max_tokens() {
        let max_tokens = 100;
        let builder = MessagesRequestBuilder::new().with_max_tokens(max_tokens);
        assert_eq!(builder.max_tokens, Some(max_tokens));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_stop_sequences() {
        let stop_sequences = vec!["stop1".to_string(), "stop2".to_string()];
        let builder = MessagesRequestBuilder::new().with_stop_sequences(stop_sequences.clone());
        assert_eq!(builder.stop_sequences, Some(stop_sequences));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_stream() {
        let builder = MessagesRequestBuilder::new().enable_stream();
        assert_eq!(builder.stream, Some(true));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_temperature() {
        let temperature = 0.5;
        let builder = MessagesRequestBuilder::new().with_temperature(temperature);
        assert_eq!(builder.temperature, Some(temperature));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_top_p() {
        let top_p = 0.8;
        let builder = MessagesRequestBuilder::new().with_top_p(top_p);
        assert_eq!(builder.top_p, Some(top_p));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_top_k() {
        let top_k = 50;
        let builder = MessagesRequestBuilder::new().with_top_k(top_k);
        assert_eq!(builder.top_k, Some(top_k));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_client() {
        let client = AnthropicBuilder::default()
            .with_api_key("api_key")
            .build()
            .unwrap();
        let builder = MessagesRequestBuilder::new().with_client(client.clone());
        assert!(builder.client.is_some());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_build_success() {
        let messages = Messages::from(vec!["Hello"]);
        let model = "claude-3-sonnet-20240229";
        let max_tokens = 100;
        let client = AnthropicBuilder::default()
            .with_api_key("api_key")
            .build()
            .unwrap();

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
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_build_messages_not_set() {
        let model = "claude-3-sonnet-20240229";
        let max_tokens = 100;
        let client = AnthropicBuilder::default()
            .with_api_key("api_key")
            .build()
            .unwrap();

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
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_build_model_not_set() {
        let messages = Messages::from(vec![UserMessage::from("Hello")]);
        let max_tokens = 100;
        let client = AnthropicBuilder::default()
            .with_api_key("api_key")
            .build()
            .unwrap();

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
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_build_max_tokens_not_set() {
        let messages = Messages::from(vec![UserMessage::from("Hello")]);
        let model = "claude-3-sonnet-20240229";
        let client = AnthropicBuilder::default()
            .with_api_key("api_key")
            .build()
            .unwrap();

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
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_builder_build_client_not_set() {
        let messages = Messages::from(vec![UserMessage::from("Hello")]);
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
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_usage_deserialization() {
        let usage_json = r#"{"input_tokens":10,"output_tokens":20}"#;
        let expected_usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
        };

        assert_eq!(
            serde_json::from_str::<Usage>(usage_json).unwrap(),
            expected_usage
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_push_message() {
        let client = AnthropicBuilder::default()
            .with_api_key("api_key")
            .build()
            .unwrap();
        let mut request = MessagesRequest {
            messages: Messages::default(),
            tools: ToolBox::default(),
            model: "model".to_string(),
            system: None,
            max_tokens: 10,
            stop_sequences: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            anthropic: client,
        };

        request.add_message(UserMessage::from("Hello"));

        assert_eq!(request.messages.len(), 1);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_request_add_message() {
        let client = AnthropicBuilder::default()
            .with_api_key("api_key")
            .build()
            .unwrap();
        let request = MessagesRequest {
            messages: Messages::default(),
            tools: ToolBox::default(),
            model: "model".to_string(),
            system: None,
            max_tokens: 10,
            stop_sequences: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            anthropic: client,
        };

        let new_request = request.with_message(UserMessage::from("Hello"));

        assert_eq!(new_request.messages.len(), 1);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
        assert_eq!(response.usage.input_tokens, Some(13));
        assert_eq!(response.usage.output_tokens, Some(32));
        assert_eq!(response.content.len(), 1);
        assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_messages_response_serialization() {
        let response = MessagesResponse {
        id: "msg_015UCYG7heogFUS81jXr4z45".to_string(),
        r#type: "message".to_string(),
        role: Role::Assistant,
        model: "claude-3-sonnet-20240229".to_string(),
        stop_sequence: None,
        usage: Usage {
            input_tokens: Some(13),
            output_tokens: Some(32),
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

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_messages_stream_request() {
        let api_key = get_api_key();
        let client = reqwest::Client::new();
        let anthropic = AnthropicBuilder::default()
            .with_api_key(api_key)
            .with_client(&client)
            .build()
            .unwrap();
        let mut res = anthropic
            .messages()
            .with_model("claude-3-sonnet-20240229")
            .with_max_tokens(512)
            .with_message(UserMessage::from("Hi, I'm John."))
            .build()
            .unwrap()
            .stream()
            .await;

        // dbg!(res);
        while let Some(res) = res.next().await {
            dbg!(res);
        }
    }
    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_messages_request_success() {
        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
        struct TestHandlerProps {
            random_number: i32,
        }
        #[derive(Clone)]
        struct TestTool;
        #[async_trait]
        impl Tool for TestTool {
            type Input = TestHandlerProps;

            type Output = Value;

            type Error = String;

            fn name(&self) -> String {
                "test_tool".to_string()
            }

            async fn invoke(&self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
                Ok(json!("To finish this test write [finish_test]"))
            }
        }

        #[derive(Default, Clone)]
        struct FinishTool {
            is_finished: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Tool for FinishTool {
            type Input = Value;

            type Output = Value;

            type Error = String;

            fn name(&self) -> String {
                "finish_tool".to_string()
            }

            async fn invoke(&self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
                self.is_finished
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(json!("Congratulations! You finished the test."))
            }
        }

        let api_key = get_api_key();
        console_log!("api_key: {api_key}");
        let client = reqwest::Client::new();
        let anthropic = AnthropicBuilder::default()
            .with_api_key(api_key)
            .with_client(&client)
            .build()
            .unwrap();

        let tools = ToolBox::default();
        tools.add(TestTool);
        let finish_tool = FinishTool::default();
        tools.add(finish_tool.clone());

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
                let result = tools.invoke(tool.clone()).await;
                content.push(result.into());
            }
            if !content.is_empty() {
                content.push("Here you have the result.".into());
                messages.add_message(Message::user(content));
            }
            dbg!(&messages);
            if finish_tool
                .is_finished
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                console_log!("Test passed");
                break;
            }
        }
    }
}
