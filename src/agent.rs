use std::{pin::Pin, str::FromStr};

use async_recursion::async_recursion;
use rustc_hash::FxHashMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_stream::{Stream, StreamExt};

use crate::{
    messages::{
        message::{Messages, MultimodalContent, Text},
        tools::{Tool, ToolContext, ToolError, ToolResult, ToolUseBuilder, Tools},
        ContentBlock, ContentBlockDelta, EventData, EventStream, MessagesRequest, MessagesResponse,
        MessagesResponseBuilder, MessagesResponseBuilderError,
    },
    Anthropic, ApiError, Model,
};

type ChatEventStream = Pin<Box<dyn Stream<Item = Result<ChatEventMessage, ApiError>> + Send>>;
type ChatContentStream = Pin<Box<dyn Stream<Item = Result<ChatContentMessage, ApiError>> + Send>>;
type ChatMessageSender = tokio::sync::mpsc::UnboundedSender<Result<ChatEventMessage, ApiError>>;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub enum Participant {
    #[default]
    System,
    Agent(String),
}

impl From<String> for Participant {
    fn from(value: String) -> Self {
        Participant::Agent(value)
    }
}

#[derive(Debug, Clone)]
pub enum ChatEventMessage {
    Request {
        from: Participant,
        to: Participant,
        content: MultimodalContent,
    },
    Response {
        from: Participant,
        to: Participant,
        event: EventData,
    },
}

#[derive(Debug, Clone)]
pub enum ChatContentMessage {
    Request {
        from: Participant,
        to: Participant,
        content: MultimodalContent,
    },
    Response {
        from: Participant,
        to: Participant,
        content: MultimodalContent,
    },
}

pub struct ChatContentMessageBuilder {
    text_builder: Option<Text>,
    tool_use_builder: Option<ToolUseBuilder>,
    on_message: Box<dyn Fn(ChatContentMessage) + Send>,
}

impl ChatContentMessageBuilder {
    pub fn new<F>(on_message: F) -> Self
    where
        F: Fn(ChatContentMessage) + Send + 'static,
    {
        Self {
            text_builder: None,
            tool_use_builder: None,
            on_message: Box::new(on_message),
        }
    }

    pub fn push_event(&mut self, from: Participant, to: Participant, event: EventData) {
        match event {
            EventData::ContentBlockStart { content_block, .. } => match content_block {
                ContentBlock::Text { .. } => {
                    self.text_builder = Some(Text::default());
                }
                ContentBlock::ToolUse { id, name, .. } => {
                    self.tool_use_builder = Some(ToolUseBuilder::new(id, name));
                }
            },
            EventData::ContentBlockDelta { delta, .. } => match delta {
                ContentBlockDelta::TextDelta { text } => {
                    if let Some(builder) = self.text_builder.as_mut() {
                        builder.push_str(&text);
                    }
                }
                ContentBlockDelta::InputJsonDelta { partial_json } => {
                    if let Some(builder) = self.tool_use_builder.as_mut() {
                        builder.push_str(&partial_json);
                    }
                }
            },
            EventData::ContentBlockStop { .. } => {
                let message = match (self.text_builder.take(), self.tool_use_builder.take()) {
                    (Some(text_builder), None) => {
                        let content = MultimodalContent::Text(text_builder);
                        ChatContentMessage::Response {
                            from: to,
                            to: from,
                            content,
                        }
                    }
                    (None, Some(tool_use_builder)) => {
                        let tool_use = tool_use_builder.build().expect("Failed to build tool use");
                        let content = MultimodalContent::ToolUse(tool_use);
                        ChatContentMessage::Response {
                            from: to,
                            to: from,
                            content,
                        }
                    }
                    _ => unreachable!("Either text_builder or tool_use_builder must be Some"),
                };
                (self.on_message)(message);
            }
            _ => {} // Ignore other event types
        }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct AgentInput {
    #[schemars(
        description = "The message content for agent communication. This field contains the actual text or data that agents use to interact and communicate with each other."
    )]
    message: String,
}

#[derive(Debug, Default)]
pub struct AgentBuilder {
    pub name: Option<String>,
    pub description: Option<String>,
    pub model: Option<String>,
    pub system_message: Option<String>,
    pub max_tokens: Option<u32>,
    subagents: Agents,
    tools: Tools,
    cx: ToolContext,
    anthropic: Option<Anthropic>,
}

#[derive(Debug, thiserror::Error)]
pub enum AgentBuilderError {
    #[error("Agent name is required")]
    MissingName,
    #[error("Agent description is required")]
    MissingDescription,
    #[error("Model is required")]
    MissingModel,
    #[error("System message is required")]
    MissingSystemMessage,
    #[error("Client is required")]
    MissingAnthropicClient,
}

impl AgentBuilder {
    #[must_use]
    pub fn with_client(mut self, anthropic: Anthropic) -> Self {
        self.anthropic = Some(anthropic);
        self
    }

    #[must_use]
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    #[must_use]
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_system_message<S: Into<String>>(mut self, system_message: S) -> Self {
        self.system_message = Some(system_message.into());
        self
    }

    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_subagent<T: Into<Agent>>(mut self, agent: T) -> Self {
        self.subagents.add(agent.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn with_tool<T: Into<Tool>>(mut self, tool: T) -> Self {
        self.tools.add(tool.into());
        self
    }

    pub fn with_context(mut self, cx: ToolContext) -> Self {
        self.cx = cx;
        self
    }

    pub fn build(self) -> Result<Agent, AgentBuilderError> {
        let name = self.name.ok_or(AgentBuilderError::MissingName)?;
        let description = self
            .description
            .ok_or(AgentBuilderError::MissingDescription)?;
        let model = self.model.ok_or(AgentBuilderError::MissingModel)?;
        let system_message = self
            .system_message
            .ok_or(AgentBuilderError::MissingSystemMessage)?;
        let max_tokens = self.max_tokens.unwrap_or(4096);

        let anthropic = self
            .anthropic
            .ok_or(AgentBuilderError::MissingAnthropicClient)?;

        Ok(Agent {
            name,
            description,
            model,
            system_message,
            max_tokens,
            subagents: self.subagents,
            tools: self.tools,
            cx: self.cx,
            anthropic,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub name: String,
    pub description: String,
    pub model: String,
    pub system_message: String,
    pub max_tokens: u32,
    subagents: Agents,
    tools: Tools,
    cx: ToolContext,
    anthropic: Anthropic,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    ApiError(#[from] ApiError),
    #[error(transparent)]
    ToolError(#[from] ToolError),
    #[error(transparent)]
    MessagesResponseBuilderError(#[from] MessagesResponseBuilderError),
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    fn get_tools(&self) -> Tools {
        let mut tools = self.tools.clone();
        tools.extend(self.subagents.as_tools());
        tools
    }

    fn prepare_request(&self, messages: Messages) -> MessagesRequest {
        self.anthropic
            .messages()
            .with_max_tokens(self.max_tokens)
            .with_model(self.model.clone())
            .with_system(self.system_message.clone())
            .with_messages(messages)
            .with_tools(self.get_tools())
            .build()
            .expect("Valid agent")
    }

    #[async_recursion]
    async fn process_tools_invoke(
        &self,
        response: MessagesResponse,
        tx: Option<ChatMessageSender>,
    ) -> Vec<Result<ToolResult, AgentError>> {
        let mut join_set = tokio::task::JoinSet::new();

        for tool_use in response.tool_uses_owned() {
            let tx = tx.clone();
            let agent = self.clone();
            join_set.spawn(async move {
                if let Some(subagent) = agent.subagents.get(&tool_use.name) {
                    let agent_input =
                        serde_json::from_value::<AgentInput>(tool_use.input.clone()).unwrap();
                    let result = subagent
                        .execute(agent.name.clone(), vec![agent_input.message], tx)
                        .await;
                    match result {
                        Err(e) => Err(e),
                        Ok(messages) => {
                            let content = messages.last().unwrap().expect_assistant().to_string();
                            Ok(ToolResult::new(tool_use.id, content))
                        }
                    }
                } else if let Some(tool) = agent.tools.get(&tool_use.name) {
                    Ok(tool
                        .invoke(tool_use.id, &tool_use.input, agent.cx.clone())
                        .await)
                } else {
                    unreachable!()
                }
            });
        }

        let mut tool_results = Vec::new();
        while let Some(Ok(result)) = join_set.join_next().await {
            tool_results.push(result);
        }
        tool_results
    }

    async fn process_event_stream<P: Into<Participant> + Clone>(
        &self,
        from: P,
        to: P,
        mut stream: EventStream,
        tx: Option<ChatMessageSender>,
    ) -> Result<MessagesResponse, AgentError> {
        let mut response_builder = MessagesResponseBuilder::default();

        while let Some(event) = stream.next().await {
            let event = match event {
                Ok(e) => e,
                Err(e) => {
                    if let Some(tx) = &tx {
                        tx.send(Err(e)).unwrap();
                    }
                    continue;
                }
            };

            response_builder.push_event(event.clone());

            match event {
                EventData::MessageStop => break,
                EventData::Error { error } => {
                    if let Some(tx) = &tx {
                        tx.send(Err(error.into())).unwrap();
                    }
                    break;
                }
                _ => {
                    if let Some(tx) = &tx {
                        let content = ChatEventMessage::Response {
                            from: from.clone().into(),
                            to: to.clone().into(),
                            event: event.clone(),
                        };
                        tx.send(Ok(content)).unwrap();
                    }
                }
            }
        }

        response_builder.build().map_err(AgentError::from)
    }

    pub async fn send<T: Into<String>>(&self, content: T) -> Result<Messages, AgentError> {
        let agent = self.clone();
        let messages = vec![content.into()];
        agent.execute(Participant::System, messages, None).await
    }

    pub fn event_stream<T: Into<String>>(&self, content: T) -> ChatEventStream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let agent = self.clone();
        let messages = vec![content.into()];
        tokio::spawn(async move {
            let _ = agent.execute(Participant::System, messages, Some(tx)).await;
        });
        Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }

    pub fn content_stream<T: Into<String>>(&self, content: T) -> ChatContentStream {
        let content = content.into();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let initial_message = ChatContentMessage::Request {
            from: Participant::System,
            to: self.name.clone().into(),
            content: content.clone().into(),
        };
        tx.send(Ok(initial_message)).unwrap();
        let mut stream = self.event_stream(content);
        tokio::spawn(async move {
            let tx_clone = tx.clone();
            let mut builder =
                ChatContentMessageBuilder::new(move |message| tx_clone.send(Ok(message)).unwrap());
            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatEventMessage::Request { from, to, content }) => {
                        let message = ChatContentMessage::Request { from, to, content };
                        tx.send(Ok(message)).unwrap();
                    }
                    Ok(ChatEventMessage::Response { from, to, event }) => {
                        builder.push_event(from, to, event);
                    }
                    Err(e) => {
                        tx.send(Err(e)).unwrap();
                    }
                }
            }
        });
        Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }

    async fn execute<T: Into<Messages>, P: Into<Participant> + Clone>(
        &self,
        from: P,
        messages: T,
        tx: Option<ChatMessageSender>,
    ) -> Result<Messages, AgentError> {
        let mut messages = messages.into();

        loop {
            let stream = self.prepare_request(messages.clone()).stream();
            let response = self
                .process_event_stream(
                    from.clone().into(),
                    self.name.clone().into(),
                    Box::pin(stream),
                    tx.clone(),
                )
                .await?;

            messages.add_message(response.clone());
            let tool_results = self.process_tools_invoke(response, tx.clone()).await;
            if tool_results.iter().any(|r| r.is_err()) {
                return Err(tool_results.into_iter().find_map(|r| r.err()).unwrap());
            }

            if !tool_results.is_empty() {
                let mut tool_content = tool_results
                    .into_iter()
                    .filter_map(Result::ok)
                    .map(MultimodalContent::ToolResult)
                    .collect::<Vec<_>>();
                if let Ok(Model::Claude3Haiku) = Model::from_str(&self.model) {
                    tool_content.push(MultimodalContent::Text(Text {
                        text: "Here are the tool results.".to_string(),
                    }))
                }
                for elem in &tool_content {
                    let content = ChatEventMessage::Request {
                        from: from.clone().into(),
                        to: self.name.clone().into(),
                        content: elem.clone(),
                    };
                    if let Some(tx) = tx.as_ref() {
                        tx.send(Ok(content)).unwrap()
                    }
                }
                messages.add_message(tool_content);
            } else {
                return Ok(messages);
            }
        }
    }

    pub fn as_tool(&self) -> Tool {
        Tool::builder()
            .with_name(self.name.clone())
            .with_props::<AgentInput>()
            .build()
            .expect("To build Tool")
    }
}

#[derive(Debug, Clone, Default)]
pub struct Agents(pub FxHashMap<String, Agent>);

impl Agents {
    #[must_use]
    pub fn new() -> Self {
        Agents::default()
    }

    pub fn add(&mut self, agent: Agent) {
        self.0.insert(agent.name.clone(), agent);
    }

    #[must_use]
    pub fn get(&self, agent_name: &str) -> Option<&Agent> {
        self.0.get(agent_name)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Agent> {
        self.0.values()
    }

    pub fn as_tools(&self) -> Tools {
        self.0
            .values()
            .fold(Tools::default(), |mut tools, subagent| {
                tools.add(subagent.as_tool());
                tools
            })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicBool, Arc};

    use serde::Serialize;
    use serde_json::json;

    use crate::{Anthropic, Model};

    use super::*;

    // Helper function to create a default AgentBuilder
    fn default_builder() -> AgentBuilder {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let anthropic = Anthropic::builder().api_key(api_key).build().unwrap();
        AgentBuilder::default()
            .with_name("Test Agent")
            .with_description("A test agent")
            .with_model("claude-3-haiku-20240307")
            .with_system_message("You are a test agent")
            .with_client(anthropic)
    }

    #[test]
    fn test_agent_builder_with_subagent() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let anthropic = Anthropic::builder().api_key(api_key).build().unwrap();
        let subagent = Agent {
            name: "Subagent".to_string(),
            description: "A subagent".to_string(),
            model: "claude-3-haiku-20240307".to_string(),
            system_message: "You are a subagent".to_string(),
            max_tokens: 2048,
            subagents: Agents::default(),
            tools: Tools::default(),
            cx: ToolContext::default(),
            anthropic,
        };
        let builder = AgentBuilder::default().with_subagent(subagent);
        assert_eq!(builder.subagents.len(), 1);
    }

    #[test]
    fn test_agent_builder_build_success() {
        let builder = default_builder();
        let result = builder.build();
        assert!(result.is_ok());
        let agent = result.unwrap();
        assert_eq!(agent.name, "Test Agent");
        assert_eq!(agent.description, "A test agent");
        assert_eq!(agent.model, "claude-3-haiku-20240307");
        assert_eq!(agent.system_message, "You are a test agent");
        assert_eq!(agent.max_tokens, 4096);
    }

    #[test]
    fn test_agent_builder_build_missing_name() {
        let builder = default_builder().with_name("");
        let result = builder.build();
        assert!(matches!(result, Err(AgentBuilderError::MissingName)));
    }

    #[test]
    fn test_agent_builder_build_missing_description() {
        let builder = default_builder().with_description("");
        let result = builder.build();
        assert!(matches!(result, Err(AgentBuilderError::MissingDescription)));
    }

    #[test]
    fn test_agent_builder_build_missing_model() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let anthropic = Anthropic::builder().api_key(api_key).build().unwrap();
        let builder = AgentBuilder::default()
            .with_name("Test Agent")
            .with_description("A test agent")
            .with_system_message("You are a test agent")
            .with_client(anthropic);
        let result = builder.build();
        assert!(matches!(result, Err(AgentBuilderError::MissingModel)));
    }

    #[test]
    fn test_agent_builder_build_missing_system_message() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let anthropic = Anthropic::builder().api_key(api_key).build().unwrap();
        let builder = AgentBuilder::default()
            .with_name("Test Agent")
            .with_description("A test agent")
            .with_model("gpt-4")
            .with_client(anthropic);
        let result = builder.build();
        assert!(matches!(
            result,
            Err(AgentBuilderError::MissingSystemMessage)
        ));
    }

    #[test]
    fn test_agent_builder_build_missing_anthropic_client() {
        let builder = AgentBuilder::default()
            .with_name("Test Agent")
            .with_description("A test agent")
            .with_model("gpt-4")
            .with_system_message("You are a test agent");
        let result = builder.build();
        assert!(matches!(
            result,
            Err(AgentBuilderError::MissingAnthropicClient)
        ));
    }

    #[test]
    fn test_agent_builder_custom_max_tokens() {
        let builder = default_builder().with_max_tokens(2048);
        let agent = builder.build().unwrap();
        assert_eq!(agent.max_tokens, 2048);
    }

    #[tokio::test]
    async fn test_agent_send_success() {
        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
        struct TestHandlerProps {
            random_number: i32,
        }
        async fn test_handler(
            _input: TestHandlerProps,
            _cx: ToolContext,
        ) -> Result<serde_json::Value, ToolError> {
            Ok(json!("To finish this test use [finish_test] tool"))
        }

        async fn finish_handler(
            _input: serde_json::Value,
            cx: ToolContext,
        ) -> Result<serde_json::Value, ToolError> {
            let is_finished = cx.expect_resource::<Arc<AtomicBool>>();
            is_finished.store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(json!(
                "Congratulations! You finished the test. To exit write \"Bingo!\" "
            ))
        }

        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let anthropic = Anthropic::builder()
            .api_key(api_key)
            .client(&client)
            .build()
            .unwrap();

        let test_tool = Tool::builder()
            .with_name("test_tool")
            .with_handler(test_handler)
            .build()
            .unwrap();

        let finish_tool = Tool::builder()
            .with_name("finish_test")
            .with_handler(finish_handler)
            .build()
            .unwrap();

        let is_finished = Arc::new(AtomicBool::new(false));
        let tcx = ToolContext::default();
        tcx.add_resource(is_finished.clone());

        let agent = Agent::builder()
            .with_name("Test Agent")
            .with_description("Test Agent")
            .with_model(Model::Claude3Haiku.to_string())
            .with_system_message("You are a Test Agent. This is a library test. Follow instructions to successfuly finish test.")
            .with_client(anthropic)
            .with_tools(vec![test_tool, finish_tool])
            .with_context(tcx)
            .build()
            .unwrap();

        let messages = agent
            .send("This is testing environent. To continue this test use [test_tool]. Don't ask questions. You are on your own! Do what need to be done!").await.unwrap();
        dbg!(messages);
    }
    #[tokio::test]
    async fn test_agent_stream_success() {
        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
        struct TestHandlerProps {
            random_number: i32,
        }
        async fn test_handler(
            _input: TestHandlerProps,
            _cx: ToolContext,
        ) -> Result<serde_json::Value, ToolError> {
            Ok(json!("To finish this test use [finish_test] tool"))
        }

        async fn finish_handler(
            _input: serde_json::Value,
            _cx: ToolContext,
        ) -> Result<serde_json::Value, ToolError> {
            Ok(json!(
                "Congratulations! You finished the test. To exit write \"Bingo!\" "
            ))
        }

        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let anthropic = Anthropic::builder()
            .api_key(api_key)
            .client(&client)
            .build()
            .unwrap();

        let test_tool = Tool::builder()
            .with_name("test_tool")
            .with_handler(test_handler)
            .build()
            .unwrap();

        let finish_tool = Tool::builder()
            .with_name("finish_test")
            .with_handler(finish_handler)
            .build()
            .unwrap();

        let agent = Agent::builder()
            .with_name("Test Agent")
            .with_description("Test Agent")
            .with_model(Model::Claude3Haiku.to_string())
            .with_system_message("You are a Test Agent. This is a library test. Follow instructions to successfuly finish test.")
            .with_client(anthropic)
            .with_tools(vec![test_tool, finish_tool])
            .build()
            .unwrap();

        let mut stream = agent
            .event_stream("This is testing environent. To continue this test use [test_tool]. Don't ask questions. You are on your own! Do what need to be done!");
        while let Some(event) = stream.next().await {
            dbg!(event);
        }
    }

    #[tokio::test]
    async fn test_multiagent_send_success() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let anthropic = Anthropic::builder()
            .api_key(api_key)
            .client(&client)
            .build()
            .unwrap();

        let agent3 = Agent::builder()
                .with_name("Test Agent3")
                .with_description("Test Agent3")
                .with_model(Model::Claude3Haiku.to_string())
                .with_system_message("You are a Test Agent3. This is a library test. Follow instructions to successfuly finish test.")
                .with_client(anthropic.clone())
                .build()
                .unwrap();

        let agent2 = Agent::builder()
            .with_name("Test Agent2")
            .with_description("Test Agent2")
            .with_model(Model::Claude3Haiku.to_string())
            .with_system_message("You are a Test Agent2. This is a library test. Follow instructions to successfuly finish test.")
            .with_client(anthropic.clone())
            .with_subagent(agent3)
            .build()
            .unwrap();

        let agent1 = Agent::builder()
            .with_name("Test Agent1")
            .with_description("Test Agent1")
            .with_model(Model::Claude3Haiku.to_string())
            .with_system_message("You are a Test Agent1. This is a library test. Follow instructions to successfuly finish test.")
            .with_client(anthropic)
            .with_subagent(agent2)
            .build()
            .unwrap();

        let messages = agent1
                .send("This is a testing environment. To proceed, verify if you can communicate with your subagent and if the subagent can communicate with their subagents.").await.unwrap();
        dbg!(messages);
    }

    #[tokio::test]
    async fn test_multiagent_stream_success() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let anthropic = Anthropic::builder()
            .api_key(api_key)
            .client(&client)
            .build()
            .unwrap();

        let agent3 = Agent::builder()
            .with_name("test_agent3")
            .with_description("Test Agent3")
            .with_model(Model::Claude35Sonnet.to_string())
            .with_system_message("{wycięte dla lepszego formatowania}")
            .with_client(anthropic.clone())
            .build()
            .unwrap();

        let agent2 = Agent::builder()
            .with_name("test_agent2")
            .with_description("Test Agent2")
            .with_model(Model::Claude35Sonnet.to_string())
            .with_system_message("{wycięte dla lepszego formatowania}")
            .with_client(anthropic.clone())
            .with_subagent(agent3)
            .build()
            .unwrap();

        let agent1 = Agent::builder()
            .with_name("test_agent1")
            .with_description("Test Agent1")
            .with_model(Model::Claude35Sonnet.to_string())
            .with_system_message("{wycięte dla lepszego formatowania}")
            .with_client(anthropic)
            .with_subagent(agent2)
            .build()
            .unwrap();

        let mut stream = agent1
                .content_stream("This is a testing environment. To proceed, verify if you can communicate with your subagent and if the subagent can communicate with their subagents. Send exacly one message to check this.");
        while let Some(event) = stream.next().await {
            println!("{event}");
        }
    }
}
