use std::{
    collections::HashMap,
    pin::Pin,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio_stream::{Stream, StreamExt};

use crate::{
    messages::{
        message::{Messages, MultimodalContent, Text},
        tool::{
            AnyTool, Tool, ToolBox, ToolError, ToolMetadataInfo, ToolResult, ToolUse,
            ToolUseBuilder,
        },
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
        content: Vec<MultimodalContent>,
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
        content: Vec<MultimodalContent>,
    },
    Response {
        from: Participant,
        to: Participant,
        content: Vec<MultimodalContent>,
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
                            content: vec![content],
                        }
                    }
                    (None, Some(tool_use_builder)) => {
                        let tool_use = tool_use_builder.build().expect("Failed to build tool use");
                        let content = MultimodalContent::ToolUse(tool_use);
                        ChatContentMessage::Response {
                            from: to,
                            to: from,
                            content: vec![content],
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
    pub message: String,
}

#[derive(Clone, Default)]
pub struct AgentBox {
    agents: Arc<RwLock<HashMap<String, Arc<dyn AnyAgent>>>>,
}

impl AgentBox {
    pub fn add<T: AnyAgent + 'static>(&self, agent: T) {
        let name = agent.name().to_string();
        self.agents
            .write()
            .expect("Failed to acquire write lock")
            .insert(name, Arc::new(agent));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn AnyAgent>> {
        self.agents
            .read()
            .expect("Failed to acquire read lock")
            .get(name)
            .cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.agents
            .read()
            .expect("Failed to acquire read lock")
            .is_empty()
    }

    pub fn len(&self) -> usize {
        self.agents
            .read()
            .expect("Failed to acquire read lock")
            .len()
    }

    pub fn metadata(&self) -> Vec<ToolMetadataInfo> {
        let agents = self.agents.read().expect("Lock poisoned");
        agents
            .values()
            .map(|agent| ToolMetadataInfo {
                name: agent.name(),
                description: agent.description().map(ToString::to_string),
                input_schema: agent.input_schema(),
            })
            .collect()
    }
}

#[async_trait]
pub trait AnyAgent: Send + Sync + 'static {
    fn name(&self) -> String;
    fn description(&self) -> Option<&str>;
    fn input_schema(&self) -> Value;
    async fn execute(
        &self,
        from: Participant,
        messages: Messages,
        tx: Option<ChatMessageSender>,
    ) -> Result<Messages, AgentError>;
}

#[async_trait]
pub trait Agent: Clone + Send + Sync + 'static {
    fn name(&self) -> String;

    fn description(&self) -> Option<&str> {
        None
    }

    fn model(&self) -> Model;

    fn system_message(&self) -> Option<&str> {
        None
    }

    fn max_tokens(&self) -> u32 {
        4096
    }

    fn client(&self) -> &Anthropic;

    fn subagents(&self) -> Option<&AgentBox> {
        None
    }

    fn get_subagent(&self, name: &str) -> Option<Arc<dyn AnyAgent>> {
        self.subagents().and_then(|agent_box| agent_box.get(name))
    }

    fn tools(&self) -> Option<&ToolBox> {
        None
    }

    fn get_tool(&self, name: &str) -> Option<Arc<dyn AnyTool>> {
        self.tools().and_then(|tool_box| tool_box.get(name))
    }

    fn input_schema(&self) -> Value {
        let mut settings = schemars::gen::SchemaSettings::draft07();
        settings.inline_subschemas = true;
        let gen = schemars::gen::SchemaGenerator::new(settings);
        let json_schema = gen.into_root_schema_for::<AgentInput>();
        let mut input_schema = serde_json::to_value(json_schema).unwrap();
        input_schema["type"] = serde_json::json!("object");
        input_schema
    }

    fn prepare_request(&self, messages: Messages) -> MessagesRequest {
        let mut builder = self
            .client()
            .messages()
            .with_max_tokens(self.max_tokens())
            .with_model(self.model().to_string())
            .with_messages(messages);

        if let Some(system_message) = self.system_message() {
            builder = builder.with_system(system_message);
        }

        if let Some(tools) = self.tools() {
            builder = builder.with_tools(tools.clone());
        }

        builder.build().expect("Valid agent")
    }

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
                if let Some(subagent) = agent.get_subagent(&tool_use.name) {
                    let message =
                        MultimodalContent::text(serde_json::to_string(&tool_use.input).unwrap());
                    let result = subagent
                        .execute(agent.name().into(), vec![message].into(), tx)
                        .await;
                    match result {
                        Err(e) => Err(e),
                        Ok(messages) => {
                            let content = messages.last().unwrap().expect_assistant().to_string();
                            Ok(ToolResult::new(tool_use.id, content))
                        }
                    }
                } else if let Some(tool) = agent.get_tool(&tool_use.name) {
                    Ok(tool.invoke_any(tool_use).await)
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

    async fn process_event_stream(
        &self,
        from: Participant,
        to: Participant,
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

            dbg!(&event);
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
                            from: from.clone(),
                            to: to.clone(),
                            event: event.clone(),
                        };
                        tx.send(Ok(content)).unwrap();
                    }
                }
            }
        }

        response_builder.build().map_err(AgentError::from)
    }

    async fn execute(
        &self,
        from: Participant,
        mut messages: Messages,
        tx: Option<ChatMessageSender>,
    ) -> Result<Messages, AgentError> {
        loop {
            let stream = self.prepare_request(messages.clone()).stream();
            let response = self
                .process_event_stream(
                    from.clone(),
                    self.name().into(),
                    Box::pin(stream),
                    tx.clone(),
                )
                .await?;
            dbg!(&response);
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
                if Model::Claude3Haiku == self.model() {
                    tool_content.push(MultimodalContent::Text(Text {
                        text: "Here are the tool results.".to_string(),
                    }))
                }
                for elem in &tool_content {
                    let content = ChatEventMessage::Request {
                        from: from.clone(),
                        to: self.name().into(),
                        content: vec![elem.clone()],
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

    async fn send(&self, content: Vec<MultimodalContent>) -> Result<Messages, AgentError> {
        let messages = vec![content];
        self.execute(Participant::System, messages.into(), None)
            .await
    }

    fn event_stream(&self, content: Vec<MultimodalContent>) -> ChatEventStream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let agent = self.clone();
        let messages = vec![content];
        tokio::spawn(async move {
            let _ = agent
                .execute(Participant::System, messages.into(), Some(tx))
                .await;
        });
        Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }

    fn content_stream(&self, content: Vec<MultimodalContent>) -> ChatContentStream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let initial_message = ChatContentMessage::Request {
            from: Participant::System,
            to: self.name().into(),
            content: content.clone(),
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
}

#[async_trait]
impl<T: Agent> AnyAgent for T {
    fn name(&self) -> String {
        Agent::name(self)
    }

    fn description(&self) -> Option<&str> {
        self.description()
    }

    fn input_schema(&self) -> Value {
        self.input_schema()
    }

    async fn execute(
        &self,
        from: Participant,
        messages: Messages,
        tx: Option<ChatMessageSender>,
    ) -> Result<Messages, AgentError> {
        Agent::execute(self, from, messages, tx).await
    }
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    ApiError(#[from] ApiError),
    #[error(transparent)]
    ToolError(#[from] ToolError),
    #[error(transparent)]
    MessagesResponseBuilderError(#[from] MessagesResponseBuilderError),
    #[error(transparent)]
    SerdeError(serde_json::Error),
}

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicBool, Arc};

    use serde::Serialize;
    use serde_json::json;

    use crate::{Anthropic, Model};

    use super::*;

    #[tokio::test]
    async fn test_agent_send_success() {
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

        #[derive(Clone)]
        struct TestAgent {
            anthropic: Anthropic,
        }

        impl TestAgent {
            pub fn new(client: Anthropic) -> Self {
                Self { anthropic: client }
            }
        }

        impl Agent for TestAgent {
            fn name(&self) -> String {
                "TestAgent".to_string()
            }

            fn model(&self) -> Model {
                Model::Claude3Haiku
            }

            fn system_message(&self) -> Option<&str> {
                Some("You are a Test Agent. This is a library test. Follow instructions to successfuly finish test.")
            }

            fn client(&self) -> &Anthropic {
                &self.anthropic
            }
        }

        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let anthropic = Anthropic::builder()
            .api_key(api_key)
            .client(&client)
            .build()
            .unwrap();

        let tools = ToolBox::default();
        tools.add(TestTool);
        let finish_tool = FinishTool::default();
        tools.add(finish_tool.clone());

        let agent = TestAgent::new(anthropic);

        let messages = agent
            .send(vec!["This is testing environent. To continue this test use [test_tool]. Don't ask questions. You are on your own! Do what need to be done!".into()]).await.unwrap();
        dbg!(messages);
    }
    #[tokio::test]
    async fn test_agent_stream_success() {
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

        #[derive(Clone)]
        struct TestAgent {
            anthropic: Anthropic,
            tools: ToolBox,
        }

        impl TestAgent {
            pub fn new(client: Anthropic) -> Self {
                let tools = ToolBox::default();
                tools.add(TestTool);
                tools.add(FinishTool::default());
                Self {
                    anthropic: client,
                    tools,
                }
            }
        }

        impl Agent for TestAgent {
            fn name(&self) -> String {
                "TestAgent".to_string()
            }

            fn model(&self) -> Model {
                Model::Claude35Sonnet
            }

            fn system_message(&self) -> Option<&str> {
                Some("You are a Test Agent. This is a library test. Follow instructions to successfuly finish test.")
            }

            fn client(&self) -> &Anthropic {
                &self.anthropic
            }

            fn tools(&self) -> Option<&ToolBox> {
                Some(&self.tools)
            }
        }

        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let anthropic = Anthropic::builder()
            .api_key(api_key)
            .client(&client)
            .build()
            .unwrap();

        let tools = ToolBox::default();
        tools.add(TestTool);
        let finish_tool = FinishTool::default();
        tools.add(finish_tool.clone());

        let agent = TestAgent::new(anthropic);

        let mut stream = agent
            .send(vec!["This is testing environent. To continue this test use [test_tool]. Don't ask questions. You are on your own! Do what need to be done!".into()]).await;
        dbg!(stream.unwrap());
        // while let Some(msg) = stream.next().await {
        //     dbg!(msg);
        // }
    }

    // #[tokio::test]
    // async fn test_multiagent_send_success() {
    //     let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
    //     let client = reqwest::Client::new();
    //     let anthropic = Anthropic::builder()
    //         .api_key(api_key)
    //         .client(&client)
    //         .build()
    //         .unwrap();
    //
    //     let agent3 = Agent::builder()
    //             .with_name("Test Agent3")
    //             .with_description("Test Agent3")
    //             .with_model(Model::Claude3Haiku.to_string())
    //             .with_system_message("You are a Test Agent3. This is a library test. Follow instructions to successfuly finish test.")
    //             .with_client(anthropic.clone())
    //             .build()
    //             .unwrap();
    //
    //     let agent2 = Agent::builder()
    //         .with_name("Test Agent2")
    //         .with_description("Test Agent2")
    //         .with_model(Model::Claude3Haiku.to_string())
    //         .with_system_message("You are a Test Agent2. This is a library test. Follow instructions to successfuly finish test.")
    //         .with_client(anthropic.clone())
    //         .with_subagent(agent3)
    //         .build()
    //         .unwrap();
    //
    //     let agent1 = Agent::builder()
    //         .with_name("Test Agent1")
    //         .with_description("Test Agent1")
    //         .with_model(Model::Claude3Haiku.to_string())
    //         .with_system_message("You are a Test Agent1. This is a library test. Follow instructions to successfuly finish test.")
    //         .with_client(anthropic)
    //         .with_subagent(agent2)
    //         .build()
    //         .unwrap();
    //
    //     let messages = agent1
    //             .send("This is a testing environment. To proceed, verify if you can communicate with your subagent and if the subagent can communicate with their subagents.").await.unwrap();
    //     dbg!(messages);
    // }
    //
    // #[tokio::test]
    // async fn test_multiagent_stream_success() {
    //     let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
    //     let client = reqwest::Client::new();
    //     let anthropic = Anthropic::builder()
    //         .api_key(api_key)
    //         .client(&client)
    //         .build()
    //         .unwrap();
    //
    //     let agent3 = Agent::builder()
    //         .with_name("test_agent3")
    //         .with_description("Test Agent3")
    //         .with_model(Model::Claude35Sonnet.to_string())
    //         .with_system_message("{wycięte dla lepszego formatowania}")
    //         .with_client(anthropic.clone())
    //         .build()
    //         .unwrap();
    //
    //     let agent2 = Agent::builder()
    //         .with_name("test_agent2")
    //         .with_description("Test Agent2")
    //         .with_model(Model::Claude35Sonnet.to_string())
    //         .with_system_message("{wycięte dla lepszego formatowania}")
    //         .with_client(anthropic.clone())
    //         .with_subagent(agent3)
    //         .build()
    //         .unwrap();
    //
    //     let agent1 = Agent::builder()
    //         .with_name("test_agent1")
    //         .with_description("Test Agent1")
    //         .with_model(Model::Claude35Sonnet.to_string())
    //         .with_system_message("{wycięte dla lepszego formatowania}")
    //         .with_client(anthropic)
    //         .with_subagent(agent2)
    //         .build()
    //         .unwrap();
    //
    //     let mut stream = agent1
    //             .content_stream("This is a testing environment. To proceed, verify if you can communicate with your subagent and if the subagent can communicate with their subagents. Send exacly one message to check this.");
    //     while let Some(event) = stream.next().await {
    //         println!("{event}");
    //     }
    // }
}
