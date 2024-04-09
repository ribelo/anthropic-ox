use derivative::Derivative;
use serde::{Deserialize, Serialize};

use super::tools::{ToolResult, ToolUse};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize, Derivative, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    #[serde(rename = "content")]
    Text(String),
    #[serde(rename = "content")]
    Multiple(Vec<MultimodalContent>),
}

impl From<String> for MessageContent {
    fn from(content: String) -> Self {
        MessageContent::Text(content)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Image {
    pub source: ImageSource,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Text {
    pub text: String,
}

impl From<String> for Text {
    fn from(text: String) -> Self {
        Text { text }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum MultimodalContent {
    Text(Text),
    Image(Image),
    ToolUse(ToolUse),
    ToolResult(ToolResult),
}

impl From<String> for MultimodalContent {
    fn from(text: String) -> Self {
        MultimodalContent::Text(text.into())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UserMessage {
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl UserMessage {
    pub fn new<T: Into<MessageContent>>(content: T) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
        }
    }
}

impl From<String> for UserMessage {
    fn from(content: String) -> Self {
        UserMessage::new(content)
    }
}

#[derive(Debug, Serialize, Deserialize, Derivative, Clone)]
pub struct AssistantMessage {
    #[derivative(Default(value = "Role::Assistant"))]
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl AssistantMessage {
    pub fn new<T: Into<MessageContent>>(content: T) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
        }
    }
}

impl From<String> for AssistantMessage {
    fn from(content: String) -> Self {
        AssistantMessage::new(content)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
}

impl Message {
    pub fn user<T: Into<MessageContent>>(content: T) -> Self {
        Message::User(UserMessage::new(content))
    }
    pub fn assistant<T: Into<MessageContent>>(content: T) -> Self {
        Message::Assistant(AssistantMessage::new(content))
    }
    pub fn content(&self) -> &MessageContent {
        match self {
            Message::User(msg) => &msg.content,
            Message::Assistant(msg) => &msg.content,
        }
    }
}

impl From<UserMessage> for Message {
    fn from(message: UserMessage) -> Self {
        Message::User(message)
    }
}

impl From<AssistantMessage> for Message {
    fn from(message: AssistantMessage) -> Self {
        Message::Assistant(message)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Messages(pub Vec<Message>);

impl Messages {
    pub fn push_message(&mut self, message: impl Into<Message>) {
        self.0.push(message.into());
    }
}

impl From<Message> for Messages {
    fn from(value: Message) -> Self {
        Messages(vec![value])
    }
}

impl<T> Extend<T> for Messages
where
    T: Into<Message>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.0.extend(iter.into_iter().map(Into::into));
    }
}

impl IntoIterator for Messages {
    type Item = Message;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
