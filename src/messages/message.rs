use derivative::Derivative;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize, Derivative, Clone)]
#[derivative(Default)]
pub struct UserMessage {
    #[derivative(Default(value = "Role::User"))]
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl UserMessage {
    pub fn new(content: impl ToString) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            name: None,
        }
    }
}

impl From<UserMessage> for String {
    fn from(message: UserMessage) -> Self {
        message.content
    }
}

impl From<&str> for UserMessage {
    fn from(content: &str) -> Self {
        UserMessage::new(content)
    }
}

#[derive(Debug, Serialize, Deserialize, Derivative, Clone)]
pub struct AssistantMessage {
    #[derivative(Default(value = "Role::Assistant"))]
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl AssistantMessage {
    pub fn new(content: impl ToString) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.to_string()),
            name: None,
        }
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
    pub fn user(content: impl ToString) -> Self {
        Message::User(UserMessage::new(content))
    }
    pub fn assistant(content: impl ToString) -> Self {
        Message::Assistant(AssistantMessage::new(content))
    }
    pub fn content(&self) -> Option<String> {
        match self {
            Message::User(msg) => Some(msg.content.clone()),
            Message::Assistant(msg) => msg.content.as_ref().cloned(),
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
        self.0.extend(iter.into_iter().map(|item| item.into()));
    }
}

impl IntoIterator for Messages {
    type Item = Message;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
