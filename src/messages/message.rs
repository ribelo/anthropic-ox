use std::{collections::HashMap, str::FromStr};

use derivative::Derivative;
use regex::Regex;
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

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct MessageBuilder {
    variables: HashMap<String, String>,
    content: Option<String>,
    role: Option<Role>,
}

#[derive(Debug, thiserror::Error)]
pub enum MessageBuilderError {
    #[error("Missing content")]
    MissingContent,
    #[error("Missing role")]
    MissingRole,
    #[error("Variable {0} not found")]
    VariableNotFound(String),
}

impl MessageBuilder {
    pub fn variable<T: ToString>(mut self, name: T, value: T) -> Self {
        self.variables.insert(name.to_string(), value.to_string());
        self
    }
    pub fn content<T: ToString>(mut self, content: T) -> Self {
        self.content = Some(content.to_string());
        self
    }
    pub fn role(mut self, role: Role) -> Self {
        self.role = Some(role);
        self
    }
    pub fn build(self) -> Result<Message, MessageBuilderError> {
        let content = self.content.ok_or(MessageBuilderError::MissingContent)?;
        let role = self.role.ok_or(MessageBuilderError::MissingRole)?;
        let replaced_content = content.replace_variables(&self.variables)?;
        match role {
            Role::User => Ok(Message::user(replaced_content)),
            Role::Assistant => Ok(Message::assistant(replaced_content)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Variable {0} not found")]
pub struct ReplaceVariableError(String);

impl From<ReplaceVariableError> for MessageBuilderError {
    fn from(value: ReplaceVariableError) -> Self {
        MessageBuilderError::VariableNotFound(value.0)
    }
}

trait ReplaceVariables: Sized {
    fn replace_variables(
        &self,
        variables: &HashMap<String, String>,
    ) -> Result<Self, ReplaceVariableError>;
}

impl<T: ?Sized + ToString + From<String>> ReplaceVariables for T {
    fn replace_variables(
        &self,
        variables: &HashMap<String, String>,
    ) -> Result<Self, ReplaceVariableError> {
        let content = self.to_string();
        let mut result = content.clone();

        let re = Regex::new(r"\{\$([A-Za-z0-9_]+)\}").unwrap();
        let placeholders = re
            .captures_iter(&content)
            .map(|cap| cap[1].to_string())
            .collect::<Vec<_>>();

        for placeholder in placeholders {
            if let Some(value) = variables.get(&placeholder) {
                let placeholder_str = format!(r"{{${}}}", placeholder);
                result = result.replace(&placeholder_str, value);
            } else {
                return Err(ReplaceVariableError(placeholder));
            }
        }

        Ok(Self::from(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_replace_variables_string_success() {
        let input = "Hello, {$name}!";
        let mut variables = HashMap::new();
        variables.insert("name".to_string(), "Alice".to_string());

        let result = input.to_string().replace_variables(&variables).unwrap();

        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_replace_variables_string_multiple_vars() {
        let input = "{$greeting}, {$name}! You are {$age} years old.";
        let mut variables = HashMap::new();
        variables.insert("greeting".to_string(), "Hi".to_string());
        variables.insert("name".to_string(), "Bob".to_string());
        variables.insert("age".to_string(), "30".to_string());

        let result = input.to_string().replace_variables(&variables).unwrap();

        assert_eq!(result, "Hi, Bob! You are 30 years old.");
    }

    #[test]
    fn test_replace_variables_string_no_vars() {
        let input = "No variables here";
        let variables = HashMap::new();

        let result = input.to_string().replace_variables(&variables).unwrap();

        assert_eq!(result, "No variables here");
    }

    #[test]
    fn test_replace_variables_string_unused_var() {
        let input = "Hello, {$name}!";
        let mut variables = HashMap::new();
        variables.insert("name".to_string(), "Alice".to_string());
        variables.insert("unused".to_string(), "value".to_string());

        let result = input.to_string().replace_variables(&variables).unwrap();

        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_replace_variables_missing_var() {
        let input = "Hello, {$name}! You are {$age} years old.";
        let mut variables = HashMap::new();
        variables.insert("name".to_string(), "Alice".to_string());
        let result = input.to_string().replace_variables(&variables);
        match result {
            Ok(_) => panic!("Expected an error but got Ok"),
            Err(e) => assert_eq!(e.to_string(), "Variable age not found"),
        }
    }
}
