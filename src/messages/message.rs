use std::{collections::HashMap, fmt, str::FromStr};

use derivative::Derivative;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::tools::{ExtractToolUse, ToolResult, ToolUse};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Image {
    pub source: ImageSource,
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "image: {}", self.source)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Text {
    pub text: String,
}

impl From<String> for Text {
    fn from(text: String) -> Self {
        Text { text }
    }
}

impl From<&str> for Text {
    fn from(text: &str) -> Self {
        Text {
            text: text.to_string(),
        }
    }
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.text)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
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

impl From<&str> for MultimodalContent {
    fn from(text: &str) -> Self {
        MultimodalContent::Text(text.into())
    }
}

impl From<Image> for MultimodalContent {
    fn from(image: Image) -> Self {
        MultimodalContent::Image(image)
    }
}

impl From<ToolUse> for MultimodalContent {
    fn from(tool_use: ToolUse) -> Self {
        MultimodalContent::ToolUse(tool_use)
    }
}

impl From<ToolResult> for MultimodalContent {
    fn from(tool_result: ToolResult) -> Self {
        MultimodalContent::ToolResult(tool_result)
    }
}

impl fmt::Display for MultimodalContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultimodalContent::Text(text) => write!(f, "{}", text),
            MultimodalContent::Image(image) => write!(f, "{}", image),
            MultimodalContent::ToolUse(tool_use) => write!(f, "{}", tool_use),
            MultimodalContent::ToolResult(tool_result) => write!(f, "{}", tool_result),
        }
    }
}

impl ExtractToolUse for MultimodalContent {
    fn extract_tool_uses(&self) -> Vec<&ToolUse> {
        match self {
            MultimodalContent::ToolUse(tool_use) => vec![tool_use],
            _ => Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
}

impl fmt::Display for ImageSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageSource::Base64 { media_type, data } => {
                write!(f, "base64 ({}, {})", media_type, data)
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct UserMessage {
    pub role: Role,
    pub content: Vec<MultimodalContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl UserMessage {
    pub fn new<T: Into<MultimodalContent>>(content: Vec<T>) -> Self {
        Self {
            role: Role::User,
            content: content.into_iter().map(Into::into).collect(),
            name: None,
        }
    }
}

impl<T: Into<MultimodalContent>> From<T> for UserMessage {
    fn from(value: T) -> Self {
        UserMessage {
            role: Role::User,
            content: vec![value.into()],
            name: None,
        }
    }
}

impl From<Vec<MultimodalContent>> for UserMessage {
    fn from(value: Vec<MultimodalContent>) -> Self {
        UserMessage {
            role: Role::User,
            content: value,
            name: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Derivative, Clone, PartialEq, Eq)]
pub struct AssistantMessage {
    #[derivative(Default(value = "Role::Assistant"))]
    pub role: Role,
    pub content: Vec<MultimodalContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl AssistantMessage {
    pub fn new<T: Into<MultimodalContent>>(content: Vec<T>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into_iter().map(Into::into).collect(),
            name: None,
        }
    }
}

impl<T: Into<MultimodalContent>> From<T> for AssistantMessage {
    fn from(value: T) -> Self {
        AssistantMessage {
            role: Role::Assistant,
            content: vec![value.into()],
            name: None,
        }
    }
}

impl From<Vec<MultimodalContent>> for AssistantMessage {
    fn from(value: Vec<MultimodalContent>) -> Self {
        AssistantMessage {
            role: Role::Assistant,
            content: value,
            name: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
}

impl Message {
    pub fn user<T: Into<MultimodalContent>>(content: Vec<T>) -> Self {
        Message::User(UserMessage::new(content))
    }
    pub fn assistant<T: Into<MultimodalContent>>(content: Vec<T>) -> Self {
        Message::Assistant(AssistantMessage::new(content))
    }
    pub fn content(&self) -> &Vec<MultimodalContent> {
        match self {
            Message::User(msg) => &msg.content,
            Message::Assistant(msg) => &msg.content,
        }
    }
    pub fn push_content<T: Into<MultimodalContent>>(&mut self, content: T) {
        match self {
            Message::User(msg) => msg.content.push(content.into()),
            Message::Assistant(msg) => msg.content.push(content.into()),
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

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Messages(pub Vec<Message>);

impl Messages {
    pub fn push_message(&mut self, message: impl Into<Message>) {
        self.0.push(message.into());
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl From<Message> for Messages {
    fn from(value: Message) -> Self {
        Messages(vec![value])
    }
}

impl From<Vec<Message>> for Messages {
    fn from(value: Vec<Message>) -> Self {
        Messages(value)
    }
}

impl From<UserMessage> for Messages {
    fn from(value: UserMessage) -> Self {
        Messages(vec![Message::User(value)])
    }
}

impl From<AssistantMessage> for Messages {
    fn from(value: AssistantMessage) -> Self {
        Messages(vec![Message::Assistant(value)])
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

impl std::ops::Index<usize> for Messages {
    type Output = Message;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
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
            Role::User => Ok(Message::user(vec![replaced_content])),
            Role::Assistant => Ok(Message::assistant(vec![replaced_content])),
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
    fn test_role_serialization() {
        let user_role = Role::User;
        let assistant_role = Role::Assistant;

        let user_json = serde_json::to_string(&user_role).unwrap();
        let assistant_json = serde_json::to_string(&assistant_role).unwrap();

        assert_eq!(user_json, "\"user\"");
        assert_eq!(assistant_json, "\"assistant\"");
    }

    #[test]
    fn test_role_deserialization() {
        let user_json = "\"user\"";
        let assistant_json = "\"assistant\"";

        let user_role: Role = serde_json::from_str(user_json).unwrap();
        let assistant_role: Role = serde_json::from_str(assistant_json).unwrap();

        assert_eq!(user_role, Role::User);
        assert_eq!(assistant_role, Role::Assistant);
    }

    #[test]
    fn test_text_serialization() {
        let text = Text {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&text).unwrap();
        assert_eq!(json, "{\"text\":\"Hello\"}");
    }

    #[test]
    fn test_text_deserialization() {
        let json = "{\"text\":\"Hello\"}";
        let text: Text = serde_json::from_str(json).unwrap();
        assert_eq!(text.text, "Hello");
    }

    #[test]
    fn test_text_from_string() {
        let text_string = "Hello".to_string();
        let text: Text = text_string.into();
        assert_eq!(text.text, "Hello");
    }

    #[test]
    fn test_message_content_multiple_deserialization() {
        let json =
            "[{\"type\":\"text\",\"text\":\"Hello\"},{\"type\":\"text\",\"text\":\"World\"}]";
        let _multiple_content: Vec<MultimodalContent> = serde_json::from_str(json).unwrap();
    }

    #[test]
    fn test_multimodal_content_text_serialization() {
        let text_content = MultimodalContent::Text("Hello".into());
        let json = serde_json::to_string(&text_content).unwrap();
        assert_eq!(json, "{\"type\":\"text\",\"text\":\"Hello\"}");
    }

    #[test]
    fn test_multimodal_content_text_deserialization() {
        let json = "{\"type\":\"text\",\"text\":\"Hello\"}";
        let text_content: MultimodalContent = serde_json::from_str(json).unwrap();
        assert!(matches!(text_content, MultimodalContent::Text(text) if text.text == "Hello"));
    }

    #[test]
    fn test_multimodal_content_image_serialization() {
        let image_content = MultimodalContent::Image(Image {
            source: ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "base64-data".to_string(),
            },
        });
        let json = serde_json::to_string(&image_content).unwrap();
        assert_eq!(
            json,
            "{\"type\":\"image\",\"source\":{\"type\":\"Base64\",\"media_type\":\"image/png\",\"data\":\"base64-data\"}}"
        );
    }

    #[test]
    fn test_multimodal_content_image_deserialization() {
        let json = "{\"type\":\"image\",\"source\":{\"type\":\"Base64\",\"media_type\":\"image/png\",\"data\":\"base64-data\"}}";
        let image_content: MultimodalContent = serde_json::from_str(json).unwrap();
        assert!(matches!(image_content, MultimodalContent::Image(_)));
    }

    #[test]
    fn test_multimodal_content_from_string() {
        let text = "Hello".to_string();
        let content: MultimodalContent = text.into();
        assert!(matches!(content, MultimodalContent::Text(text) if text.text == "Hello"));
    }

    #[test]
    fn test_image_source_base64_serialization() {
        let base64_source = ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "base64-data".to_string(),
        };
        let json = serde_json::to_string(&base64_source).unwrap();
        assert_eq!(
            json,
            "{\"type\":\"Base64\",\"media_type\":\"image/png\",\"data\":\"base64-data\"}"
        );
    }

    #[test]
    fn test_image_source_base64_deserialization() {
        let json = "{\"type\":\"Base64\",\"media_type\":\"image/png\",\"data\":\"base64-data\"}";
        let base64_source: ImageSource = serde_json::from_str(json).unwrap();
        assert!(matches!(
            base64_source,
            ImageSource::Base64 {
                media_type,
                data
            } if media_type == "image/png" && data == "base64-data"
        ));
    }

    #[test]
    fn test_user_message_serialization() {
        let user_message = UserMessage {
            role: Role::User,
            content: vec![MultimodalContent::Text(Text {
                text: "Hello".to_string(),
            })],
            name: Some("John".to_string()),
        };
        let json = serde_json::to_string(&user_message).unwrap();
        dbg!(&json);
        assert_eq!(
            json,
            "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}],\"name\":\"John\"}"
        );
    }

    #[test]
    fn test_user_message_deserialization() {
        let json = "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}],\"name\":\"John\"}";
        let user_message: UserMessage = serde_json::from_str(json).unwrap();
        assert_eq!(user_message.role, Role::User);
    }

    #[test]
    fn test_user_message_multiple_deserialization() {
        let json = "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}],\"name\":\"John\"}";
        let user_message: UserMessage = serde_json::from_str(json).unwrap();
        dbg!(&user_message);
        assert_eq!(user_message.role, Role::User);
        assert_eq!(user_message.name, Some("John".to_string()));
    }

    #[test]
    fn test_user_message_without_name_serialization() {
        let user_message = UserMessage {
            role: Role::User,
            content: vec![MultimodalContent::Text(Text {
                text: "Hello".to_string(),
            })],
            name: None,
        };
        let json = serde_json::to_string(&user_message).unwrap();
        assert_eq!(
            json,
            "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}]}"
        );
    }

    #[test]
    fn test_user_message_without_name_deserialization() {
        let json = "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}]}";
        let user_message: UserMessage = serde_json::from_str(json).unwrap();
        assert_eq!(user_message.role, Role::User);
        assert_eq!(user_message.name, None);
    }

    #[test]
    fn test_user_message_new() {
        let content = "Hello".to_string();
        let user_message = UserMessage::new(vec![content.clone()]);
        assert_eq!(user_message.role, Role::User);
        assert_eq!(user_message.name, None);
    }

    #[test]
    fn test_user_message_from_string() {
        let content = "Hello".to_string();
        let user_message: UserMessage = content.clone().into();
        assert_eq!(user_message.role, Role::User);
        assert_eq!(user_message.name, None);
    }

    #[test]
    fn test_assistant_message_serialization() {
        let assistant_message = AssistantMessage {
            role: Role::Assistant,
            content: vec![MultimodalContent::Text(Text {
                text: "Hello".to_string(),
            })],
            name: Some("Assistant".to_string()),
        };
        let json = serde_json::to_string(&assistant_message).unwrap();
        assert_eq!(
            json,
            "{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}],\"name\":\"Assistant\"}"
        );
    }

    #[test]
    fn test_assistant_message_deserialization() {
        let json =
            "{\"role\":\"assistant\",\"content\":[{\"type\": \"text\", \"text\":\"Hello\"}],\"name\":\"Assistant\"}";
        let assistant_message: AssistantMessage = serde_json::from_str(json).unwrap();
        dbg!(&assistant_message);
        assert_eq!(assistant_message.role, Role::Assistant);
        assert_eq!(assistant_message.name, Some("Assistant".to_string()));
    }

    #[test]
    fn test_assistant_message_without_name_serialization() {
        let assistant_message = AssistantMessage {
            role: Role::Assistant,
            content: vec![MultimodalContent::Text(Text {
                text: "Hello".to_string(),
            })],
            name: None,
        };
        let json = serde_json::to_string(&assistant_message).unwrap();
        assert_eq!(
            json,
            "{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}]}"
        );
    }

    #[test]
    fn test_assistant_message_without_name_deserialization() {
        let json = "{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}]}";
        let assistant_message: AssistantMessage = serde_json::from_str(json).unwrap();
        assert_eq!(assistant_message.role, Role::Assistant);
        assert_eq!(assistant_message.name, None);
    }

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

    #[test]
    fn test_multimodal_content_tool_use_serialization() {
        let tool_use = ToolUse {
            id: "test_id".to_string(),
            name: "test_name".to_string(),
            input: serde_json::json!({
                "key": "value"
            }),
        };

        let multimodal_content = MultimodalContent::ToolUse(tool_use.clone());

        let serialized = serde_json::to_string(&multimodal_content).unwrap();
        let deserialized: MultimodalContent = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, multimodal_content);

        if let MultimodalContent::ToolUse(deserialized_tool_use) = deserialized {
            assert_eq!(deserialized_tool_use.id, tool_use.id);
            assert_eq!(deserialized_tool_use.name, tool_use.name);
            assert_eq!(deserialized_tool_use.input, tool_use.input);
        } else {
            panic!("Deserialized content is not ToolUse variant");
        }
    }

    #[test]
    fn test_multimodal_content_tool_use_type() {
        let tool_use = ToolUse {
            id: "test_id".to_string(),
            name: "test_name".to_string(),
            input: serde_json::json!({
                "key": "value"
            }),
        };

        let multimodal_content = MultimodalContent::ToolUse(tool_use);

        let serialized = serde_json::to_string(&multimodal_content).unwrap();
        dbg!(&serialized);

        assert!(serialized.contains(r#""type":"tool_use""#));
    }
}
