use std::{fmt, path::Path};

use base64::Engine;
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use super::tool::{ToolResult, ToolUse};

use strum::{Display, EnumString};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Display, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
}

impl ImageSource {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;
        let base64_data = base64::engine::general_purpose::STANDARD.encode(data);
        let media_type = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();

        Ok(ImageSource::Base64 {
            media_type,
            data: base64_data,
        })
    }
}

impl fmt::Display for ImageSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageSource::Base64 { media_type, data } => {
                let truncated_data = if data.len() > 20 {
                    format!("{}...", &data[..20])
                } else {
                    data.clone()
                };
                write!(f, "Base64 ({}, {})", media_type, truncated_data)
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Image {
    pub source: ImageSource,
}

impl Image {
    pub fn new(source: ImageSource) -> Self {
        Self { source }
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let source = ImageSource::from_path(path)?;
        Ok(Self::new(source))
    }

    pub fn from_base64(media_type: String, data: String) -> Self {
        let source = ImageSource::Base64 { media_type, data };
        Self::new(source)
    }
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Image: {}", self.source)
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Text {
    pub text: String,
}

impl Text {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn push_str(&mut self, string: &str) {
        self.text.push_str(string);
    }

    pub fn to_lowercase(&self) -> Self {
        Self {
            text: self.text.to_lowercase(),
        }
    }

    pub fn to_uppercase(&self) -> Self {
        Self {
            text: self.text.to_uppercase(),
        }
    }
}

impl From<String> for Text {
    fn from(text: String) -> Self {
        Text { text }
    }
}

impl From<&str> for Text {
    fn from(text: &str) -> Self {
        Text {
            text: text.to_owned(),
        }
    }
}

impl From<&String> for Text {
    fn from(text: &String) -> Self {
        Text { text: text.clone() }
    }
}

impl From<Box<str>> for Text {
    fn from(text: Box<str>) -> Self {
        Text {
            text: text.into_string(),
        }
    }
}

impl From<std::borrow::Cow<'_, str>> for Text {
    fn from(text: std::borrow::Cow<'_, str>) -> Self {
        Text {
            text: text.into_owned(),
        }
    }
}

impl From<serde_json::Value> for Text {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(s) => Text { text: s },
            _ => Text {
                text: value.to_string(),
            },
        }
    }
}

impl From<Text> for String {
    fn from(text: Text) -> Self {
        text.text
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

impl MultimodalContent {
    pub fn text<T: Into<String>>(text: T) -> Self {
        Self::Text(Text { text: text.into() })
    }

    pub fn image(source: ImageSource) -> Self {
        Self::Image(Image { source })
    }

    pub fn tool_use(tool_use: ToolUse) -> Self {
        Self::ToolUse(tool_use)
    }

    pub fn tool_result(tool_result: ToolResult) -> Self {
        Self::ToolResult(tool_result)
    }

    pub fn as_text(&self) -> Option<&Text> {
        if let Self::Text(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_image(&self) -> Option<&Image> {
        if let Self::Image(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_tool_use(&self) -> Option<&ToolUse> {
        if let Self::ToolUse(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResult> {
        if let Self::ToolResult(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_json(&self) -> Option<serde_json::Value> {
        match self {
            Self::Text(text) => serde_json::from_str(&text.text).ok(),
            _ => None,
        }
    }
}

impl<T: Into<Text>> From<T> for MultimodalContent {
    fn from(text: T) -> Self {
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
            Self::Text(text) => fmt::Display::fmt(text, f),
            Self::Image(image) => fmt::Display::fmt(image, f),
            Self::ToolUse(tool_use) => fmt::Display::fmt(tool_use, f),
            Self::ToolResult(tool_result) => fmt::Display::fmt(tool_result, f),
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

    pub fn add_content<T: Into<MultimodalContent>>(&mut self, content: T) {
        self.content.push(content.into());
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }
}

impl fmt::Display for UserMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{}: ", name)?;
        }
        for (i, content) in self.content.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", content)?;
        }
        Ok(())
    }
}

impl<T: Into<MultimodalContent>> From<T> for UserMessage {
    fn from(content: T) -> Self {
        Self::new(vec![content])
    }
}

impl From<Vec<MultimodalContent>> for UserMessage {
    fn from(content: Vec<MultimodalContent>) -> Self {
        Self::new(content)
    }
}

impl From<UserMessage> for Vec<MultimodalContent> {
    fn from(message: UserMessage) -> Self {
        message.content
    }
}

#[derive(Debug, Serialize, Deserialize, Derivative, Clone, PartialEq, Eq)]
pub struct AssistantMessage {
    #[derivative(Default(value = "Role::Assistant"))]
    pub role: Role,
    pub content: Vec<MultimodalContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl AssistantMessage {
    pub fn new<T: Into<MultimodalContent>>(content: Vec<T>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into_iter().map(Into::into).collect(),
            name: None,
            id: None,
            model: None,
        }
    }

    pub fn add_content<T: Into<MultimodalContent>>(&mut self, content: T) {
        self.content.push(content.into());
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn len(&self) -> usize {
        self.content.len()
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

impl fmt::Display for AssistantMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{}: ", name)?;
        }
        for (i, content) in self.content.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", content)?;
        }
        Ok(())
    }
}

impl<T: Into<MultimodalContent>> From<T> for AssistantMessage {
    fn from(content: T) -> Self {
        Self::new(vec![content])
    }
}

impl From<Vec<MultimodalContent>> for AssistantMessage {
    fn from(content: Vec<MultimodalContent>) -> Self {
        Self::new(content)
    }
}

impl From<AssistantMessage> for Vec<MultimodalContent> {
    fn from(message: AssistantMessage) -> Self {
        message.content
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

    pub fn content(&self) -> &[MultimodalContent] {
        match self {
            Message::User(msg) => &msg.content,
            Message::Assistant(msg) => &msg.content,
        }
    }

    pub fn add_content<T: Into<MultimodalContent>>(&mut self, content: T) {
        match self {
            Message::User(msg) => msg.content.push(content.into()),
            Message::Assistant(msg) => msg.content.push(content.into()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.content().is_empty()
    }

    pub fn len(&self) -> usize {
        self.content().len()
    }

    pub fn role(&self) -> Role {
        match self {
            Message::User(_) => Role::User,
            Message::Assistant(_) => Role::Assistant,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Message::User(msg) => msg.name.as_deref(),
            Message::Assistant(msg) => msg.name.as_deref(),
        }
    }

    pub fn as_user(&self) -> Option<&UserMessage> {
        match self {
            Message::User(msg) => Some(msg),
            _ => None,
        }
    }

    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            Message::Assistant(msg) => Some(msg),
            _ => None,
        }
    }

    pub fn expect_user(&self) -> &UserMessage {
        self.as_user().expect("User message")
    }

    pub fn expect_assistant(&self) -> &AssistantMessage {
        self.as_assistant().expect("Assistant message")
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Message::User(msg) => write!(f, "User: {}", msg),
            Message::Assistant(msg) => write!(f, "Assistant: {}", msg),
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

impl<T: Into<MultimodalContent>> From<T> for Message {
    fn from(content: T) -> Self {
        Message::User(UserMessage::from(content))
    }
}

impl From<Vec<MultimodalContent>> for Message {
    fn from(content: Vec<MultimodalContent>) -> Self {
        Message::User(UserMessage::from(content))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Messages(pub Vec<Message>);

impl Messages {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn with_message<T: Into<Message>>(mut self, message: T) -> Self {
        self.add_message(message);
        self
    }

    pub fn add_message<T: Into<Message>>(&mut self, message: T) {
        self.0.push(message.into());
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Message> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Message> {
        self.0.iter_mut()
    }

    pub fn last(&self) -> Option<&Message> {
        self.0.last()
    }

    pub fn last_mut(&mut self) -> Option<&mut Message> {
        self.0.last_mut()
    }
}

impl From<Message> for Messages {
    fn from(value: Message) -> Self {
        Messages(vec![value])
    }
}

impl From<AssistantMessage> for Messages {
    fn from(value: AssistantMessage) -> Self {
        Messages(vec![Message::Assistant(value)])
    }
}

impl<T> From<Vec<T>> for Messages
where
    T: Into<Message>,
{
    fn from(value: Vec<T>) -> Self {
        Messages(value.into_iter().map(Into::into).collect())
    }
}

impl std::ops::Index<usize> for Messages {
    type Output = Message;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Messages {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for Messages {
    type Item = Message;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Messages {
    type Item = &'a Message;
    type IntoIter = std::slice::Iter<'a, Message>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut Messages {
    type Item = &'a mut Message;
    type IntoIter = std::slice::IterMut<'a, Message>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            id: None,
            model: None,
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
            id: None,
            model: None,
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
