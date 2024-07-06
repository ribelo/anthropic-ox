use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fmt,
    sync::{Arc, RwLock},
};

use super::message::MultimodalContent;

#[async_trait]
pub trait AnyTool {
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str>;
    async fn invoke_any(&self, input: Value) -> Result<Value, ToolError>;
    fn input_schema(&self) -> Value;
}

#[async_trait]
pub trait Tool: Send + Sync {
    type Input: JsonSchema + DeserializeOwned + Send + Sync;
    type Output: Serialize + Send + Sync;
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str> {
        None
    }
    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError>;
    fn input_schema(&self) -> Value {
        let mut settings = schemars::gen::SchemaSettings::draft07();
        settings.inline_subschemas = true;
        let gen = schemars::gen::SchemaGenerator::new(settings);
        let json_schema = gen.into_root_schema_for::<Self::Input>();
        let mut input_schema = serde_json::to_value(json_schema).unwrap();
        input_schema["type"] = serde_json::json!("object");
        input_schema
    }
}

#[async_trait]
impl<T: Tool + Send + Sync> AnyTool for T {
    fn name(&self) -> &str {
        self.name()
    }

    fn description(&self) -> Option<&str> {
        self.description()
    }

    async fn invoke_any(&self, input: Value) -> Result<Value, ToolError> {
        let typed_input: T::Input = serde_json::from_value(input)
            .map_err(|e| ToolError::InputDeserializationFailed(e.to_string()))?;
        let output = self.invoke(typed_input).await?;
        serde_json::to_value(output)
            .map_err(|e| ToolError::OutputSerializationFailed(e.to_string()))
    }

    fn input_schema(&self) -> Value {
        self.input_schema()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadataInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Clone, Default)]
pub struct ToolBox {
    tools: Arc<RwLock<std::collections::HashMap<String, Arc<dyn AnyTool>>>>,
}

impl fmt::Debug for ToolBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tools = self.tools.read().map_err(|_| fmt::Error)?;
        f.debug_struct("ToolBox")
            .field("tools", &format!("HashMap with {} entries", tools.len()))
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Failed to execute tool: {0}")]
    ExecutionFailed(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Failed to deserialize input: {0}")]
    InputDeserializationFailed(String),
    #[error("Failed to serialize output: {0}")]
    OutputSerializationFailed(String),
    #[error("Failed to generate input schema: {0}")]
    SchemaGenerationFailed(String),
}

impl ToolBox {
    pub fn add<T: Tool + 'static>(&self, tool: T) {
        let name = tool.name().to_string();
        self.tools.write().unwrap().insert(name, Arc::new(tool));
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn AnyTool>> {
        self.tools.read().unwrap().get(name).cloned()
    }

    pub async fn invoke(&self, tool_use: ToolUse) -> ToolResult {
        match self.get(&tool_use.name) {
            Some(tool) => match tool.invoke_any(tool_use.input).await {
                Ok(result) => ToolResult::new(tool_use.id, result),
                Err(error) => ToolResult::error(tool_use.id, error),
            },
            None => ToolResult::error(tool_use.id, ToolError::ToolNotFound(tool_use.name)),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.read().unwrap().is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.read().unwrap().len()
    }

    #[must_use]
    pub fn metadata(&self) -> Vec<ToolMetadataInfo> {
        self.tools
            .read()
            .unwrap()
            .values()
            .map(|tool| ToolMetadataInfo {
                name: tool.name().to_string(),
                description: tool.description().map(std::string::ToString::to_string),
                input_schema: tool.input_schema(),
            })
            .collect()
    }
}

impl Serialize for ToolBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        self.metadata().serialize(serializer)
    }
}

impl<T: Tool + 'static> From<Vec<T>> for ToolBox {
    fn from(tools: Vec<T>) -> Self {
        let toolbox = ToolBox::default();
        for tool in tools {
            toolbox.add(tool);
        }
        toolbox
    }
}

impl FromIterator<Arc<dyn AnyTool>> for ToolBox {
    fn from_iter<I: IntoIterator<Item = Arc<dyn AnyTool>>>(iter: I) -> Self {
        let toolbox = ToolBox::default();
        for tool in iter {
            toolbox
                .tools
                .write()
                .unwrap()
                .insert(tool.name().to_string(), tool);
        }
        toolbox
    }
}

#[derive(Debug, Clone)]
pub struct ToolUseBuilder {
    id: String,
    name: String,
    input: String,
}

impl ToolUseBuilder {
    pub fn new<T: Into<String>>(id: T, name: T) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input: String::new(),
        }
    }

    pub fn push_str(&mut self, s: &str) -> &mut Self {
        self.input.push_str(s);
        self
    }

    pub fn build(self) -> Result<ToolUse, serde_json::Error> {
        Ok(ToolUse {
            id: self.id,
            name: self.name,
            input: if self.input.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&self.input)?
            },
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

impl ToolUse {
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    pub fn deserialize_input<T: serde::de::DeserializeOwned>(
        &self,
    ) -> Result<T, serde_json::Error> {
        serde_json::from_value(self.input.clone())
    }

    pub fn is_empty(&self) -> bool {
        self.input.is_null()
    }
}

impl std::fmt::Display for ToolUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ToolUse {{ id: {}, name: {}, input: ... }}",
            self.id, self.name
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResult {
    #[serde(rename = "tool_use_id")]
    pub id: String,
    pub content: Vec<MultimodalContent>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

impl ToolResult {
    pub fn new<T: Into<String>, U: Into<MultimodalContent>>(id: T, content: U) -> Self {
        Self {
            id: id.into(),
            content: vec![content.into()],
            is_error: false,
        }
    }

    pub fn add_content<T: Into<MultimodalContent>>(&mut self, content: T) {
        self.content.push(content.into());
    }

    pub fn error<T: Into<String>>(id: T, error: ToolError) -> Self {
        Self {
            id: id.into(),
            content: vec![error.to_string().into()],
            is_error: true,
        }
    }

    pub fn is_success(&self) -> bool {
        !self.is_error
    }

    pub fn content(&self) -> &[MultimodalContent] {
        &self.content
    }

    pub fn into_content(self) -> Vec<MultimodalContent> {
        self.content
    }
}

impl std::fmt::Display for ToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ToolResult {{ id: {}, content: ..., is_error: {} }}",
            self.id, self.is_error
        )
    }
}
