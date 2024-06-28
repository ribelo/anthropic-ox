use async_trait::async_trait;
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    future::Future,
    marker::PhantomData,
    sync::Arc,
};

use super::message::MultimodalContent;

#[derive(Default)]
pub struct ToolBuilder {
    name: Option<String>,
    description: Option<String>,
    input_schema: Option<serde_json::Value>,
    handler: Option<Arc<dyn ErasedToolHandler>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolBuilderError {
    #[error("Tool name is required")]
    MissingName,
    #[error("Tool parameters is required")]
    MissingParameters,
}

impl ToolBuilder {
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
    pub fn with_handler<H, T, R>(mut self, handler: H) -> Self
    where
        H: ToolHandler<T, R> + Send + Sync + 'static,
        T: JsonSchema + DeserializeOwned + Send + Sync + 'static,
        R: Serialize + Send + Sync + 'static,
    {
        let mut settings = schemars::gen::SchemaSettings::draft07();
        settings.inline_subschemas = true;
        let gen = schemars::gen::SchemaGenerator::new(settings);
        let json_schema = gen.into_root_schema_for::<T>();
        let mut input_schema = serde_json::to_value(json_schema).unwrap();
        input_schema["type"] = serde_json::json!("object");
        self.input_schema = Some(input_schema);

        let wrapper = ToolHandlerWrapper::<H, T, R> {
            handler,
            phantom: PhantomData,
        };
        self.handler = Some(Arc::new(wrapper));

        self
    }

    #[must_use]
    pub fn with_props<T>(mut self) -> Self
    where
        T: JsonSchema + DeserializeOwned + Send + Sync + 'static,
    {
        let mut settings = schemars::gen::SchemaSettings::draft07();
        settings.inline_subschemas = true;
        let gen = schemars::gen::SchemaGenerator::new(settings);
        let json_schema = gen.into_root_schema_for::<T>();
        self.input_schema = Some(serde_json::to_value(json_schema).unwrap());

        self
    }

    /// Consumes the builder, returning a [`Tool`] if all required fields have been set.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` or `description` are not set.
    pub fn build(self) -> Result<Tool, ToolBuilderError> {
        let name = self.name.ok_or(ToolBuilderError::MissingName)?;
        let input_schema = self
            .input_schema
            .ok_or(ToolBuilderError::MissingParameters)?;

        Ok(Tool {
            name,
            description: self.description,
            input_schema,
            handler: self.handler,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    input_schema: serde_json::Value,
    #[serde(skip)]
    handler: Option<Arc<dyn ErasedToolHandler>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Tool handler not set")]
    HandlerNotSet,
    #[error("Failed to execute tool: {0}")]
    ExecutionFailed(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Failed to deserialize input: {0}")]
    InputDeserializationFailed(String),
    #[error("Failed to serialize output: {0}")]
    OutputSerializationFailed(String),
    #[error("Failed to invoke tool: {0}")]
    InvokeFailed(String),
}

impl Tool {
    pub fn builder() -> ToolBuilder {
        ToolBuilder::default()
    }
    pub async fn invoke<T: Into<String>>(
        &self,
        id: T,
        input: serde_json::Value,
        context: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let handler = self.handler.as_ref().ok_or(ToolError::HandlerNotSet)?;

        match handler.call(input, context).await {
            Ok(content) => Ok(ToolResult::new(id, content)),
            Err(err) => Ok(ToolResult::error(id, err)),
        }
    }
}

impl PartialEq for Tool {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    resources: Arc<Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
}

impl ToolContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_resource<T: Any + Send + Sync + Clone>(&mut self, resource: T) {
        self.resources
            .lock()
            .insert(TypeId::of::<T>(), Box::new(resource));
    }

    pub fn get_resource<T: Any + Send + Sync + Clone>(&self) -> Option<T> {
        self.resources
            .lock()
            .get(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_ref::<T>())
            .cloned()
    }

    pub fn expect_resource<T: Any + Send + Sync + Clone>(&self) -> T {
        self.get_resource::<T>()
            .unwrap_or_else(|| panic!("Resource of type {} not found", std::any::type_name::<T>()))
    }

    pub fn remove_resource<T: Any + Send + Sync>(&mut self) -> Option<T> {
        self.resources
            .lock()
            .remove(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    pub fn len(&self) -> usize {
        self.resources.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.resources.lock().is_empty()
    }
}

impl std::fmt::Display for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ToolsContext {{ resources: {} }}", self.len())
    }
}

#[derive(Debug, Clone, Default)]
pub struct Tools {
    tools: HashMap<String, Tool>,
    context: ToolContext,
}

impl Extend<Tool> for Tools {
    fn extend<T: IntoIterator<Item = Tool>>(&mut self, iter: T) {
        for tool in iter {
            let name = tool.name.clone();
            self.tools.insert(name, tool);
        }
    }
}

impl Serialize for Tools {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        let tools = self.tools.values().collect::<Vec<_>>();
        tools.serialize(serializer)
    }
}

impl Tools {
    pub fn new() -> Self {
        Tools::default()
    }

    pub fn set_context(&mut self, cx: ToolContext) {
        self.context = cx
    }

    pub fn add_tool(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn get_tool(&self, tool_name: &str) -> Option<&Tool> {
        self.tools.get(tool_name)
    }

    pub fn add_resource<T: Any + Send + Sync + Clone>(&mut self, resource: T) {
        self.context.add_resource(resource);
    }

    pub fn get_resource<T: Any + Send + Sync + Clone>(&self) -> Option<T> {
        self.context.get_resource()
    }

    pub fn expect_resource<T: Any + Send + Sync + Clone>(&self) -> T {
        self.context.expect_resource()
    }

    pub async fn invoke(&self, tool_use: ToolUse) -> Result<ToolResult, ToolError> {
        let tool = self
            .get_tool(&tool_use.name)
            .ok_or_else(|| ToolError::ToolNotFound(tool_use.name.clone()))?;

        tool.invoke(tool_use.id, tool_use.input, self.context.clone())
            .await
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Tool> {
        self.tools.values()
    }
}

impl FromIterator<Tool> for Tools {
    fn from_iter<I: IntoIterator<Item = Tool>>(iter: I) -> Self {
        let mut tools = Tools::new();
        for tool in iter {
            tools.add_tool(tool);
        }
        tools
    }
}

impl From<Tool> for Tools {
    fn from(tool: Tool) -> Self {
        let mut tools = Tools::default();
        tools.add_tool(tool);
        tools
    }
}

impl From<Vec<Tool>> for Tools {
    fn from(tools: Vec<Tool>) -> Self {
        tools.into_iter().collect()
    }
}

impl IntoIterator for Tools {
    type Item = Tool;
    type IntoIter = std::collections::hash_map::IntoValues<String, Tool>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.into_values()
    }
}

impl<'a> IntoIterator for &'a Tools {
    type Item = &'a Tool;
    type IntoIter = std::collections::hash_map::Values<'a, String, Tool>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.values()
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

#[async_trait]
pub trait ToolHandler<T, R> {
    async fn call(&self, input: T, cx: ToolContext) -> Result<R, ToolError>;
}

#[async_trait]
impl<T, R, F, Fut> ToolHandler<T, R> for F
where
    T: Send + 'static,
    R: Serialize + Send + 'static,
    F: Fn(T, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<R, ToolError>> + Send,
{
    async fn call(&self, input: T, cx: ToolContext) -> Result<R, ToolError> {
        (self)(input, cx).await
    }
}

#[async_trait]
pub trait ErasedToolHandler: Send + Sync {
    async fn call(
        &self,
        input: serde_json::Value,
        cx: ToolContext,
    ) -> Result<serde_json::Value, ToolError>;
}

impl fmt::Debug for dyn ErasedToolHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ErasedToolHandler").finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct ToolHandlerWrapper<H, T, R>
where
    H: ToolHandler<T, R>,
{
    pub handler: H,
    pub phantom: PhantomData<(T, R)>,
}

#[async_trait]
impl<H, T, R> ErasedToolHandler for ToolHandlerWrapper<H, T, R>
where
    H: ToolHandler<T, R> + Send + Sync,
    T: JsonSchema + DeserializeOwned + Send + Sync,
    R: Serialize + Send + Sync,
{
    async fn call(
        &self,
        input: serde_json::Value,
        cx: ToolContext,
    ) -> Result<serde_json::Value, ToolError> {
        let props: T = serde_json::from_value(input)
            .map_err(|e| ToolError::InputDeserializationFailed(e.to_string()))?;
        let result = self.handler.call(props, cx).await?;
        serde_json::to_value(result)
            .map_err(|e| ToolError::OutputSerializationFailed(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct Props {
        name: String,
        email: String,
        age: i32,
    }

    async fn empty_handler(
        _input: Props,
        _cx: ToolContext,
    ) -> Result<serde_json::Value, ToolError> {
        Ok(json!({}))
    }

    #[tokio::test]
    async fn test_tool_empty_handler() {
        let tool = Tool::builder()
            .with_name("user_create")
            .with_handler(empty_handler)
            .build()
            .unwrap();

        let context = ToolContext::default();

        let input = serde_json::json!({
            "name": "John Doe",
            "email": "john.doe@email.com",
            "age": 30,
        });

        let result = tool.invoke("id", input, context).await.unwrap();
        assert_eq!(result.id, "id");
        assert_eq!(result.content[0].as_json().unwrap(), json!({}));
    }

    #[tokio::test]
    async fn test_tool_handler() {
        async fn handler(
            input: serde_json::Value,
            _cx: ToolContext,
        ) -> Result<serde_json::Value, ToolError> {
            Ok(json!({
            "name": input["name"],
            "email": input["email"],
            "age": input["age"]
            }))
        }

        let tool = Tool::builder()
            .with_name("user_create")
            .with_handler(handler)
            .build()
            .unwrap();

        let context = ToolContext::default();

        let input = serde_json::json!({
            "name": "John Doe",
            "email": 123,
            "age": "30"
        });

        let result = tool
            .invoke("id".to_owned(), input.clone(), context)
            .await
            .unwrap();
        assert_eq!(result.id, "id");
        assert_eq!(result.content[0].as_json().unwrap(), input);
    }

    #[test]
    fn test_add_and_get_resource_primitive_type() {
        let mut context = ToolContext::default();
        let value: i32 = 42;
        context.add_resource(value);

        let retrieved_value = context.get_resource::<i32>().unwrap();
        assert_eq!(retrieved_value, value);
    }

    #[derive(Debug, Clone, PartialEq)]
    struct TestStruct {
        field1: String,
        field2: i32,
    }

    #[test]
    fn test_add_and_get_resource_custom_struct() {
        let mut context = ToolContext::default();
        let test_struct = TestStruct {
            field1: "test".to_string(),
            field2: 42,
        };
        context.add_resource(test_struct.clone());

        let retrieved_struct = context.get_resource::<TestStruct>().unwrap();
        assert_eq!(retrieved_struct, test_struct);
    }

    #[test]
    fn test_get_resource_not_found() {
        let context = ToolContext::default();
        let retrieved_value = context.get_resource::<i32>();
        assert!(retrieved_value.is_none());
    }

    struct TestFromContext {
        value: i32,
    }

    #[tokio::test]
    async fn test_tool_handler_with_context() {
        async fn handler(
            input: serde_json::Value,
            cx: ToolContext,
        ) -> Result<serde_json::Value, ToolError> {
            let salary = cx.expect_resource::<i32>();
            Ok(json!({
                "name": input["name"],
                "email": input["email"],
                "age": input["age"],
                "salary": salary
            }))
        }

        let tool = Tool::builder()
            .with_name("user_create")
            .with_handler(handler)
            .build()
            .unwrap();

        let mut context = ToolContext::default();
        context.add_resource(1000_i32);

        let input = serde_json::json!({
            "name": "John Doe",
            "email": 123,
            "age": "30"
        });

        let result = tool
            .invoke("id".to_owned(), input.clone(), context)
            .await
            .unwrap();
        assert_eq!(result.id, "id");
        assert_eq!(result.content[0].as_json().unwrap()["salary"], 1000_i32);
    }
}
