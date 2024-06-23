use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    marker::PhantomData,
    sync::Arc,
};

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
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    #[must_use]
    pub fn handler<H, T, R>(mut self, handler: H) -> Self
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
    pub fn props<T>(mut self) -> Self
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
}

impl Tool {
    pub fn builder() -> ToolBuilder {
        ToolBuilder::default()
    }
    pub fn invoke<T: Into<String>>(
        &self,
        id: T,
        input: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let handler = self.handler.as_ref().ok_or(ToolError::HandlerNotSet)?;

        match handler.call(input, context) {
            Ok(content) => Ok(ToolResult {
                id: id.into(),
                content,
                is_error: false,
            }),
            Err(err) => Ok(ToolResult {
                id: id.into(),
                content: serde_json::json!(err.to_string()),
                is_error: true,
            }),
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
    resources: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl ToolContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_resource<T: Any + Send + Sync + Clone>(&mut self, resource: T) {
        self.resources.insert(TypeId::of::<T>(), Arc::new(resource));
    }

    pub fn with_resource<T: Any + Send + Sync + Clone>(mut self, resource: T) -> Self {
        self.add_resource(resource);
        self
    }

    pub fn get_resource<T: Any + Send + Sync + Clone>(&self) -> Option<T> {
        let boxed_resource = self.resources.get(&TypeId::of::<T>())?;
        Some(boxed_resource.downcast_ref::<T>().unwrap().clone())
    }

    pub fn expect_resource<T: Any + Send + Sync + Clone>(&self) -> T {
        self.get_resource::<T>()
            .unwrap_or_else(|| panic!("Resource of type {} not found", std::any::type_name::<T>()))
    }

    pub fn remove_resource<T: Any + Send + Sync>(&mut self) -> Option<Arc<T>> {
        self.resources
            .remove(&TypeId::of::<T>())
            .and_then(|r| r.downcast::<T>().ok())
    }

    pub fn len(&self) -> usize {
        self.resources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
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

    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.add_tool(tool);
        self
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

    pub fn invoke(&self, tool_use: ToolUse) -> Result<ToolResult, ToolError> {
        self.get_tool(&tool_use.name)
            .ok_or(ToolError::ToolNotFound(tool_use.name.clone()))
            .and_then(|tool| tool.invoke(tool_use.id, tool_use.input, &self.context))
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
        Tools::new().with_tool(tool)
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
    pub content: serde_json::Value,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

impl ToolResult {
    pub fn new(id: String, content: serde_json::Value) -> Self {
        Self {
            id,
            content,
            is_error: false,
        }
    }

    pub fn error(id: String, error_content: serde_json::Value) -> Self {
        Self {
            id,
            content: error_content,
            is_error: true,
        }
    }

    pub fn is_success(&self) -> bool {
        !self.is_error
    }

    pub fn content(&self) -> &serde_json::Value {
        &self.content
    }

    pub fn deserialize_content<T: serde::de::DeserializeOwned>(
        &self,
    ) -> Result<T, serde_json::Error> {
        serde_json::from_value(self.content.clone())
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

pub trait ToolHandler<T, R> {
    fn call(&self, input: T, cx: &ToolContext) -> Result<R, ToolError>;
}

impl<T, R, F> ToolHandler<T, R> for F
where
    F: Fn(T, &ToolContext) -> Result<R, ToolError>,
    R: Serialize,
{
    fn call(&self, input: T, cx: &ToolContext) -> Result<R, ToolError> {
        (self)(input, cx)
    }
}

pub trait ErasedToolHandler: Send + Sync {
    fn call(
        &self,
        input: serde_json::Value,
        cx: &ToolContext,
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

impl<H, T, R> ErasedToolHandler for ToolHandlerWrapper<H, T, R>
where
    H: ToolHandler<T, R> + Send + Sync,
    T: JsonSchema + DeserializeOwned + Send + Sync,
    R: Serialize + Send + Sync,
{
    fn call(
        &self,
        input: serde_json::Value,
        cx: &ToolContext,
    ) -> Result<serde_json::Value, ToolError> {
        let props: T = serde_json::from_value(input)
            .map_err(|e| ToolError::InputDeserializationFailed(e.to_string()))?;
        let result = self.handler.call(props, &cx)?;
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

    fn empty_handler(_input: Props, _cx: &ToolContext) -> Result<serde_json::Value, ToolError> {
        Ok(json!({}))
    }

    #[test]
    fn test_tool_empty_handler() {
        let tool = Tool::builder()
            .name("user_create")
            .handler(empty_handler)
            .build()
            .unwrap();

        let context = ToolContext::default();

        let input = serde_json::json!({
            "name": "John Doe",
            "email": "john.doe@email.com",
            "age": 30,
        });

        let result = tool.invoke("id", input, &context).unwrap();
        assert_eq!(result.id, "id");
        assert_eq!(result.content, json!({}));
    }

    #[test]
    fn test_tool_handler() {
        fn handler(
            input: serde_json::Value,
            _cx: &ToolContext,
        ) -> Result<serde_json::Value, ToolError> {
            Ok(json!({
            "name": input["name"],
            "email": input["email"],
            "age": input["age"]
            }))
        }

        let tool = Tool::builder()
            .name("user_create")
            .handler(handler)
            .build()
            .unwrap();

        let context = ToolContext::default();

        let input = serde_json::json!({
            "name": "John Doe",
            "email": 123,
            "age": "30"
        });

        let result = tool
            .invoke("id".to_owned(), input.clone(), &context)
            .unwrap();
        assert_eq!(result.id, "id");
        assert_eq!(result.content, input);
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

    #[test]
    fn test_tool_handler_with_context() {
        fn handler(
            input: serde_json::Value,
            cx: &ToolContext,
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
            .name("user_create")
            .handler(handler)
            .build()
            .unwrap();

        let context = ToolContext::default().with_resource(1000_i32);

        let input = serde_json::json!({
            "name": "John Doe",
            "email": 123,
            "age": "30"
        });

        let result = tool
            .invoke("id".to_owned(), input.clone(), &context)
            .unwrap();
        assert_eq!(result.id, "id");
        assert_eq!(result.content["salary"], 1000_i32);
    }
}
