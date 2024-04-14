use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    future::Future,
    marker::PhantomData,
    sync::Arc,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ValueType {
    String,
    Number,
    Boolean,
    Array,
    Object,
}

#[derive(Debug, Clone, Serialize)]
pub struct Property {
    name: String,
    #[serde(rename = "type")]
    value_type: ValueType,
    description: Option<String>,
    #[serde(skip)]
    required: bool,
}

impl Property {
    pub fn new(name: &str) -> Self {
        Property {
            name: name.to_string(),
            value_type: ValueType::String,
            description: None,
            required: false,
        }
    }

    pub fn value_type(mut self, value_type: ValueType) -> Self {
        self.value_type = value_type;
        self
    }

    pub fn string_type(mut self) -> Self {
        self.value_type = ValueType::String;
        self
    }

    pub fn number_type(mut self) -> Self {
        self.value_type = ValueType::Number;
        self
    }

    pub fn boolean_type(mut self) -> Self {
        self.value_type = ValueType::Boolean;
        self
    }

    pub fn array_type(mut self) -> Self {
        self.value_type = ValueType::Array;
        self
    }

    pub fn object_type(mut self) -> Self {
        self.value_type = ValueType::Object;
        self
    }

    pub fn description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct InputSchema {
    #[serde(rename = "type")]
    value_type: ValueType,
    properties: HashMap<String, Property>,
    required: Vec<String>,
}

impl Default for InputSchema {
    fn default() -> Self {
        InputSchema {
            value_type: ValueType::Object,
            properties: HashMap::new(),
            required: Vec::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Missing property: {0}")]
    MissingProperty(String),
    #[error("Invalid property type: {0}")]
    InvalidPropertyType(String),
}

#[derive(Debug)]
pub struct ToolBuilder {
    name: String,
    description: Option<String>,
    input_schema: InputSchema,
    handler: Option<Arc<dyn ToolHandlerFn>>,
}

#[derive(Debug, thiserror::Error)]
pub enum BuildToolError {
    #[error("Tool handler is missing")]
    MissingHandler,
}

impl ToolBuilder {
    pub fn new(name: &str) -> Self {
        ToolBuilder {
            name: name.to_string(),
            description: None,
            input_schema: InputSchema::default(),
            handler: None,
        }
    }

    pub fn description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    pub fn add_property(mut self, property: Property) -> Self {
        if property.required {
            self.input_schema.required.push(property.name.clone());
        }

        self.input_schema
            .properties
            .insert(property.name.clone(), property);

        self
    }

    pub fn handler<H, T>(mut self, handler: H) -> Self
    where
        H: ToolHandler<T> + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        let wrapper = ToolHandlerWrapper {
            handler,
            phantom: PhantomData,
        };
        self.handler = Some(Arc::new(wrapper));
        self
    }

    pub fn build(self) -> Result<Tool, BuildToolError> {
        let handler = self.handler.ok_or(BuildToolError::MissingHandler)?;

        Ok(Tool {
            name: self.name,
            description: self.description,
            input_schema: self.input_schema,
            handler,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    name: String,
    description: Option<String>,
    input_schema: InputSchema,
    #[serde(skip)]
    handler: Arc<dyn ToolHandlerFn>,
}

impl Tool {
    pub fn validate_input(&self, input: &serde_json::Value) -> Result<(), ValidationError> {
        if let Some(input_obj) = input.as_object() {
            for (name, property) in &self.input_schema.properties {
                if property.required && !input_obj.contains_key(name) {
                    return Err(ValidationError::MissingProperty(name.to_string()));
                }
                if let Some(value) = input_obj.get(name) {
                    match property.value_type {
                        ValueType::String => {
                            if !value.is_string() {
                                return Err(ValidationError::InvalidPropertyType(name.to_string()));
                            }
                        }
                        ValueType::Number => {
                            if !value.is_number() {
                                return Err(ValidationError::InvalidPropertyType(name.to_string()));
                            }
                        }
                        ValueType::Boolean => {
                            if !value.is_boolean() {
                                return Err(ValidationError::InvalidPropertyType(name.to_string()));
                            }
                        }
                        ValueType::Array => {
                            if !value.is_array() {
                                return Err(ValidationError::InvalidPropertyType(name.to_string()));
                            }
                        }
                        ValueType::Object => {
                            if !value.is_object() {
                                return Err(ValidationError::InvalidPropertyType(name.to_string()));
                            }
                        }
                    }
                }
            }
            Ok(())
        } else {
            Err(ValidationError::InvalidPropertyType(
                "Input must be an object".to_string(),
            ))
        }
    }
    pub async fn call<T: ToString>(
        &self,
        id: T,
        input: serde_json::Value,
        context: &ToolsContext,
    ) -> ToolResult {
        ToolResult {
            id: id.to_string(),
            content: self.handler.call_with_context(input, context).await,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolsContext {
    resources: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl ToolsContext {
    pub fn add_resource<T: Any + Send + Sync>(&mut self, resource: T) {
        self.resources.insert(TypeId::of::<T>(), Arc::new(resource));
    }

    pub fn with_resource<T: Any + Send + Sync>(mut self, resource: T) -> Self {
        self.add_resource(resource);
        self
    }

    pub fn get_resource<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.resources
            .get(&TypeId::of::<T>())
            .and_then(|resource| resource.downcast_ref())
    }
}

pub trait FromContext: Send + Sync + 'static {
    fn from_context(context: &ToolsContext) -> Self;
}

impl<T> FromContext for T
where
    T: Default + Any + Send + Sync + Clone,
{
    fn from_context(context: &ToolsContext) -> Self {
        match context.get_resource::<T>() {
            Some(resource) => resource.clone(),
            None => Default::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Tools {
    tools: HashMap<String, Tool>,
    context: ToolsContext,
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

    pub fn add_tool(mut self, tool: Tool) -> Self {
        self.tools.insert(tool.name.clone(), tool);
        self
    }

    pub fn push_tool(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn get(&self, tool_name: &str) -> Option<&Tool> {
        self.tools.get(tool_name)
    }

    pub async fn call(&self, tool_use: ToolUse) -> Option<ToolResult> {
        if let Some(tool) = self.get(&tool_use.name) {
            Some(
                tool.call(tool_use.id.clone(), tool_use.input, &self.context)
                    .await,
            )
        } else {
            None
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl From<Tool> for Tools {
    fn from(tool: Tool) -> Self {
        Tools::new().add_tool(tool)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUse {
    id: String,
    name: String,
    input: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    #[serde(rename = "tool_use_id")]
    id: String,
    content: serde_json::Value,
}

#[async_trait]
pub trait ToolHandler<T>: Send + Sync {
    async fn call(&self, input: serde_json::Value, context: &ToolsContext) -> serde_json::Value;
}

macro_rules! tuple_impls {
    ($($t:ident),*; $f:ident) => {
        #[async_trait]
        impl<$($t),*, $f, Fut> ToolHandler<($($t,)*)> for $f
        where
            $f: Fn(serde_json::Value, $($t),*) -> Fut + Send + Sync,
            $($t: FromContext,)*
            Fut: Future<Output = serde_json::Value> + Send,
        {
            async fn call(&self, input: serde_json::Value, context: &ToolsContext) -> serde_json::Value {
                (self)(input, $(<$t>::from_context(&context),)*).await
            }
        }
    }
}

macro_rules! impl_handler {
    (($($t:ident),*), $f:ident) => {
        tuple_impls!($($t),*; $f);
    };
}

#[async_trait]
impl<F, Fut> ToolHandler<()> for F
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync,
    Fut: Future<Output = serde_json::Value> + Send,
{
    async fn call(&self, input: serde_json::Value, _context: &ToolsContext) -> serde_json::Value {
        (self)(input).await
    }
}

impl_handler!((T1), F);
impl_handler!((T1, T2), F);
impl_handler!((T1, T2, T3), F);
impl_handler!((T1, T2, T3, T4), F);
impl_handler!((T1, T2, T3, T4, T5), F);
impl_handler!((T1, T2, T3, T4, T5, T6), F);
impl_handler!((T1, T2, T3, T4, T5, T6, T7), F);
impl_handler!((T1, T2, T3, T4, T5, T6, T7, T8), F);
impl_handler!((T1, T2, T3, T4, T5, T6, T7, T8, T9), F);
impl_handler!((T1, T2, T3, T4, T5, T6, T7, T8, T9, T10), F);
impl_handler!((T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11), F);
impl_handler!((T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12), F);

#[async_trait]
pub trait ToolHandlerFn: Send + Sync {
    async fn call_with_context(
        &self,
        input: serde_json::Value,
        context: &ToolsContext,
    ) -> serde_json::Value;
}

impl fmt::Debug for dyn ToolHandlerFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<ToolHandlerFn>")
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ToolHandlerWrapper<H, T>
where
    H: ToolHandler<T>,
{
    pub handler: H,
    pub phantom: PhantomData<T>,
}

#[async_trait]
impl<H, T> ToolHandlerFn for ToolHandlerWrapper<H, T>
where
    H: ToolHandler<T>,
    T: Send + Sync,
{
    async fn call_with_context(
        &self,
        input: serde_json::Value,
        context: &ToolsContext,
    ) -> serde_json::Value {
        self.handler.call(input, context).await
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    async fn empty_handler(_input: serde_json::Value) -> serde_json::Value {
        json!({})
    }

    #[test]
    fn test_property_new() {
        let property = Property::new("name");
        assert_eq!(property.name, "name");
        assert_eq!(property.value_type, ValueType::String);
        assert_eq!(property.description, None);
        assert!(!property.required);
    }

    #[test]
    fn test_property_value_type() {
        let property = Property::new("age").value_type(ValueType::Number);
        assert_eq!(property.value_type, ValueType::Number);
    }

    #[test]
    fn test_property_description() {
        let property = Property::new("email").description("User email address");
        assert_eq!(property.description, Some("User email address".to_string()));
    }

    #[test]
    fn test_property_required() {
        let property = Property::new("username").required();
        assert!(property.required);
    }

    #[test]
    fn test_property_builder_pattern() {
        let property = Property::new("admin")
            .boolean_type()
            .description("Admin user flag")
            .required();

        assert_eq!(property.name, "admin");
        assert_eq!(property.value_type, ValueType::Boolean);
        assert_eq!(property.description, Some("Admin user flag".to_string()));
        assert!(property.required);
    }

    #[test]
    fn test_tool_builder_new() {
        let tool = ToolBuilder::new("user_create");
        assert_eq!(tool.name, "user_create");
        assert_eq!(tool.description, None);
        assert_eq!(tool.input_schema.value_type, ValueType::Object);
        assert!(tool.input_schema.properties.is_empty());
    }

    #[test]
    fn test_tool_builder_description() {
        let tool = ToolBuilder::new("user_update").description("Update user profile");
        assert_eq!(tool.description, Some("Update user profile".to_string()));
    }

    #[test]
    fn test_tool_builder_add_property() {
        let tool = ToolBuilder::new("user_create")
            .add_property(Property::new("name").required())
            .add_property(Property::new("email").string_type())
            .add_property(Property::new("age").number_type());

        assert_eq!(tool.input_schema.properties.len(), 3);

        let name_prop = tool.input_schema.properties.get("name").unwrap();
        assert!(name_prop.required);

        let email_prop = tool.input_schema.properties.get("email").unwrap();
        assert_eq!(email_prop.value_type, ValueType::String);

        let age_prop = tool.input_schema.properties.get("age").unwrap();
        assert_eq!(age_prop.value_type, ValueType::Number);
    }

    #[test]
    fn test_validate_input_valid() {
        let tool = ToolBuilder::new("user_create")
            .add_property(Property::new("name").required())
            .add_property(Property::new("email").string_type())
            .add_property(Property::new("age").number_type())
            .handler(empty_handler)
            .build()
            .unwrap();

        let input = serde_json::json!({
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30
        });

        assert!(tool.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_missing_required() {
        let tool = ToolBuilder::new("user_create")
            .add_property(Property::new("name").required())
            .add_property(Property::new("email").string_type())
            .add_property(Property::new("age").number_type())
            .handler(empty_handler)
            .build()
            .unwrap();

        let input = serde_json::json!({
            "email": "john@example.com",
            "age": 30
        });

        match tool.validate_input(&input) {
            Err(ValidationError::MissingProperty(prop)) => assert_eq!(prop, "name"),
            _ => panic!("Expected MissingProperty error"),
        }
    }

    #[test]
    fn test_validate_input_invalid_type() {
        let tool = ToolBuilder::new("user_create")
            .add_property(Property::new("name").required())
            .add_property(Property::new("email").string_type())
            .add_property(Property::new("age").number_type())
            .handler(empty_handler)
            .build()
            .unwrap();

        let input = serde_json::json!({
            "name": "John Doe",
            "email": 123,
            "age": "30"
        });

        match tool.validate_input(&input) {
            Err(ValidationError::InvalidPropertyType(prop)) => assert_eq!(prop, "email"),
            _ => panic!("Expected InvalidPropertyType error"),
        }
    }

    #[test]
    fn test_validate_input_not_object() {
        let tool = ToolBuilder::new("user_create")
            .handler(empty_handler)
            .build()
            .unwrap();

        let input = serde_json::json!("invalid");

        match tool.validate_input(&input) {
            Err(ValidationError::InvalidPropertyType(msg)) => {
                assert_eq!(msg, "Input must be an object")
            }
            _ => panic!("Expected InvalidPropertyType error"),
        }
    }

    #[tokio::test]
    async fn test_tool_empty_handler() {
        let tool = ToolBuilder::new("user_create")
            .add_property(Property::new("name").required())
            .add_property(Property::new("email").string_type())
            .add_property(Property::new("age").number_type())
            .handler(empty_handler)
            .build()
            .unwrap();

        let context = ToolsContext::default();

        let input = serde_json::json!({
            "name": "John Doe",
            "email": 123,
            "age": "30"
        });

        let result = tool.call("id", input, &context).await;
        assert_eq!(result.id, "id");
        assert_eq!(result.content, json!({}));
    }

    #[tokio::test]
    async fn test_tool_handler() {
        async fn handler(input: serde_json::Value) -> serde_json::Value {
            json!({
            "name": input["name"],
            "email": input["email"],
            "age": input["age"]
            })
        }

        let tool = ToolBuilder::new("user_create")
            .add_property(Property::new("name").required())
            .add_property(Property::new("email").string_type())
            .add_property(Property::new("age").number_type())
            .handler(handler)
            .build()
            .unwrap();

        let context = ToolsContext::default();

        let input = serde_json::json!({
            "name": "John Doe",
            "email": 123,
            "age": "30"
        });

        let result = tool.call("id".to_owned(), input.clone(), &context).await;
        assert_eq!(result.id, "id");
        assert_eq!(result.content, input);
    }

    #[test]
    fn test_add_and_get_resource_primitive_type() {
        let mut context = ToolsContext::default();
        let value: i32 = 42;
        context.add_resource(value);

        let retrieved_value = context.get_resource::<i32>().unwrap();
        assert_eq!(*retrieved_value, value);
    }

    #[derive(Debug, Clone, PartialEq)]
    struct TestStruct {
        field1: String,
        field2: i32,
    }

    #[test]
    fn test_add_and_get_resource_custom_struct() {
        let mut context = ToolsContext::default();
        let test_struct = TestStruct {
            field1: "test".to_string(),
            field2: 42,
        };
        context.add_resource(test_struct.clone());

        let retrieved_struct = context.get_resource::<TestStruct>().unwrap();
        assert_eq!(*retrieved_struct, test_struct);
    }

    #[test]
    fn test_get_resource_not_found() {
        let context = ToolsContext::default();
        let retrieved_value = context.get_resource::<i32>();
        assert!(retrieved_value.is_none());
    }

    struct TestFromContext {
        value: i32,
    }

    impl FromContext for TestFromContext {
        fn from_context(context: &ToolsContext) -> Self {
            let value = context.get_resource::<i32>().unwrap();
            TestFromContext { value: *value }
        }
    }

    #[test]
    fn test_from_context_trait() {
        let mut context = ToolsContext::default();
        let value: i32 = 42;
        context.add_resource(value);

        let test_from_context = TestFromContext::from_context(&context);
        assert_eq!(test_from_context.value, value);
    }

    #[tokio::test]
    async fn test_tool_handler_with_context() {
        async fn handler(input: serde_json::Value, salary: i32) -> serde_json::Value {
            json!({
            "name": input["name"],
            "email": input["email"],
            "age": input["age"],
            "salary": salary
            })
        }

        let tool = ToolBuilder::new("user_create")
            .add_property(Property::new("name").required())
            .add_property(Property::new("email").string_type())
            .add_property(Property::new("age").number_type())
            .handler(handler)
            .build()
            .unwrap();

        let context = ToolsContext::default().with_resource(1000_i32);

        let input = serde_json::json!({
            "name": "John Doe",
            "email": 123,
            "age": "30"
        });

        let result = tool.call("id".to_owned(), input.clone(), &context).await;
        assert_eq!(result.id, "id");
        assert_eq!(result.content["salary"], 1000_i32);
    }
}
