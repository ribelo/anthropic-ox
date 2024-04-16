use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    future::Future,
    marker::PhantomData,
    sync::{Arc, Mutex, OnceLock},
};

use super::message::MultimodalContent;

const NAME_REGEX_PATTERN: &str = r"^[a-zA-Z0-9_-]{1,64}$";
static NAME_REGEX: OnceLock<Regex> = OnceLock::new();

#[derive(Debug, Clone, Serialize, thiserror::Error)]
#[error("Invalid key name: {0}")]
pub struct PropertyNameError(String);

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct StringProperty {
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    enumerate: Option<Vec<String>>,
}

impl StringProperty {
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }
    pub fn enumerate<T: IntoIterator<Item = U>, U: Into<String>>(mut self, enumerate: T) -> Self {
        self.enumerate = Some(enumerate.into_iter().map(|s| s.into()).collect());
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct NumberProperty {
    description: Option<String>,
}

impl NumberProperty {
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct BooleanProperty {
    description: Option<String>,
}

impl BooleanProperty {
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct ArrayProperty {
    description: Option<String>,
    items: Vec<Box<Property>>,
}

impl ArrayProperty {
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }
    pub fn items(mut self, items: Vec<Box<Property>>) -> Self {
        self.items = items;
        self
    }
    pub fn push_item(mut self, item: Box<Property>) -> Self {
        self.items.push(item);
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct ObjectProperty {
    description: Option<String>,
    required: Vec<String>,
    properties: HashMap<String, Box<Property>>,
}

impl ObjectProperty {
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }
    pub fn required(mut self, required: Vec<String>) -> Self {
        self.required = required;
        self
    }
    pub fn properties(mut self, properties: HashMap<String, Box<Property>>) -> Self {
        self.properties = properties;
        self
    }
    pub fn insert_property(
        &mut self,
        name: &str,
        is_required: bool,
        property: Property,
    ) -> Result<(), PropertyNameError> {
        let regex = NAME_REGEX.get_or_init(|| Regex::new(NAME_REGEX_PATTERN).unwrap());
        if !regex.is_match(name) {
            return Err(PropertyNameError(name.to_string()));
        }
        self.properties.insert(name.to_string(), Box::new(property));
        if is_required {
            self.required.push(name.to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Property {
    String(StringProperty),
    Number(NumberProperty),
    Boolean(BooleanProperty),
    Array(ArrayProperty),
    Object(ObjectProperty),
}

impl From<StringProperty> for Property {
    fn from(prop: StringProperty) -> Self {
        Property::String(prop)
    }
}

impl From<NumberProperty> for Property {
    fn from(prop: NumberProperty) -> Self {
        Property::Number(prop)
    }
}

impl From<BooleanProperty> for Property {
    fn from(prop: BooleanProperty) -> Self {
        Property::Boolean(prop)
    }
}

impl From<ArrayProperty> for Property {
    fn from(prop: ArrayProperty) -> Self {
        Property::Array(prop)
    }
}

impl From<ObjectProperty> for Property {
    fn from(prop: ObjectProperty) -> Self {
        Property::Object(prop)
    }
}

impl Property {
    pub fn string() -> Self {
        StringProperty::default().into()
    }
    pub fn number() -> Self {
        NumberProperty::default().into()
    }
    pub fn boolean() -> Self {
        BooleanProperty::default().into()
    }
    pub fn array() -> Self {
        ArrayProperty::default().into()
    }
    pub fn object() -> Self {
        ObjectProperty::default().into()
    }
    pub fn as_string(&self) -> Option<&StringProperty> {
        match self {
            Property::String(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn as_number(&self) -> Option<&NumberProperty> {
        match self {
            Property::Number(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn as_boolean(&self) -> Option<&BooleanProperty> {
        match self {
            Property::Boolean(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&ArrayProperty> {
        match self {
            Property::Array(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn as_object(&self) -> Option<&ObjectProperty> {
        match self {
            Property::Object(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn as_string_mut(&mut self) -> Option<&mut StringProperty> {
        match self {
            Property::String(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn as_number_mut(&mut self) -> Option<&mut NumberProperty> {
        match self {
            Property::Number(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn as_boolean_mut(&mut self) -> Option<&mut BooleanProperty> {
        match self {
            Property::Boolean(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn as_array_mut(&mut self) -> Option<&mut ArrayProperty> {
        match self {
            Property::Array(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn as_object_mut(&mut self) -> Option<&mut ObjectProperty> {
        match self {
            Property::Object(prop) => Some(prop),
            _ => None,
        }
    }
    pub fn validate_input(&self, input: &serde_json::Value) -> Result<(), ValidationError> {
        match self {
            Property::String(StringProperty { enumerate, .. }) => {
                if let Some(value) = input.as_str() {
                    if let Some(enum_values) = enumerate {
                        if !enum_values.contains(&value.to_string()) {
                            return Err(ValidationError::InvalidEnum(value.to_string()));
                        }
                    }
                    Ok(())
                } else {
                    Err(ValidationError::InvalidValue(input.clone()))
                }
            }
            Property::Number(_) => {
                if input.is_number() {
                    Ok(())
                } else {
                    Err(ValidationError::InvalidValue(input.clone()))
                }
            }
            Property::Boolean(_) => {
                if input.is_boolean() {
                    Ok(())
                } else {
                    Err(ValidationError::InvalidValue(input.clone()))
                }
            }
            Property::Array(ArrayProperty { items, .. }) => {
                if let Some(array) = input.as_array() {
                    for (index, item) in array.iter().enumerate() {
                        if let Some(property) = items.get(index) {
                            property.validate_input(item)?;
                        } else {
                            return Err(ValidationError::InvalidValue(item.clone()));
                        }
                    }
                    Ok(())
                } else {
                    Err(ValidationError::InvalidValue(input.clone()))
                }
            }
            Property::Object(ObjectProperty {
                properties,
                required,
                ..
            }) => {
                if let Some(object) = input.as_object() {
                    for (name, value) in object {
                        if let Some(property) = properties.get(name) {
                            property.validate_input(value)?;
                        } else {
                            return Err(ValidationError::InvalidValue(value.clone()));
                        }
                    }
                    for field in required {
                        if !object.contains_key(field) {
                            return Err(ValidationError::MissingField(field.clone()));
                        }
                    }
                    Ok(())
                } else {
                    Err(ValidationError::InvalidValue(input.clone()))
                }
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Invalid enum value: {0}")]
    InvalidEnum(String),
    #[error("Invalid property types: {0:?}")]
    InvalidValue(serde_json::Value),
}

#[derive(Debug)]
pub struct ToolBuilder {
    name: String,
    description: Option<String>,
    input_schema: Property,
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
            input_schema: Property::object(),
            handler: None,
        }
    }

    pub fn description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    pub fn schema(mut self, property: Property) -> Self {
        self.input_schema = property;
        self
    }

    pub fn add_required_property(
        mut self,
        name: &str,
        property: Property,
    ) -> Result<Self, PropertyNameError> {
        self.input_schema
            .as_object_mut()
            .unwrap()
            .insert_property(name, true, property)?;
        Ok(self)
    }

    pub fn add_optional_property(
        mut self,
        name: &str,
        property: Property,
    ) -> Result<Self, PropertyNameError> {
        self.input_schema
            .as_object_mut()
            .unwrap()
            .insert_property(name, false, property)?;
        Ok(self)
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
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: Property,
    #[serde(skip)]
    handler: Arc<dyn ToolHandlerFn>,
}

impl PartialEq for Tool {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Tool {}

impl Tool {
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
    pub fn push_resource<T: Any + Send + Sync + Clone>(&mut self, resource: T) {
        self.resources.insert(TypeId::of::<T>(), Arc::new(resource));
    }
    pub fn add_resource<T: Any + Send + Sync + Clone>(mut self, resource: T) -> Self {
        self.push_resource(resource);
        self
    }

    pub fn get_resource<T: Any + Send + Sync + Clone>(&self) -> Option<T> {
        let boxed_resource = self.resources.get(&TypeId::of::<T>())?;
        Some(boxed_resource.downcast_ref::<T>().unwrap().clone())
    }
}

pub trait FromContext: Send + 'static {
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

    pub fn push_tool(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn add_tool(mut self, tool: Tool) -> Self {
        self.push_tool(tool);
        self
    }

    pub fn get_tool(&self, tool_name: &str) -> Option<&Tool> {
        self.tools.get(tool_name)
    }

    pub fn push_resource<T: Any + Send + Sync + Clone>(&mut self, resource: T) {
        self.context.push_resource(resource);
    }

    pub fn add_resource<T: Any + Send + Sync + Clone>(mut self, resource: T) -> Self {
        self.push_resource(resource);
        self
    }

    pub fn get_resource<T: Any + Send + Sync + Clone>(&self) -> Option<T> {
        self.context.get_resource()
    }

    pub async fn use_tool(&self, tool_use: ToolUse) -> Option<ToolResult> {
        if let Some(tool) = self.get_tool(&tool_use.name) {
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

impl From<Vec<Tool>> for Tools {
    fn from(tools: Vec<Tool>) -> Self {
        let mut tools_map = Tools::new();
        for tool in tools {
            tools_map.push_tool(tool);
        }
        tools_map
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

impl fmt::Display for ToolUse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "tool_use: {}, name: {}, input: {}",
            self.id, self.name, self.input
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResult {
    #[serde(rename = "tool_use_id")]
    pub id: String,
    pub content: serde_json::Value,
}

impl fmt::Display for ToolResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "tool_result: {}, content: {}", self.id, self.content)
    }
}

pub trait ExtractToolUse {
    fn extract_tool_uses(&self) -> Vec<&ToolUse>;
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
    fn test_string_property_default() {
        let prop = StringProperty::default();
        assert_eq!(prop.description, None);
        assert_eq!(prop.enumerate, None);
    }

    #[test]
    fn test_string_property_description() {
        let prop = StringProperty::default().description("A string property");
        assert_eq!(prop.description, Some("A string property".to_string()));
        assert_eq!(prop.enumerate, None);
    }

    #[test]
    fn test_string_property_description_string() {
        let desc = "Another description".to_string();
        let prop = StringProperty::default().description(desc.clone());
        assert_eq!(prop.description, Some(desc));
    }

    #[test]
    fn test_string_property_enumerate() {
        let values = vec!["one".to_string(), "two".to_string()];
        let prop = StringProperty::default().enumerate(values.clone());
        assert_eq!(prop.description, None);
        assert_eq!(prop.enumerate, Some(values));
    }

    #[test]
    fn test_string_property_enumerate_vec() {
        let values = vec!["a", "b", "c"];
        let prop = StringProperty::default().enumerate(values.clone());
        assert_eq!(
            prop.enumerate,
            Some(values.iter().map(|&s| s.to_string()).collect())
        );
    }

    #[test]
    fn test_string_property_chained_builder() {
        let prop = StringProperty::default()
            .description("A string property")
            .enumerate(vec!["one", "two"]);

        assert_eq!(prop.description, Some("A string property".to_string()));
        assert_eq!(
            prop.enumerate,
            Some(vec!["one".to_string(), "two".to_string()])
        );
    }

    #[test]
    fn test_number_property_default() {
        let prop = NumberProperty::default();
        assert_eq!(prop.description, None);
    }

    #[test]
    fn test_number_property_description() {
        let prop = NumberProperty::default().description("A number property");
        assert_eq!(prop.description, Some("A number property".to_string()));
    }

    #[test]
    fn test_number_property_description_string() {
        let desc = "Another number description".to_string();
        let prop = NumberProperty::default().description(desc.clone());
        assert_eq!(prop.description, Some(desc));
    }

    #[test]
    fn test_boolean_property_default() {
        let prop = BooleanProperty::default();
        assert_eq!(prop.description, None);
    }

    #[test]
    fn test_boolean_property_description() {
        let prop = BooleanProperty::default().description("A boolean property");
        assert_eq!(prop.description, Some("A boolean property".to_string()));
    }

    #[test]
    fn test_boolean_property_description_string() {
        let desc = "Another boolean description".to_string();
        let prop = BooleanProperty::default().description(desc.clone());
        assert_eq!(prop.description, Some(desc));
    }

    fn test_array_property_default() {
        let prop = ArrayProperty::default();
        assert_eq!(prop.description, None);
        assert_eq!(prop.items, Vec::new());
    }

    #[test]
    fn test_array_property_description() {
        let prop = ArrayProperty::default().description("An array property");
        assert_eq!(prop.description, Some("An array property".to_string()));
        assert_eq!(prop.items, Vec::new());
    }

    #[test]
    fn test_array_property_description_string() {
        let desc = "Another array description".to_string();
        let prop = ArrayProperty::default().description(desc.clone());
        assert_eq!(prop.description, Some(desc));
    }

    #[test]
    fn test_array_property_items() {
        let items = vec![
            Box::new(Property::String(StringProperty::default())),
            Box::new(Property::Number(NumberProperty::default())),
        ];
        let prop = ArrayProperty::default().items(items.clone());
        assert_eq!(prop.description, None);
        assert_eq!(prop.items, items);
    }

    #[test]
    fn test_array_property_push_item() {
        let item1 = Box::new(Property::String(StringProperty::default()));
        let item2 = Box::new(Property::Number(NumberProperty::default()));
        let prop = ArrayProperty::default()
            .push_item(item1.clone())
            .push_item(item2.clone());
        assert_eq!(prop.items, vec![item1, item2]);
    }

    #[test]
    fn test_array_property_chained_builder() {
        let item = Box::new(Property::Boolean(BooleanProperty::default()));
        let prop = ArrayProperty::default()
            .description("An array property")
            .push_item(item.clone());
        assert_eq!(prop.description, Some("An array property".to_string()));
        assert_eq!(prop.items, vec![item]);
    }

    #[tokio::test]
    async fn test_tool_empty_handler() {
        let tool = ToolBuilder::new("user_create")
            .add_required_property("name", Property::string())
            .unwrap()
            .add_required_property("email", Property::string())
            .unwrap()
            .add_required_property("age", Property::number())
            .unwrap()
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
            .add_required_property("name", Property::string())
            .unwrap()
            .add_required_property("email", Property::string())
            .unwrap()
            .add_required_property("age", Property::number())
            .unwrap()
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
        context.push_resource(value);

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
        let mut context = ToolsContext::default();
        let test_struct = TestStruct {
            field1: "test".to_string(),
            field2: 42,
        };
        context.push_resource(test_struct.clone());

        let retrieved_struct = context.get_resource::<TestStruct>().unwrap();
        assert_eq!(retrieved_struct, test_struct);
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
            TestFromContext { value: value }
        }
    }

    #[test]
    fn test_from_context_trait() {
        let mut context = ToolsContext::default();
        let value: i32 = 42;
        context.push_resource(value);

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
            .add_required_property("name", Property::string())
            .unwrap()
            .add_required_property("email", Property::string())
            .unwrap()
            .add_required_property("age", Property::number())
            .unwrap()
            .handler(handler)
            .build()
            .unwrap();

        let context = ToolsContext::default().add_resource(1000_i32);

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
