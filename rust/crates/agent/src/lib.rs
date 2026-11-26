//! Tool-call parsing and argument validation for megaRL's agentic loop.
//!
//! The architecture baseline assigns tokenizer/template/tool parsing to the
//! Rust spine (`devdocs/architecture.md` §0). This crate keeps that work off the
//! Python control path: the agent loop hands it a decoded assistant message and
//! gets back structured, canonical tool calls that can be framed and replayed.
//!
//! Two wire formats are supported:
//!   * `json` — OpenAI-style `{"tool_calls":[{"name":...,"arguments":{...}}]}`,
//!     a bare array, a single object, or a fenced ```json block.
//!   * `xml`  — `<tool_call name="calc" id="c1">{"expr":"1+2"}</tool_call>`.
//!   * `auto` — try `json`, then `xml`.
//!
//! `validate_arguments` implements the dependency-free subset of JSON Schema
//! used by tool definitions (type/required/properties/enum/items/bounds). It
//! returns path-aware errors so an invalid tool call never reaches execution.

use serde::Serialize;
use serde_json::Value;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum AgentError {
    Parse(String),
    Schema(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::Parse(msg) => write!(f, "tool-call parse error: {msg}"),
            AgentError::Schema(msg) => write!(f, "tool argument error: {msg}"),
        }
    }
}

impl std::error::Error for AgentError {}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ParsedToolCall {
    /// Call id supplied by the model; empty when absent (the loop assigns one).
    pub id: String,
    pub name: String,
    /// Canonical JSON object string, ready for `validate_arguments`.
    pub arguments_json: String,
}

/// Parse tool calls from a decoded assistant message.
pub fn parse_tool_calls(text: &str, format: &str) -> Result<Vec<ParsedToolCall>, AgentError> {
    match format {
        "json" => parse_json(text),
        "xml" => parse_xml(text),
        // Only fall back to XML when the message actually carries an XML tool
        // call; otherwise a malformed JSON tool call would be silently treated
        // as a plain final answer.
        "auto" => parse_json(text).or_else(|err| {
            if text.contains("<tool_call") {
                parse_xml(text)
            } else {
                Err(err)
            }
        }),
        other => Err(AgentError::Parse(format!(
            "unknown tool-call format: {other}"
        ))),
    }
}

fn as_str(value: Option<&Value>, field: &str) -> Result<String, AgentError> {
    match value {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(AgentError::Parse(format!("{field} must be a string"))),
        None => Ok(String::new()),
    }
}

/// Normalize `arguments`: object -> canonical JSON, string -> parsed JSON object.
fn arguments_json(value: Option<&Value>) -> Result<String, AgentError> {
    let value = match value {
        Some(v) => v,
        None => return Ok("{}".to_string()),
    };
    let object = match value {
        Value::Object(_) => value.clone(),
        Value::String(raw) => serde_json::from_str::<Value>(raw).map_err(|e| {
            AgentError::Parse(format!("arguments string is not valid JSON: {e}"))
        })?,
        Value::Null => return Ok("{}".to_string()),
        _ => {
            return Err(AgentError::Parse(
                "arguments must be a JSON object".to_string(),
            ))
        }
    };
    if !object.is_object() {
        return Err(AgentError::Parse(
            "arguments must be a JSON object".to_string(),
        ));
    }
    Ok(object.to_string())
}

fn call_from_value(item: &Value) -> Result<ParsedToolCall, AgentError> {
    // OpenAI tool calls wrap the payload in `function`.
    let payload = if item.get("function").is_some() {
        item.get("function").unwrap()
    } else {
        item
    };
    let name = as_str(payload.get("name").or_else(|| item.get("name")), "name")?;
    if name.trim().is_empty() {
        return Err(AgentError::Parse("tool call has no name".to_string()));
    }
    let id = as_str(item.get("id"), "id")?;
    let arguments = arguments_json(
        payload
            .get("arguments")
            .or_else(|| item.get("arguments")),
    )?;
    Ok(ParsedToolCall {
        id,
        name,
        arguments_json: arguments,
    })
}

/// Strip a leading ```json ... ``` fence if the whole text is not plain JSON.
fn fenced_json(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return None;
    }
    let after_fence = trimmed.find('\n')? + 1;
    let close = trimmed[after_fence..].find("```")? + after_fence;
    Some(trimmed[after_fence..close].trim())
}

fn parse_json(text: &str) -> Result<Vec<ParsedToolCall>, AgentError> {
    let trimmed = text.trim();
    // A plain final answer carries no tool-call syntax at all. Treat it as
    // terminal instead of failing the episode on a JSON parse error.
    if trimmed.is_empty() || (!trimmed.contains('{') && !trimmed.contains('[')) {
        return Ok(Vec::new());
    }
    let value = match serde_json::from_str::<Value>(trimmed) {
        Ok(v) => v,
        Err(original) => match fenced_json(trimmed) {
            Some(inner) => serde_json::from_str::<Value>(inner).map_err(|e| {
                AgentError::Parse(format!(
                    "fenced tool call is not valid JSON: {e} ({original})"
                ))
            })?,
            None => return Err(AgentError::Parse(original.to_string())),
        },
    };
    let mut calls = Vec::new();
    match &value {
        Value::Array(items) => {
            for item in items {
                calls.push(call_from_value(item)?);
            }
        }
        Value::Object(_) => {
            if let Some(Value::Array(items)) = value.get("tool_calls") {
                for item in items {
                    calls.push(call_from_value(item)?);
                }
            } else if value.get("name").is_some() {
                calls.push(call_from_value(&value)?);
            } else {
                return Err(AgentError::Parse(
                    "JSON object has neither `tool_calls` nor `name`".to_string(),
                ));
            }
        }
        _ => {
            return Err(AgentError::Parse(
                "tool-call JSON must be an object or array".to_string(),
            ))
        }
    }
    Ok(calls)
}

fn attribute(open_tag: &str, key: &str) -> String {
    for token in open_tag.split_whitespace().skip(1) {
        let token = token.trim_end_matches('>').trim_start_matches('<');
        if let Some(rest) = token.strip_prefix(&format!("{key}=")) {
            let rest = rest.trim_matches(|c| c == '"' || c == '\'');
            return rest.to_string();
        }
    }
    String::new()
}

fn parse_xml(text: &str) -> Result<Vec<ParsedToolCall>, AgentError> {
    let mut calls = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find("<tool_call") {
        let start = cursor + rel;
        let open_end = text[start..]
            .find('>')
            .ok_or_else(|| AgentError::Parse("unclosed <tool_call> tag".to_string()))?
            + start;
        let close_rel = text[open_end..]
            .find("</tool_call>")
            .ok_or_else(|| AgentError::Parse("missing </tool_call>".to_string()))?;
        let close = open_end + close_rel;
        let open_tag = &text[start..=open_end];
        let inner = text[open_end + 1..close].trim();
        let parsed: Value = serde_json::from_str(inner).map_err(|e| {
            AgentError::Parse(format!("tool_call body is not valid JSON: {e}"))
        })?;
        if !parsed.is_object() {
            return Err(AgentError::Parse(
                "tool_call body must be a JSON object".to_string(),
            ));
        }
        let name = as_str(parsed.get("name"), "name")?;
        let name = if name.is_empty() {
            attribute(open_tag, "name")
        } else {
            name
        };
        if name.trim().is_empty() {
            return Err(AgentError::Parse("tool call has no name".to_string()));
        }
        // Body keys other than name/id become arguments unless an explicit
        // `arguments` object is present.
        let arguments = if parsed.get("arguments").is_some() {
            arguments_json(parsed.get("arguments"))?
        } else {
            let mut map = parsed.as_object().unwrap().clone();
            map.remove("name");
            map.remove("id");
            Value::Object(map).to_string()
        };
        let id = {
            let from_body = as_str(parsed.get("id"), "id")?;
            if from_body.is_empty() {
                attribute(open_tag, "id")
            } else {
                from_body
            }
        };
        calls.push(ParsedToolCall {
            id,
            name,
            arguments_json: arguments,
        });
        cursor = close + "</tool_call>".len();
    }
    Ok(calls)
}

/// Validate canonical tool arguments against the dependency-free JSON Schema
/// subset used by tool definitions. `path` in errors points at the bad field.
pub fn validate_arguments(schema_json: &str, arguments_json: &str) -> Result<(), AgentError> {
    let schema: Value = serde_json::from_str(schema_json)
        .map_err(|e| AgentError::Schema(format!("invalid tool schema: {e}")))?;
    let args: Value = serde_json::from_str(arguments_json)
        .map_err(|e| AgentError::Schema(format!("invalid tool arguments: {e}")))?;
    check(&args, &schema, "$")
}

fn type_matches(value: &Value, expected: &str) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true, // unknown annotations are not enforced
    }
}

fn check(value: &Value, schema: &Value, path: &str) -> Result<(), AgentError> {
    if let Some(expected) = schema.get("type").and_then(Value::as_str) {
        if !type_matches(value, expected) {
            return Err(AgentError::Schema(format!(
                "{path}: expected {expected}, got {}",
                value_type(value)
            )));
        }
    }
    if let Some(Value::Array(options)) = schema.get("enum") {
        if !options.iter().any(|option| option == value) {
            return Err(AgentError::Schema(format!("{path}: not in enum")));
        }
    }
    if let Some(Value::Object(properties)) = schema.get("properties") {
        for (key, subschema) in properties {
            if let Some(child) = value.get(key) {
                check(child, subschema, &format!("{path}.{key}"))?;
            }
        }
    }
    if let Some(Value::Array(required)) = schema.get("required") {
        for key in required {
            if let Some(key) = key.as_str() {
                if value.get(key).is_none() {
                    return Err(AgentError::Schema(format!("{path}.{key}: required")));
                }
            }
        }
    }
    if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
        if let (Some(properties), Some(object)) = (schema.get("properties"), value.as_object()) {
            let allowed = properties.as_object().ok_or_else(|| {
                AgentError::Schema("`properties` must be an object".to_string())
            })?;
            for key in object.keys() {
                if !allowed.contains_key(key) {
                    return Err(AgentError::Schema(format!(
                        "{path}.{key}: additional property not allowed"
                    )));
                }
            }
        }
    }
    if let Some(subschema) = schema.get("items") {
        if let Some(array) = value.as_array() {
            for (index, item) in array.iter().enumerate() {
                check(item, subschema, &format!("{path}[{index}]"))?;
            }
        }
    }
    if let Some(number) = value.as_f64() {
        if let Some(min) = schema.get("minimum").and_then(Value::as_f64) {
            if number < min {
                return Err(AgentError::Schema(format!("{path}: below minimum {min}")));
            }
        }
        if let Some(max) = schema.get("maximum").and_then(Value::as_f64) {
            if number > max {
                return Err(AgentError::Schema(format!("{path}: above maximum {max}")));
            }
        }
    }
    if let Some(text) = value.as_str() {
        if let Some(min) = schema.get("minLength").and_then(Value::as_u64) {
            if (text.chars().count() as u64) < min {
                return Err(AgentError::Schema(format!("{path}: shorter than {min}")));
            }
        }
        if let Some(max) = schema.get("maxLength").and_then(Value::as_u64) {
            if (text.chars().count() as u64) > max {
                return Err(AgentError::Schema(format!("{path}: longer than {max}")));
            }
        }
    }
    Ok(())
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Object(_) => "object",
        Value::Array(_) => "array",
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "boolean",
        Value::Null => "null",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_style_calls() {
        let text = r#"{"tool_calls":[{"id":"c1","name":"calc","arguments":{"expr":"1+2"}},{"id":"c2","name":"echo","arguments":"{\"text\":\"hi\"}"}]}"#;
        let calls = parse_tool_calls(text, "json").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "calc");
        assert_eq!(calls[0].arguments_json, r#"{"expr":"1+2"}"#);
        assert_eq!(calls[1].arguments_json, r#"{"text":"hi"}"#);
    }

    #[test]
    fn parses_fenced_and_function_wrapped_calls() {
        let text = "```json\n{\"tool_calls\":[{\"id\":\"c1\",\"function\":{\"name\":\"calc\",\"arguments\":\"{\\\"expr\\\":\\\"2*3\\\"}\"}}]}\n```";
        let calls = parse_tool_calls(text, "json").unwrap();
        assert_eq!(calls[0].name, "calc");
        assert_eq!(calls[0].arguments_json, r#"{"expr":"2*3"}"#);
    }

    #[test]
    fn plain_answer_has_no_calls() {
        assert!(parse_tool_calls("The answer is 4.", "json")
            .unwrap()
            .is_empty());
        assert!(parse_tool_calls("The answer is 4.", "xml")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn parses_xml_calls() {
        let text = r#"Let me compute. <tool_call name="calc" id="c1">{"expr":"1+2"}</tool_call>"#;
        let calls = parse_tool_calls(text, "xml").unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "calc");
        assert_eq!(calls[0].id, "c1");
        assert_eq!(calls[0].arguments_json, r#"{"expr":"1+2"}"#);
    }

    #[test]
    fn auto_falls_back_to_xml() {
        let text = r#"<tool_call name="calc">{"expr":"7+8"}</tool_call>"#;
        assert_eq!(parse_tool_calls(text, "auto").unwrap().len(), 1);
    }

    #[test]
    fn rejects_non_object_arguments() {
        let text = r#"{"name":"calc","arguments":[1,2]}"#;
        assert!(parse_tool_calls(text, "json").is_err());
    }

    #[test]
    fn validates_arguments_against_schema() {
        let schema = r#"{"type":"object","required":["expr"],"properties":{"expr":{"type":"string","minLength":1}},"additionalProperties":false}"#;
        validate_arguments(schema, r#"{"expr":"1+2"}"#).unwrap();
        assert!(validate_arguments(schema, r#"{}"#).is_err());
        assert!(validate_arguments(schema, r#"{"expr":3}"#).is_err());
        assert!(validate_arguments(schema, r#"{"expr":"1","extra":true}"#).is_err());
    }

    #[test]
    fn validates_nested_items_and_bounds() {
        let schema = r#"{"type":"object","properties":{"xs":{"type":"array","items":{"type":"integer","minimum":0}}}}"#;
        validate_arguments(schema, r#"{"xs":[1,2]}"#).unwrap();
        assert!(validate_arguments(schema, r#"{"xs":[1,-2]}"#).is_err());
        assert!(validate_arguments(schema, r#"{"xs":["a"]}"#).is_err());
    }
}
