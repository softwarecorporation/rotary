use serde::{Deserialize, Serialize};

/// A provider tool call shared by the native and portable agent APIs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}
