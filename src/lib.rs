//! Rotary agent capabilities for native hosts and portable Wasm hosts.
//!
//! Default features preserve the native agent API. For Cloudflare Workers use
//! `default-features = false, features = ["lite"]` and `lite::Agent`.
//! The lite agent yields provider requests and tool calls to its host; it does
//! not create a runtime, fetch credentials, run tools, or persist state itself.
#![forbid(unsafe_code)]

pub mod permissions;
pub mod secrets;
mod tool_call;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "lite")]
pub mod lite;

// Permission callbacks share the native tool-call contract without compiling
// the native executor and its operating-system capabilities into Wasm.
#[cfg(not(feature = "native"))]
pub mod agent {
    pub use crate::tool_call::ToolCall;
}

#[cfg(feature = "native")]
include!("native.rs");
