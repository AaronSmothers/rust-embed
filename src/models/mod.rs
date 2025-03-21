pub mod mini_lm;

// Include the generated Protobuf code
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/embeddings.rs"));
}

// Common model traits and utilities
/// Model configuration trait for managing embedding model parameters
pub trait ModelConfig {
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
    fn model_version(&self) -> &str;
} 