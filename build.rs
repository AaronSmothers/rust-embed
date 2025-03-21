use std::io::Result;

fn main() -> Result<()> {
    // Compile the protobuf definitions
    prost_build::compile_protos(&["proto/embeddings.proto"], &["proto/"])?;
    Ok(())
} 