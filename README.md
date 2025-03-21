# Rust Embeddings

A pure Rust implementation of text embedding algorithms, particularly the all-MiniLM-L6-v2 model.

## Features

- Pure Rust implementation of text embedding
- Efficient storage using Protocol Buffers
- Simple interface for embedding text
- Optimized for large-scale embedding tasks (supports processing 100GB+ datasets)
- Command-line interface for easy usage

## Usage

### Command Line

```bash
# Embed a single text
cargo run -- --text "This is a sample text to embed" --output embeddings.pb

# Embed multiple texts from a file (one per line)
cargo run -- --file input.txt --output embeddings.pb
```

### As a Library

```rust
use rust_embed::{Embedder, MiniLMEmbedder};

fn main() -> Result<()> {
    // Create the embedder
    let embedder = MiniLMEmbedder::new()?;
    
    // Embed a text
    let text = "This is a sample text to embed";
    let embedding = embedder.embed_text(text)?;
    
    // Use the embedding for similarity comparison
    let text2 = "This is another sample text";
    let embedding2 = embedder.embed_text(text2)?;
    
    let similarity = embedder.cosine_similarity(&embedding, &embedding2);
    println!("Similarity: {}", similarity);
    
    Ok(())
}
```

## Project Structure

- `src/embedding.rs`: Core embedding trait definition
- `src/models/mini_lm/`: Implementation of the all-MiniLM-L6-v2 model
- `src/utils.rs`: Utility functions for saving/loading embeddings
- `proto/embeddings.proto`: Protocol Buffers schema for storing embeddings

## Performance

This implementation is designed to be efficient for large-scale embedding tasks. For a 106GB dataset, it uses Protocol Buffers to minimize storage requirements and maximize serialization/deserialization speed.

## Future Improvements

- [ ] Implement true MiniLM model weights loading
- [ ] Add support for more embedding models
- [ ] Optimize for parallel processing
- [ ] Add benchmarking tools
- [ ] Implement retrieval functionality for similarity search

## License

This project is licensed under the MIT License - see the LICENSE file for details. 