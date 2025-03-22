use anyhow::Result;
use clap::Parser;
use ndarray::s;
use rust_embed::{
    embedding::Embedder,
    models::mini_lm::MiniLMEmbedder,
    utils,
};
use std::path::PathBuf;

/// Command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to embed
    #[arg(short, long)]
    text: Option<String>,
    
    /// File containing text to embed (one text per line)
    #[arg(short, long)]
    file: Option<PathBuf>,
    
    /// Output file for the embeddings
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Parse command line arguments
    let args = Args::parse();
    
    // Create the MiniLM embedder
    let mut embedder = MiniLMEmbedder::new();
    
    // Load the tokenizer and model
    println!("Loading tokenizer...");
    embedder.load_or_download_tokenizer()?;
    
    println!("Loading model...");
    embedder.load_or_download_model()?;
    println!("Using the all-MiniLM-L6-v2 model for generating embeddings.");
    
    // Process text based on input source
    if let Some(text) = args.text {
        println!("Embedding single text: {}", text);
        let embedding = embedder.embed_text(&text)?;
        println!("Embedding size: {}", embedding.len());
        println!("First few values: {:?}", &embedding.slice(s![..5]));
        
        // Save to file if output is specified
        if let Some(output) = args.output {
            let text_vec = vec![text];
            utils::save_embeddings(
                &[embedding], 
                Some(&text_vec),
                embedder.model_name(),
                embedder.model_version(),
                embedder.dimension() as i32,
                output
            )?;
            println!("Embedding saved to file");
        }
    } else if let Some(file) = args.file {
        println!("Embedding texts from file: {:?}", file);
        
        // Read file line by line
        let content = std::fs::read_to_string(file)?;
        let texts: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        
        // Embed each line
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in &texts {
            let embedding = embedder.embed_text(text)?;
            embeddings.push(embedding);
        }
        
        println!("Embedded {} texts", embeddings.len());
        
        // Save to file if output is specified
        if let Some(output) = args.output {
            utils::save_embeddings(
                &embeddings, 
                Some(&texts),
                embedder.model_name(),
                embedder.model_version(),
                embedder.dimension() as i32,
                output
            )?;
            println!("Embeddings saved to file");
        }
    } else {
        println!("Please provide either --text or --file argument");
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_embedding() -> Result<()> {
        let mut embedder = MiniLMEmbedder::new();
        embedder.load_or_download_tokenizer()?;
        embedder.load_or_download_model()?;
        let text = "This is a test sentence for embedding.";
        let embedding = embedder.embed_text(text)?;
        
        // Check dimensions
        assert_eq!(embedding.len(), 384);
        
        // Check normalization (length should be close to 1.0)
        let norm = embedding.dot(&embedding).sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        
        Ok(())
    }
    
    #[test]
    fn test_similarity() -> Result<()> {
        let mut embedder = MiniLMEmbedder::new();
        embedder.load_or_download_tokenizer()?;
        embedder.load_or_download_model()?;
        let text1 = "Dogs are pets that bark.";
        let text2 = "Canines are domesticated animals that make barking sounds.";
        let text3 = "Quantum physics explores the nature of subatomic particles.";
        
        let emb1 = embedder.embed_text(text1)?;
        let emb2 = embedder.embed_text(text2)?;
        let emb3 = embedder.embed_text(text3)?;
        
        // Similar texts should have higher similarity
        let sim12 = embedder.cosine_similarity(&emb1, &emb2);
        let sim13 = embedder.cosine_similarity(&emb1, &emb3);
        
        // With real embeddings, we can now make assertions about similarity
        println!("Similarity between similar texts: {}", sim12);
        println!("Similarity between different texts: {}", sim13);
        
        // Similar texts should have higher similarity than dissimilar texts
        assert!(sim12 > sim13);
        
        Ok(())
    }
}
