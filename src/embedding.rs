use anyhow::Result;
use ndarray::Array1;

/// The Embedder trait defines the interface for text embedding implementations.
pub trait Embedder {
    /// Embeds a single text string into a vector representation.
    fn embed_text(&self, text: &str) -> Result<Array1<f32>>;
    
    /// Embeds multiple text strings into vector representations.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Array1<f32>>> {
        // Default implementation that calls embed_text for each string
        texts.iter()
            .map(|text| self.embed_text(text))
            .collect()
    }
    
    /// Computes the cosine similarity between two embedding vectors.
    fn cosine_similarity(&self, vec1: &Array1<f32>, vec2: &Array1<f32>) -> f32 {
        let dot_product = vec1.dot(vec2);
        let norm1 = vec1.dot(vec1).sqrt();
        let norm2 = vec2.dot(vec2).sqrt();
        
        if norm1 * norm2 == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm1 * norm2)
    }
} 