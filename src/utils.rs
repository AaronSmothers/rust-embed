use anyhow::Result;
use bytes::{Bytes, BytesMut};
use ndarray::Array1;
use prost::Message;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::models::proto::{Embedding, EmbeddingCollection};

/// Saves embedding vectors to a Protobuf file
pub fn save_embeddings<P: AsRef<Path>>(
    embeddings: &[Array1<f32>], 
    texts: Option<&[String]>,
    model_name: &str,
    model_version: &str,
    dimension: i32,
    path: P
) -> Result<()> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    
    // Create embedding collection
    let mut collection = EmbeddingCollection {
        embeddings: Vec::with_capacity(embeddings.len()),
        model_name: model_name.to_string(),
        model_version: model_version.to_string(),
        dimension,
    };
    
    // Add each embedding
    for (i, arr) in embeddings.iter().enumerate() {
        let values = arr.iter().copied().collect();
        let text = match texts {
            Some(texts) if i < texts.len() => texts[i].clone(),
            _ => String::new(),
        };
        
        collection.embeddings.push(Embedding {
            values,
            text,
            timestamp,
        });
    }
    
    // Serialize to bytes
    let mut buf = BytesMut::with_capacity(collection.encoded_len());
    collection.encode(&mut buf)?;
    
    // Write to file
    fs::write(path, buf.freeze())?;
    
    Ok(())
}

/// Loads embedding vectors from a Protobuf file
pub fn load_embeddings<P: AsRef<Path>>(path: P) -> Result<(Vec<Array1<f32>>, Option<Vec<String>>)> {
    // Read the file
    let data = fs::read(path)?;
    let collection = EmbeddingCollection::decode(Bytes::from(data))?;
    
    // Extract embeddings and texts
    let mut embeddings = Vec::with_capacity(collection.embeddings.len());
    let mut texts = Vec::with_capacity(collection.embeddings.len());
    let has_texts = collection.embeddings.iter().any(|e| !e.text.is_empty());
    
    for embedding in collection.embeddings {
        embeddings.push(Array1::from(embedding.values));
        if has_texts {
            texts.push(embedding.text);
        }
    }
    
    let texts = if has_texts { Some(texts) } else { None };
    
    Ok((embeddings, texts))
}

/// Normalizes a vector to unit length
pub fn normalize(vec: &mut Array1<f32>) {
    let norm = vec.dot(vec).sqrt();
    if norm > 0.0 {
        vec.mapv_inplace(|x| x / norm);
    }
}

/// Preprocesses text for embedding
pub fn preprocess_text(text: &str) -> String {
    // Simple preprocessing: trim, lowercase, collapse whitespace
    let text = text.trim().to_lowercase();
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    text
} 