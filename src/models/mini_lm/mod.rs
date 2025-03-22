use crate::embedding::Embedder;
use crate::models::ModelConfig;
use anyhow::{anyhow, Result, Context};
use ndarray::Array1;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tch::Device;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};

// Constants for the MiniLM model (all-MiniLM-L6-v2)
const MODEL_NAME: &str = "all-MiniLM-L6-v2";
const MODEL_VERSION: &str = "v1.0";
const EMBEDDING_DIM: usize = 384; // Dimension of MiniLM embedding vectors
const TOKENIZER_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/tokenizer.json";

/// Configuration for the MiniLM model
#[derive(Debug, Clone)]
pub struct MiniLMConfig {
    pub model_name: String,
    pub model_version: String,
    pub dimension: usize,
    pub model_path: Option<PathBuf>,
    pub device: Device,
}

impl Default for MiniLMConfig {
    fn default() -> Self {
        Self {
            model_name: MODEL_NAME.to_string(),
            model_version: MODEL_VERSION.to_string(),
            dimension: EMBEDDING_DIM,
            model_path: None,
            device: Device::Cpu,
        }
    }
}

impl ModelConfig for MiniLMConfig {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_version(&self) -> &str {
        &self.model_version
    }
}

/// MiniLM embedder implementation
pub struct MiniLMEmbedder {
    config: MiniLMConfig,
    tokenizer: Option<Tokenizer>,
    embeddings_model: Option<rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel>,
}

impl MiniLMEmbedder {
    /// Create a new MiniLMEmbedder with default configuration
    pub fn new() -> Self {
        Self {
            config: MiniLMConfig::default(),
            tokenizer: None,
            embeddings_model: None,
        }
    }

    /// Create a new MiniLMEmbedder with the given configuration
    pub fn with_config(config: MiniLMConfig) -> Self {
        Self {
            config,
            tokenizer: None,
            embeddings_model: None,
        }
    }

    /// Get the model dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
    
    /// Get the model name
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }
    
    /// Get the model version
    pub fn model_version(&self) -> &str {
        &self.config.model_version
    }

    /// Save the model to disk
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(path.as_ref()).context("Failed to create model directory")?;
        
        // Save tokenizer if available
        if let Some(tokenizer) = &self.tokenizer {
            let tokenizer_path = path.as_ref().join("tokenizer.json");
            tokenizer.save(&tokenizer_path.to_string_lossy().to_string(), false)
                .map_err(|e| anyhow!("Failed to save tokenizer: {}", e))?;
        }
        
        // Save model info
        let info_path = path.as_ref().join("model_info.txt");
        let mut file = File::create(info_path).context("Failed to create model info file")?;
        file.write_all(format!("MiniLM model: {}\nVersion: {}\nDimension: {}", 
            self.config.model_name, 
            self.config.model_version,
            self.config.dimension
        ).as_bytes()).context("Failed to write model info")?;
        
        Ok(())
    }

    /// Load the model from disk
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut embedder = Self::new();
        
        // Check if path exists
        if !path.as_ref().exists() {
            return Err(anyhow!("Model path does not exist: {:?}", path.as_ref()));
        }
        
        // Load tokenizer if available
        let tokenizer_path = path.as_ref().join("tokenizer.json");
        if tokenizer_path.exists() {
            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
            embedder.tokenizer = Some(tokenizer);
        }
        
        // Load model info if available
        let info_path = path.as_ref().join("model_info.txt");
        if info_path.exists() {
            let mut info_content = String::new();
            let mut file = File::open(info_path).context("Failed to open model info file")?;
            file.read_to_string(&mut info_content).context("Failed to read model info file")?;
            
            // Parse model info (very basic implementation)
            for line in info_content.lines() {
                if line.starts_with("MiniLM model:") {
                    embedder.config.model_name = line.strip_prefix("MiniLM model:").unwrap_or("").trim().to_string();
                } else if line.starts_with("Version:") {
                    embedder.config.model_version = line.strip_prefix("Version:").unwrap_or("").trim().to_string();
                } else if line.starts_with("Dimension:") {
                    if let Ok(dim) = line.strip_prefix("Dimension:").unwrap_or("").trim().parse::<usize>() {
                        embedder.config.dimension = dim;
                    }
                }
            }
        }
        
        // Initialize the embedding model
        embedder.load_or_download_model().context("Failed to load model")?;
        
        Ok(embedder)
    }

    /// Tokenize the input text
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if let Some(tokenizer) = &self.tokenizer {
            let encoding = tokenizer.encode(text, false)
                .map_err(|e| anyhow!("Tokenization error: {}", e))?;
            
            // Convert token IDs to tokens
            let tokens = encoding.get_tokens().to_vec();
            Ok(tokens)
        } else {
            // Fallback to simple whitespace tokenization
            let tokens = text.split_whitespace().map(String::from).collect();
            Ok(tokens)
        }
    }

    /// Load or download the tokenizer
    pub fn load_or_download_tokenizer(&mut self) -> Result<()> {
        // First check if model path is specified and contains a tokenizer
        if let Some(model_path) = &self.config.model_path {
            let tokenizer_path = model_path.join("tokenizer.json");
            if tokenizer_path.exists() {
                let tokenizer = Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| anyhow!("Failed to load tokenizer from {}: {}", 
                        tokenizer_path.display(), e))?;
                self.tokenizer = Some(tokenizer);
                return Ok(());
            }
        }
        
        // If no tokenizer found, download from HuggingFace
        let cache_dir = match dirs::cache_dir() {
            Some(dir) => dir.join("rust_embed").join(MODEL_NAME),
            None => std::env::temp_dir().join("rust_embed").join(MODEL_NAME),
        };
        
        std::fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;
        
        let tokenizer_path = cache_dir.join("tokenizer.json");
        
        // Download tokenizer if it doesn't exist
        if !tokenizer_path.exists() {
            println!("Downloading tokenizer from {}", TOKENIZER_URL);
            let tokenizer_content = reqwest::blocking::get(TOKENIZER_URL)
                .context("Failed to download tokenizer")?
                .text()
                .context("Failed to read tokenizer content")?;
                
            let mut file = File::create(&tokenizer_path).context("Failed to create tokenizer file")?;
            file.write_all(tokenizer_content.as_bytes()).context("Failed to write tokenizer file")?;
        }
        
        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        self.tokenizer = Some(tokenizer);
        
        // Update model path
        self.config.model_path = Some(cache_dir);
        
        Ok(())
    }

    /// Load or download the model
    pub fn load_or_download_model(&mut self) -> Result<()> {
        // Initialize the sentence embedding model from rust-bert
        println!("Initializing the all-MiniLM-L6-v2 model from rust-bert...");
        
        // Use the SentenceEmbeddingsBuilder to create a sentence embeddings model
        let sentence_model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL6V2
        )
        .with_device(self.config.device)
        .create_model()?;
        
        self.embeddings_model = Some(sentence_model);
        println!("Model loaded successfully!");
        
        Ok(())
    }

    /// Calculate cosine similarity between two embedding vectors
    pub fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        // Calculate dot product
        let dot_product = a.dot(b);
        
        // Calculate magnitudes
        let mag_a = a.dot(a).sqrt();
        let mag_b = b.dot(b).sqrt();
        
        // Return cosine similarity
        if mag_a > 0.0 && mag_b > 0.0 {
            dot_product / (mag_a * mag_b)
        } else {
            0.0
        }
    }
}

impl Embedder for MiniLMEmbedder {
    fn embed_text(&self, text: &str) -> Result<Array1<f32>> {
        // Use the actual model to generate embeddings
        if let Some(model) = &self.embeddings_model {
            // Get embeddings using the rust-bert model
            let embeddings = model.encode(&[text])?;
            
            // Convert the first embedding to ndarray Array1
            if let Some(embedding_vec) = embeddings.first() {
                let embedding = Array1::from(embedding_vec.clone());
                Ok(embedding)
            } else {
                Err(anyhow!("Failed to generate embedding: empty result"))
            }
        } else {
            Err(anyhow!("Model not loaded. Call load_or_download_model() first."))
        }
    }
} 