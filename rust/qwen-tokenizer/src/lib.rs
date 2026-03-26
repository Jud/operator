use std::sync::Arc;
use tokenizers::Tokenizer;

uniffi::include_scaffolding!("qwen_tokenizer");

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("Failed to load tokenizer")]
    LoadFailed,
    #[error("Failed to encode text")]
    EncodeFailed,
    #[error("Failed to decode tokens")]
    DecodeFailed,
}

// UniFFI needs From<uniffi::UnexpectedUniFFICallbackError> for error types
impl From<uniffi::UnexpectedUniFFICallbackError> for TokenizerError {
    fn from(_: uniffi::UnexpectedUniFFICallbackError) -> Self {
        TokenizerError::LoadFailed
    }
}

const SUFFIX: &str = "</transcription>\n<|im_end|>\n<|im_start|>assistant\n";

pub struct QwenTokenizer {
    inner: Tokenizer,
    suffix_cache: Vec<i32>,
}

impl QwenTokenizer {
    pub fn encode(&self, text: String) -> Vec<i32> {
        match self.inner.encode(text.as_str(), false) {
            Ok(enc) => enc.get_ids().iter().map(|&id| id as i32).collect(),
            Err(_) => vec![],
        }
    }

    pub fn decode(&self, ids: Vec<i32>) -> Result<String, TokenizerError> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner
            .decode(&u32_ids, false)
            .map_err(|_| TokenizerError::DecodeFailed)
    }

    pub fn suffix_ids(&self) -> Vec<i32> {
        self.suffix_cache.clone()
    }
}

pub fn load_tokenizer(tokenizer_json_path: String) -> Result<Arc<QwenTokenizer>, TokenizerError> {
    let inner =
        Tokenizer::from_file(&tokenizer_json_path).map_err(|_| TokenizerError::LoadFailed)?;

    let suffix_cache = match inner.encode(SUFFIX, false) {
        Ok(enc) => enc.get_ids().iter().map(|&id| id as i32).collect(),
        Err(_) => return Err(TokenizerError::EncodeFailed),
    };

    Ok(Arc::new(QwenTokenizer {
        inner,
        suffix_cache,
    }))
}
