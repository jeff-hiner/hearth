//! CLIP BPE tokenizer.
//!
//! Hand-rolled implementation of the CLIP text tokenizer pipeline:
//! NFC normalize → collapse whitespace → lowercase → GPT-2 regex pre-tokenize →
//! byte-level encode → BPE merge → sequence assembly.

use crate::types::Backend;
use burn::prelude::*;
use std::{collections::HashMap, path::Path};
use unicode_normalization::UnicodeNormalization as _;

/// Error type for tokenizer operations.
#[derive(Debug)]
pub enum TokenizerError {
    /// Failed to load tokenizer files.
    Load(String),
    /// Failed to encode text.
    Encode(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Load(msg) => write!(f, "tokenizer load error: {msg}"),
            TokenizerError::Encode(msg) => write!(f, "tokenizer encode error: {msg}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// CLIP BPE tokenizer.
///
/// Implements the full CLIP tokenization pipeline without external tokenizer
/// libraries. Loads vocabulary and merge rules from standard `vocab.json` and
/// `merges.txt` files.
pub struct ClipTokenizer {
    /// Token string → ID mapping (from `vocab.json`).
    encoder: HashMap<String, u32>,
    /// BPE merge pair → priority rank (lower = merge first).
    bpe_ranks: HashMap<(String, String), usize>,
}

impl ClipTokenizer {
    /// Beginning of sequence token ID.
    pub const BOS_TOKEN: u32 = 49406;
    /// End of sequence token ID (also used as padding for CLIP-L).
    pub const EOS_TOKEN: u32 = 49407;
    /// Padding token for OpenCLIP-G (zero, not EOS).
    pub const PAD_TOKEN_OPEN_CLIP: u32 = 0;
    /// Maximum sequence length.
    pub const MAX_LENGTH: usize = 77;

    /// Load tokenizer from vocab.json and merges.txt files.
    ///
    /// These files can be downloaded from HuggingFace's openai/clip-vit-large-patch14 repo.
    ///
    /// # Arguments
    /// * `vocab_path` - Path to vocab.json
    /// * `merges_path` - Path to merges.txt
    ///
    /// # Returns
    /// Loaded tokenizer or error
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> Result<Self, TokenizerError> {
        // Parse vocab.json
        let vocab_data = std::fs::read_to_string(vocab_path)
            .map_err(|e| TokenizerError::Load(format!("reading {}: {e}", vocab_path.display())))?;
        let encoder: HashMap<String, u32> = serde_json::from_str(&vocab_data)
            .map_err(|e| TokenizerError::Load(format!("parsing vocab.json: {e}")))?;

        // Parse merges.txt: skip header, each line is "left right" → rank by line index
        let merges_data = std::fs::read_to_string(merges_path)
            .map_err(|e| TokenizerError::Load(format!("reading {}: {e}", merges_path.display())))?;
        let mut bpe_ranks = HashMap::new();
        for (rank, line) in merges_data.lines().enumerate() {
            // Skip the #version header line
            if line.starts_with('#') {
                continue;
            }
            let Some((left, right)) = line.split_once(' ') else {
                continue;
            };
            bpe_ranks.insert((left.to_string(), right.to_string()), rank);
        }

        Ok(Self { encoder, bpe_ranks })
    }

    /// Encode text to token IDs tensor for CLIP-L (pads with EOS token).
    ///
    /// The output is padded/truncated to MAX_LENGTH (77) tokens:
    /// - Prepends BOS token (49406)
    /// - Appends EOS token (49407)
    /// - Pads with EOS to reach 77 tokens
    /// - Truncates to 77 tokens if longer
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Token IDs tensor of shape `[1, 77]`
    pub fn encode(
        &self,
        text: &str,
        device: &Device<Backend>,
    ) -> Result<Tensor<Backend, 2, Int>, TokenizerError> {
        self.encode_with_padding(text, Self::EOS_TOKEN, device)
    }

    /// Encode text to token IDs tensor for OpenCLIP-G (pads with 0).
    ///
    /// OpenCLIP uses 0 as the padding token instead of EOS. Using the wrong
    /// pad token changes the embedding of padding positions and can cause
    /// subtle divergence in the text encoder output.
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Token IDs tensor of shape `[1, 77]`
    pub fn encode_open_clip(
        &self,
        text: &str,
        device: &Device<Backend>,
    ) -> Result<Tensor<Backend, 2, Int>, TokenizerError> {
        self.encode_with_padding(text, Self::PAD_TOKEN_OPEN_CLIP, device)
    }

    /// Encode text with a configurable padding token.
    fn encode_with_padding(
        &self,
        text: &str,
        pad_token: u32,
        device: &Device<Backend>,
    ) -> Result<Tensor<Backend, 2, Int>, TokenizerError> {
        let token_ids = self.tokenize(text)?;

        // Build final sequence: BOS + tokens + EOS + padding
        let mut ids = Vec::with_capacity(Self::MAX_LENGTH);
        ids.push(Self::BOS_TOKEN);

        // Add encoded tokens (up to MAX_LENGTH - 2 to leave room for BOS and EOS)
        let max_content_tokens = Self::MAX_LENGTH - 2;
        for &id in token_ids.iter().take(max_content_tokens) {
            ids.push(id);
        }

        // Add EOS
        ids.push(Self::EOS_TOKEN);

        // Pad to MAX_LENGTH
        while ids.len() < Self::MAX_LENGTH {
            ids.push(pad_token);
        }

        // Convert to i32 for Burn's Int tensor
        let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();

        // Create tensor [1, 77]
        let tensor: Tensor<Backend, 1, Int> = Tensor::from_ints(ids_i32.as_slice(), device);
        Ok(tensor.unsqueeze())
    }

    /// Encode multiple texts to a batched token IDs tensor (CLIP-L padding).
    ///
    /// # Arguments
    /// * `texts` - Slice of input texts to encode
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Token IDs tensor of shape `[batch, 77]`
    pub fn encode_batch(
        &self,
        texts: &[&str],
        device: &Device<Backend>,
    ) -> Result<Tensor<Backend, 2, Int>, TokenizerError> {
        self.encode_batch_with_padding(texts, Self::EOS_TOKEN, device)
    }

    /// Encode multiple texts with OpenCLIP-G padding (pad with 0).
    ///
    /// # Arguments
    /// * `texts` - Slice of input texts to encode
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Token IDs tensor of shape `[batch, 77]`
    pub fn encode_batch_open_clip(
        &self,
        texts: &[&str],
        device: &Device<Backend>,
    ) -> Result<Tensor<Backend, 2, Int>, TokenizerError> {
        self.encode_batch_with_padding(texts, Self::PAD_TOKEN_OPEN_CLIP, device)
    }

    /// Encode multiple texts with a configurable padding token.
    fn encode_batch_with_padding(
        &self,
        texts: &[&str],
        pad_token: u32,
        device: &Device<Backend>,
    ) -> Result<Tensor<Backend, 2, Int>, TokenizerError> {
        let batch_size = texts.len();
        let mut all_ids = Vec::with_capacity(batch_size * Self::MAX_LENGTH);

        for text in texts {
            let token_ids = self.tokenize(text)?;

            // Build sequence: BOS + tokens + EOS + padding
            all_ids.push(Self::BOS_TOKEN as i32);

            let max_content_tokens = Self::MAX_LENGTH - 2;
            for &id in token_ids.iter().take(max_content_tokens) {
                all_ids.push(id as i32);
            }

            all_ids.push(Self::EOS_TOKEN as i32);

            // Pad
            let current_len = 2 + token_ids.len().min(max_content_tokens);
            for _ in current_len..Self::MAX_LENGTH {
                all_ids.push(pad_token as i32);
            }
        }

        let tensor: Tensor<Backend, 1, Int> = Tensor::from_ints(all_ids.as_slice(), device);
        Ok(tensor.reshape([batch_size, Self::MAX_LENGTH]))
    }

    /// Run the full tokenization pipeline: normalize → pre-tokenize → byte-encode → BPE.
    ///
    /// Returns raw token IDs (without BOS/EOS/padding).
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let normalized = self.normalize(text);
        let pre_tokens = self.pre_tokenize(&normalized);
        let mut ids = Vec::new();

        for token in &pre_tokens {
            // Byte-level encode: UTF-8 bytes → GPT-2 unicode chars
            let byte_encoded: String = token
                .as_bytes()
                .iter()
                .map(|&b| GPT2_BYTES_TO_UNICODE[b as usize])
                .collect();

            // Append </w> suffix (marks end of word, as CLIP uses)
            let bpe_input = format!("{byte_encoded}</w>");

            // Run BPE merges
            let merged = self.bpe(&bpe_input);

            // Look up each BPE piece in vocab
            for piece in &merged {
                if let Some(&id) = self.encoder.get(piece) {
                    ids.push(id);
                }
                // Unknown tokens are silently dropped (matches CLIP behavior)
            }
        }

        Ok(ids)
    }

    /// NFC normalize → collapse whitespace → lowercase.
    fn normalize(&self, text: &str) -> String {
        // NFC normalization (needed for CJK and accented characters)
        let nfc: String = text.nfc().collect();

        // Collapse whitespace runs to single space, trim, then lowercase
        let mut result = String::with_capacity(nfc.len());
        let mut prev_ws = true; // start true to trim leading whitespace
        for c in nfc.chars() {
            if c.is_whitespace() {
                if !prev_ws {
                    result.push(' ');
                    prev_ws = true;
                }
            } else {
                for lc in c.to_lowercase() {
                    result.push(lc);
                }
                prev_ws = false;
            }
        }
        // Trim trailing space
        if result.ends_with(' ') {
            result.pop();
        }
        result
    }

    /// Pre-tokenize using the CLIP/GPT-2 pattern (hand-rolled state machine).
    ///
    /// Matches the regex:
    /// ```text
    /// <\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
    /// ```
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            // Skip whitespace
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }

            // Try special tokens: <|startoftext|> or <|endoftext|>
            if chars[i] == '<' {
                let rest: String = chars[i..].iter().collect();
                if rest.starts_with("<|startoftext|>") {
                    tokens.push("<|startoftext|>".to_string());
                    i += "<|startoftext|>".len();
                    continue;
                }
                if rest.starts_with("<|endoftext|>") {
                    tokens.push("<|endoftext|>".to_string());
                    i += "<|endoftext|>".len();
                    continue;
                }
            }

            // Try contractions: 's, 't, 're, 've, 'm, 'll, 'd
            if chars[i] == '\'' && i + 1 < len {
                let next = chars[i + 1];
                // Check two-char contractions first
                if i + 2 < len {
                    let next2 = chars[i + 2];
                    if ((next == 'r' || next == 'v') && next2 == 'e')
                        || (next == 'l' && next2 == 'l')
                    {
                        let contraction: String = chars[i..i + 3].iter().collect();
                        tokens.push(contraction);
                        i += 3;
                        continue;
                    }
                }
                // Single-char contractions
                if next == 's' || next == 't' || next == 'm' || next == 'd' {
                    let contraction: String = chars[i..i + 2].iter().collect();
                    tokens.push(contraction);
                    i += 2;
                    continue;
                }
            }

            // Letter run: [\p{L}]+
            if chars[i].is_alphabetic() {
                let start = i;
                while i < len && chars[i].is_alphabetic() {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                tokens.push(word);
                continue;
            }

            // Single digit: [\p{N}]
            if chars[i].is_numeric() {
                tokens.push(chars[i].to_string());
                i += 1;
                continue;
            }

            // Punctuation / other run: [^\s\p{L}\p{N}]+
            let start = i;
            while i < len
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_numeric()
            {
                i += 1;
            }
            if i > start {
                let punct: String = chars[start..i].iter().collect();
                tokens.push(punct);
            }
        }

        tokens
    }

    /// Apply BPE merges to a token string.
    ///
    /// Takes a byte-encoded string (with `</w>` suffix already appended) and
    /// iteratively merges the highest-priority adjacent pair until no more
    /// merges are possible.
    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

        if word.len() <= 1 {
            return word;
        }

        loop {
            // Find the adjacent pair with the lowest rank (highest priority)
            let mut best_rank = usize::MAX;
            let mut best_pair = None;

            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.bpe_ranks.get(&pair)
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_pair = Some(pair);
                }
            }

            let Some((left, right)) = best_pair else {
                break;
            };

            // Merge all occurrences of the best pair
            let merged = format!("{left}{right}");
            let mut new_word = Vec::with_capacity(word.len());
            let mut j = 0;
            while j < word.len() {
                if j + 1 < word.len() && word[j] == left && word[j + 1] == right {
                    new_word.push(merged.clone());
                    j += 2;
                } else {
                    new_word.push(word[j].clone());
                    j += 1;
                }
            }
            word = new_word;

            if word.len() == 1 {
                break;
            }
        }

        word
    }
}

impl std::fmt::Debug for ClipTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipTokenizer")
            .field("max_length", &Self::MAX_LENGTH)
            .finish_non_exhaustive()
    }
}

/// GPT-2 bytes-to-unicode mapping table, computed at compile time.
///
/// Maps all 256 byte values to unicode characters. Printable ASCII bytes
/// (33–126, 161–172, 174–255) map to themselves; remaining bytes (control
/// characters etc.) map to code points starting at U+0100. This avoids
/// whitespace and control characters appearing in token strings.
const GPT2_BYTES_TO_UNICODE: [char; 256] = {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    let mut b = 0u16;

    while b < 256 {
        let c = b as u8;
        // Printable ranges that map to themselves:
        // '!' (33) through '~' (126), '¡' (161) through '¬' (172), '®' (174) through 'ÿ' (255)
        if (c >= 33 && c <= 126) || (c >= 161 && c <= 172) || c >= 174 {
            table[b as usize] = c as char;
        } else {
            // Map to U+0100 + n; all values are < 512, well within valid unicode
            table[b as usize] =
                char::from_u32(256 + n).expect("GPT-2 byte table: invalid code point");
            n += 1;
        }
        b += 1;
    }

    table
};
