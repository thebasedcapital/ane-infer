//! BPE tokenizer for GGUF models (GPT-2 style byte-level BPE).
//!
//! GPT-2 BPE maps each byte to a Unicode character to avoid tokenizer issues
//! with control chars. Space (0x20) → Ġ (U+0120), etc.

use std::collections::HashMap;

pub struct BpeTokenizer {
    /// Token ID → string (in GPT-2 unicode space)
    pub vocab: Vec<Vec<u8>>,
    /// Raw token string → token ID
    token_to_id: HashMap<Vec<u8>, u32>,
    /// Merge rules: (pair) → priority (lower = higher priority)
    merge_rank: HashMap<(Vec<u8>, Vec<u8>), usize>,
    /// EOS token ID
    pub eos_id: Option<u32>,
    /// Additional EOS token strings to check
    eos_strings: Vec<String>,
}

/// GPT-2 byte-to-unicode mapping.
/// Maps byte values to unicode codepoints such that no byte maps to
/// a whitespace/control character.
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;

    // Printable ASCII and Latin-1 supplement ranges pass through
    for b in 0u16..256 {
        let ch = b as u8;
        if (b'!'..=b'~').contains(&ch) || (0xA1..=0xAC).contains(&ch) || (0xAE..=0xFF).contains(&ch)
        {
            table[b as usize] = char::from(ch);
        }
    }

    // Everything else gets mapped to U+0100+
    n = 0;
    for b in 0u16..256 {
        let ch = b as u8;
        if !((b'!'..=b'~').contains(&ch)
            || (0xA1..=0xAC).contains(&ch)
            || (0xAE..=0xFF).contains(&ch))
        {
            table[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }

    table
}

/// Reverse: unicode char → byte
fn unicode_to_byte() -> HashMap<char, u8> {
    let b2u = byte_to_unicode();
    let mut u2b = HashMap::new();
    for (b, &c) in b2u.iter().enumerate() {
        u2b.insert(c, b as u8);
    }
    u2b
}

impl BpeTokenizer {
    pub fn from_gguf(
        token_strings: Vec<String>,
        merges: Option<Vec<String>>,
        eos_id: Option<u32>,
    ) -> Self {
        let mut token_to_id = HashMap::with_capacity(token_strings.len());
        let mut vocab = Vec::with_capacity(token_strings.len());

        for (i, t) in token_strings.iter().enumerate() {
            let bytes = t.as_bytes().to_vec();
            token_to_id.insert(bytes.clone(), i as u32);
            vocab.push(bytes);
        }

        let mut merge_rank = HashMap::new();
        if let Some(ref merge_list) = merges {
            for (i, m) in merge_list.iter().enumerate() {
                if let Some(space_idx) = m.find(' ') {
                    let a = m[..space_idx].as_bytes().to_vec();
                    let b = m[space_idx + 1..].as_bytes().to_vec();
                    merge_rank.insert((a, b), i);
                }
            }
        }

        // Collect known EOS strings
        let eos_strings = vec![
            "<|endoftext|>".to_string(),
            "<|im_end|>".to_string(),
            "<|end|>".to_string(),
            "</s>".to_string(),
            "<|eot_id|>".to_string(),
            "<|end_of_text|>".to_string(),
        ];

        Self {
            vocab,
            token_to_id,
            merge_rank,
            eos_id,
            eos_strings,
        }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Step 1: Convert text bytes to GPT-2 unicode characters
        let b2u = byte_to_unicode();
        let unicode_text: String = text.bytes().map(|b| b2u[b as usize]).collect();

        // Step 2: Split into initial tokens (each unicode char = one token)
        let mut tokens: Vec<Vec<u8>> = unicode_text
            .chars()
            .map(|c| {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                s.as_bytes().to_vec()
            })
            .collect();

        // Step 3: Apply BPE merges
        if !self.merge_rank.is_empty() {
            loop {
                if tokens.len() < 2 {
                    break;
                }

                let mut best_idx = None;
                let mut best_rank = usize::MAX;

                for i in 0..tokens.len() - 1 {
                    if let Some(&rank) = self
                        .merge_rank
                        .get(&(tokens[i].clone(), tokens[i + 1].clone()))
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = Some(i);
                        }
                    }
                }

                if let Some(idx) = best_idx {
                    let mut merged = tokens[idx].clone();
                    merged.extend_from_slice(&tokens[idx + 1]);
                    tokens[idx] = merged;
                    tokens.remove(idx + 1);
                } else {
                    break;
                }
            }
        }

        // Step 4: Map to token IDs
        tokens
            .iter()
            .filter_map(|t| self.token_to_id.get(t).copied())
            .collect()
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let u2b = unicode_to_byte();

        let mut bytes = Vec::new();
        for &id in ids {
            if (id as usize) < self.vocab.len() {
                let token_bytes = &self.vocab[id as usize];
                // Convert from GPT-2 unicode space back to raw bytes
                if let Ok(token_str) = std::str::from_utf8(token_bytes) {
                    for ch in token_str.chars() {
                        if let Some(&byte) = u2b.get(&ch) {
                            bytes.push(byte);
                        } else {
                            // Unknown char, try direct UTF-8
                            let mut buf = [0u8; 4];
                            let encoded = ch.encode_utf8(&mut buf);
                            bytes.extend_from_slice(encoded.as_bytes());
                        }
                    }
                } else {
                    bytes.extend_from_slice(token_bytes);
                }
            }
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Check if a token ID is EOS.
    pub fn is_eos(&self, id: u32) -> bool {
        if let Some(eos) = self.eos_id {
            if id == eos {
                return true;
            }
        }
        let idx = id as usize;
        if idx < self.vocab.len() {
            if let Ok(t) = std::str::from_utf8(&self.vocab[idx]) {
                return self.eos_strings.iter().any(|e| e == t);
            }
        }
        false
    }
}
