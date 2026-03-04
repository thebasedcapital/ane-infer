//! GGUF file format parser
//!
//! GGUF v3 format:
//! - Magic: "GGUF" (4 bytes)
//! - Version: u32
//! - Tensor count: u64
//! - Metadata KV count: u64
//! - Metadata KV pairs
//! - Tensor info array
//! - Padding to alignment
//! - Tensor data

use std::collections::HashMap;
use std::io::{Read, Cursor};
use anyhow::{Result, bail, Context};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in LE

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => bail!("unknown GGUF type: {v}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self> {
        // Only map the ones we care about
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            _ => bail!("unsupported GGML type: {v}"),
        }
    }

    /// Block size for quantized types (number of elements per block)
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
            _ => 1,
        }
    }

    /// Bytes per block for quantized types
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,    // 2 (scale) + 16 (32 nibbles)
            Self::Q4_1 => 20,    // 2 (scale) + 2 (min) + 16
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,    // 2 (scale) + 32 (bytes)
            Self::Q4K => 144,    // complex block structure
            Self::Q5K => 176,
            Self::Q6K => 210,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub typ: GgmlType,
    pub offset: u64, // offset from start of tensor data section
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    pub fn data_size(&self) -> u64 {
        let n = self.n_elements() as usize;
        let bs = self.typ.block_size();
        let bpb = self.typ.bytes_per_block();
        ((n + bs - 1) / bs * bpb) as u64
    }
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: u64, // absolute file offset where tensor data starts
}

fn read_u8(r: &mut impl Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(r: &mut impl Read) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(r: &mut impl Read) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).context("invalid UTF-8 in GGUF string")
}

fn read_metadata_value(r: &mut impl Read, typ: GgufType) -> Result<MetadataValue> {
    match typ {
        GgufType::U8 => Ok(MetadataValue::U8(read_u8(r)?)),
        GgufType::I8 => Ok(MetadataValue::I8(read_u8(r)? as i8)),
        GgufType::U16 => Ok(MetadataValue::U16(read_u16(r)?)),
        GgufType::I16 => Ok(MetadataValue::I16(read_u16(r)? as i16)),
        GgufType::U32 => Ok(MetadataValue::U32(read_u32(r)?)),
        GgufType::I32 => Ok(MetadataValue::I32(read_i32(r)?)),
        GgufType::F32 => Ok(MetadataValue::F32(read_f32(r)?)),
        GgufType::U64 => Ok(MetadataValue::U64(read_u64(r)?)),
        GgufType::I64 => Ok(MetadataValue::I64(read_i64(r)?)),
        GgufType::F64 => Ok(MetadataValue::F64(read_f64(r)?)),
        GgufType::Bool => Ok(MetadataValue::Bool(read_u8(r)? != 0)),
        GgufType::String => Ok(MetadataValue::String(read_string(r)?)),
        GgufType::Array => {
            let elem_type = GgufType::from_u32(read_u32(r)?)?;
            let count = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(count.min(10000));
            for _ in 0..count {
                arr.push(read_metadata_value(r, elem_type)?);
            }
            Ok(MetadataValue::Array(arr))
        }
    }
}

impl GgufFile {
    /// Parse a GGUF file from a byte slice.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut r = Cursor::new(data);

        let magic = read_u32(&mut r)?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (magic: 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x})");
        }

        let version = read_u32(&mut r)?;
        if version < 2 || version > 3 {
            bail!("unsupported GGUF version: {version}");
        }

        let tensor_count = read_u64(&mut r)? as usize;
        let metadata_kv_count = read_u64(&mut r)? as usize;

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut r)?;
            let val_type = GgufType::from_u32(read_u32(&mut r)?)?;
            let val = read_metadata_value(&mut r, val_type)?;
            metadata.insert(key, val);
        }

        // Parse tensor info
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)? as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dimensions.push(read_u64(&mut r)?);
            }
            let typ = GgmlType::from_u32(read_u32(&mut r)?)?;
            let offset = read_u64(&mut r)?;
            tensors.push(TensorInfo { name, dimensions, typ, offset });
        }

        // Tensor data starts after alignment padding
        let pos = r.position();
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;
        let tensor_data_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(Self { version, metadata, tensors, tensor_data_offset })
    }

    /// Get a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get the architecture name (e.g., "llama")
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    /// Get model dimension from metadata.
    pub fn embedding_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.embedding_length");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get number of attention heads.
    pub fn head_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.attention.head_count");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get number of KV heads (for GQA).
    pub fn head_count_kv(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.attention.head_count_kv");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get number of layers.
    pub fn block_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.block_count");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get feed-forward hidden dimension.
    pub fn feed_forward_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.feed_forward_length");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> Option<u32> {
        // Try tokenizer.ggml.tokens array length
        if let Some(MetadataValue::Array(arr)) = self.metadata.get("tokenizer.ggml.tokens") {
            return Some(arr.len() as u32);
        }
        None
    }

    /// Get RoPE frequency base.
    pub fn rope_freq_base(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.rope.freq_base");
        self.metadata.get(&key)?.as_f32()
    }

    /// Get context length.
    pub fn context_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.context_length");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get key length (head dimension for keys).
    pub fn key_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.attention.key_length");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get value length (head dimension for values).
    pub fn value_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.attention.value_length");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get RMS norm epsilon.
    pub fn rms_norm_eps(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.attention.layer_norm_rms_epsilon");
        self.metadata.get(&key)?.as_f32()
    }

    // --- SSM / DeltaNet specific ---

    /// Get full attention interval (e.g. 4 = every 4th layer is full attention).
    pub fn full_attention_interval(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.full_attention_interval");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get SSM state size (recurrent state dimension).
    pub fn ssm_state_size(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.ssm.state_size");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get SSM convolution kernel size.
    pub fn ssm_conv_kernel(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.ssm.conv_kernel");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get SSM inner size (projection dimension).
    pub fn ssm_inner_size(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.ssm.inner_size");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get SSM group count (number of heads for DeltaNet).
    pub fn ssm_group_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.ssm.group_count");
        self.metadata.get(&key)?.as_u32()
    }

    /// Get SSM time step rank.
    pub fn ssm_time_step_rank(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.ssm.time_step_rank");
        self.metadata.get(&key)?.as_u32()
    }

    /// Determine if a layer is DeltaNet or full attention based on tensor names.
    pub fn is_deltanet_layer(&self, layer_idx: usize) -> bool {
        let name = format!("blk.{layer_idx}.ssm_a");
        self.get_tensor(&name).is_some()
    }

    /// Get all metadata keys (for debugging).
    pub fn metadata_keys(&self) -> Vec<&String> {
        let mut keys: Vec<_> = self.metadata.keys().collect();
        keys.sort();
        keys
    }

    /// Extract tokenizer vocabulary (token strings).
    pub fn tokenizer_tokens(&self) -> Option<Vec<String>> {
        if let Some(MetadataValue::Array(arr)) = self.metadata.get("tokenizer.ggml.tokens") {
            let mut tokens = Vec::with_capacity(arr.len());
            for v in arr {
                match v {
                    MetadataValue::String(s) => tokens.push(s.clone()),
                    _ => tokens.push(String::new()),
                }
            }
            Some(tokens)
        } else {
            None
        }
    }

    /// Extract BPE merge rules.
    pub fn tokenizer_merges(&self) -> Option<Vec<String>> {
        if let Some(MetadataValue::Array(arr)) = self.metadata.get("tokenizer.ggml.merges") {
            let mut merges = Vec::with_capacity(arr.len());
            for v in arr {
                match v {
                    MetadataValue::String(s) => merges.push(s.clone()),
                    _ => {}
                }
            }
            Some(merges)
        } else {
            None
        }
    }

    /// Get the tokenizer model type (e.g., "gpt2", "llama").
    pub fn tokenizer_model(&self) -> Option<&str> {
        self.metadata.get("tokenizer.ggml.model")?.as_str()
    }

    /// Get EOS token ID.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.metadata.get("tokenizer.ggml.eos_token_id")?.as_u32()
    }
}
