use std::convert::TryInto;
use std::io::{self, Read, Write};

const DIM: usize = 256;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn add_feature(acc: &mut [f32; DIM], feature: &[u8], weight: f32) {
    let h = fnv1a(feature);
    let idx = (h as usize) & (DIM - 1);
    let sign = if (h & 0x100) != 0 { 1.0 } else { -1.0 };
    acc[idx] += sign * weight;
}

fn is_token_start(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

fn is_token_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'/' | b'-')
}

fn lower_ascii_slice(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    for &b in src {
        out.push(b.to_ascii_lowercase());
    }
    out
}

fn vectorize_one(text: &[u8], out: &mut [u8]) {
    let mut acc = [0.0_f32; DIM];
    let mut tokens: Vec<Vec<u8>> = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if !is_token_start(text[i]) {
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        while i < text.len() && is_token_continue(text[i]) {
            i += 1;
        }
        let tok = lower_ascii_slice(&text[start..i]);
        add_feature(&mut acc, &tok, 1.0);
        if tok.len() >= 8 {
            for gram_start in 0..=(tok.len() - 4) {
                add_feature(&mut acc, &tok[gram_start..gram_start + 4], 0.25);
            }
        }
        tokens.push(tok);
    }

    let mut bigram = Vec::new();
    for pair in tokens.windows(2) {
        bigram.clear();
        bigram.extend_from_slice(&pair[0]);
        bigram.push(b' ');
        bigram.extend_from_slice(&pair[1]);
        add_feature(&mut acc, &bigram, 0.5);
    }

    let norm_sq: f32 = acc.iter().map(|v| v * v).sum();
    if norm_sq <= 1e-24 {
        out.fill(0);
        return;
    }
    let norm = norm_sq.sqrt();
    for (i, v) in acc.iter().enumerate() {
        let scaled = ((*v / norm).clamp(-1.0, 1.0) * 127.0).round() as i8;
        out[i] = scaled as u8;
    }
}

fn read_u32(input: &[u8], pos: &mut usize) -> io::Result<u32> {
    if input.len().saturating_sub(*pos) < 4 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short u32"));
    }
    let bytes: [u8; 4] = input[*pos..*pos + 4].try_into().unwrap();
    *pos += 4;
    Ok(u32::from_le_bytes(bytes))
}

fn main() -> io::Result<()> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;
    let mut pos = 0;
    let count = read_u32(&input, &mut pos)? as usize;
    let mut output = vec![0_u8; count * DIM];
    for row in 0..count {
        let len = read_u32(&input, &mut pos)? as usize;
        if input.len().saturating_sub(pos) < len {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short text"));
        }
        let text = &input[pos..pos + len];
        pos += len;
        vectorize_one(text, &mut output[row * DIM..(row + 1) * DIM]);
    }
    io::stdout().write_all(&output)?;
    Ok(())
}
