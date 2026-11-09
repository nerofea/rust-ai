//! CPU reference runtime; GPU backends are future feature-gated implementations.
use rust_ai_gpu_core::TensorShape;
pub fn add_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>, &'static str> {
    if a.len() != b.len() { return Err("shape mismatch"); }
    Ok(a.iter().zip(b).map(|(x,y)| x+y).collect())
}
pub fn validate_shape(shape: &TensorShape, len: usize) -> Result<(), &'static str> {
    if shape.element_count() != Some(len) { return Err("invalid shape"); }
    Ok(())
}
#[cfg(test)]
mod tests { use super::*; #[test] fn add() { assert_eq!(add_f32(&[1.,2.], &[3.,4.]).unwrap(), vec![4.,6.]); } }
