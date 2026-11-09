//! Backend-independent tensor and buffer descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend { Cpu, Wgpu, Cuda }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape(pub Vec<usize>);
impl TensorShape {
    pub fn element_count(&self) -> Option<usize> {
        self.0.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType { F32, F16, I32, U32 }
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn shape_count() { assert_eq!(TensorShape(vec![2,3,4]).element_count(), Some(24)); }
    #[test] fn overflow() { assert_eq!(TensorShape(vec![usize::MAX,2]).element_count(), None); }
}
