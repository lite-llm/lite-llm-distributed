#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveError {
    Unsupported,
    InvalidInput,
}

pub trait CollectiveOps {
    fn all_reduce_sum(&self, input: &[f32]) -> Result<Vec<f32>, CollectiveError>;
    fn all_to_all(&self, segments: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, CollectiveError>;
}

#[derive(Debug, Default)]
pub struct DeterministicCollectives;

impl CollectiveOps for DeterministicCollectives {
    fn all_reduce_sum(&self, input: &[f32]) -> Result<Vec<f32>, CollectiveError> {
        Ok(input.to_vec())
    }

    fn all_to_all(&self, segments: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, CollectiveError> {
        Ok(segments.to_vec())
    }
}

