#[derive(Debug, Clone, Copy)]
pub struct ParallelismConfig {
    pub data_parallel: usize,
    pub tensor_parallel: usize,
    pub pipeline_parallel: usize,
    pub expert_parallel: usize,
}

impl ParallelismConfig {
    pub fn world_size(self) -> usize {
        self.data_parallel * self.tensor_parallel * self.pipeline_parallel * self.expert_parallel
    }
}

