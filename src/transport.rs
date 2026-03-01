#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportBackend {
    Rdma,
    Nccl,
    Quic,
}

#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub backend: TransportBackend,
    pub endpoint: String,
}

pub trait Transport {
    fn send(&self, to_rank: usize, payload: &[u8]) -> Result<(), &'static str>;
    fn receive(&self, from_rank: usize) -> Result<Vec<u8>, &'static str>;
}

