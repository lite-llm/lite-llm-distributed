use crate::error::{DistributedError, DistributedResult};
use crate::parallelism::ExpertAddress;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsensusConfig {
    pub seed: u64,
    pub quantization_scale: f32,
}

impl ConsensusConfig {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            quantization_scale: 1_000_000.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenRoute {
    pub token_index: u32,
    pub experts: Vec<ExpertAddress>,
    pub checksum: u64,
}

#[derive(Debug, Clone)]
pub struct RoutingConsensus {
    config: ConsensusConfig,
}

impl RoutingConsensus {
    pub fn new(config: ConsensusConfig) -> Self {
        Self { config }
    }

    pub fn select_experts(
        &self,
        hidden_state: &[f32],
        token_index: u32,
        layer: u32,
        candidates: &[ExpertAddress],
        k: usize,
    ) -> DistributedResult<TokenRoute> {
        if candidates.is_empty() {
            return Err(DistributedError::InvalidCollectiveInput(
                "candidates cannot be empty",
            ));
        }
        if k == 0 {
            return Err(DistributedError::InvalidCollectiveInput(
                "k must be greater than zero",
            ));
        }

        let token_seed = derive_token_seed(self.config.seed, layer, token_index);
        let mut scored: Vec<(usize, i64, u64)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let score = projection_score(hidden_state, token_seed, *candidate);
                let quantized = quantize(score, self.config.quantization_scale.max(1.0));
                let tie = seeded_index_hash(token_seed, idx as u64);
                (idx, quantized, tie)
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(a.2.cmp(&b.2))
                .then(a.0.cmp(&b.0))
        });

        let selected: Vec<ExpertAddress> = scored
            .into_iter()
            .take(k.min(candidates.len()))
            .map(|entry| candidates[entry.0])
            .collect();

        let checksum = route_checksum(&selected);
        Ok(TokenRoute {
            token_index,
            experts: selected,
            checksum,
        })
    }

    pub fn verify_checksums(&self, rank_checksums: &[(usize, u64)]) -> DistributedResult<u64> {
        if rank_checksums.is_empty() {
            return Err(DistributedError::InvalidCollectiveInput(
                "rank checksums cannot be empty",
            ));
        }

        let expected = rank_checksums[0].1;
        for (rank, checksum) in rank_checksums {
            if *checksum != expected {
                return Err(DistributedError::ConsensusMismatch {
                    expected,
                    found: *checksum,
                    rank: *rank,
                });
            }
        }

        Ok(expected)
    }
}

pub fn route_checksum(experts: &[ExpertAddress]) -> u64 {
    let mut bytes = Vec::with_capacity(experts.len() * 10);
    for expert in experts {
        bytes.extend_from_slice(&expert.tier.to_le_bytes());
        bytes.extend_from_slice(&expert.group.to_le_bytes());
        bytes.extend_from_slice(&expert.expert.to_le_bytes());
    }
    fnv1a64(0xcbf29ce484222325, &bytes)
}

fn derive_token_seed(base_seed: u64, layer: u32, token_index: u32) -> u64 {
    let mut payload = [0u8; 16];
    payload[0..8].copy_from_slice(&base_seed.to_le_bytes());
    payload[8..12].copy_from_slice(&layer.to_le_bytes());
    payload[12..16].copy_from_slice(&token_index.to_le_bytes());
    fnv1a64(0xcbf29ce484222325, &payload)
}

fn seeded_index_hash(seed: u64, index: u64) -> u64 {
    let mut payload = [0u8; 16];
    payload[0..8].copy_from_slice(&seed.to_le_bytes());
    payload[8..16].copy_from_slice(&index.to_le_bytes());
    fnv1a64(0xcbf29ce484222325, &payload)
}

fn projection_score(hidden_state: &[f32], seed: u64, candidate: ExpertAddress) -> f32 {
    if hidden_state.is_empty() {
        return 0.0;
    }

    let mut candidate_payload = [0u8; 10];
    candidate_payload[0..2].copy_from_slice(&candidate.tier.to_le_bytes());
    candidate_payload[2..6].copy_from_slice(&candidate.group.to_le_bytes());
    candidate_payload[6..10].copy_from_slice(&candidate.expert.to_le_bytes());
    let candidate_seed = fnv1a64(seed, &candidate_payload);

    let mut score = 0.0_f32;
    for (idx, value) in hidden_state.iter().enumerate() {
        let hash = seeded_index_hash(candidate_seed, idx as u64);
        let signed = ((hash as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
        score += *value * signed;
    }
    score
}

fn quantize(value: f32, scale: f32) -> i64 {
    let scaled = (value as f64 * scale as f64).round();
    if scaled > i64::MAX as f64 {
        i64::MAX
    } else if scaled < i64::MIN as f64 {
        i64::MIN
    } else {
        scaled as i64
    }
}

fn fnv1a64(seed: u64, payload: &[u8]) -> u64 {
    let mut hash = seed;
    for byte in payload {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{ConsensusConfig, RoutingConsensus};
    use crate::parallelism::ExpertAddress;

    #[test]
    fn selection_is_deterministic_with_same_seed() {
        let consensus = RoutingConsensus::new(ConsensusConfig::new(1234));
        let hidden = vec![0.1, -0.2, 0.3, 0.4];
        let candidates = vec![
            ExpertAddress {
                tier: 1,
                group: 0,
                expert: 0,
            },
            ExpertAddress {
                tier: 1,
                group: 0,
                expert: 1,
            },
            ExpertAddress {
                tier: 10,
                group: 1,
                expert: 0,
            },
        ];

        let first = consensus
            .select_experts(&hidden, 0, 0, &candidates, 2)
            .expect("selection should succeed");
        let second = consensus
            .select_experts(&hidden, 0, 0, &candidates, 2)
            .expect("selection should succeed");

        assert_eq!(first, second);
    }
}
