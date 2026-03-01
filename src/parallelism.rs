use crate::error::{DistributedError, DistributedResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertAddress {
    pub tier: u16,
    pub group: u32,
    pub expert: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RankCoordinate {
    pub dp: usize,
    pub tp: usize,
    pub pp: usize,
    pub ep: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelismConfig {
    pub data_parallel: usize,
    pub tensor_parallel: usize,
    pub pipeline_parallel: usize,
    pub expert_parallel: usize,
}

impl ParallelismConfig {
    pub fn validate(self) -> DistributedResult<()> {
        if self.data_parallel == 0 {
            return Err(DistributedError::InvalidTopology(
                "data_parallel must be greater than zero",
            ));
        }
        if self.tensor_parallel == 0 {
            return Err(DistributedError::InvalidTopology(
                "tensor_parallel must be greater than zero",
            ));
        }
        if self.pipeline_parallel == 0 {
            return Err(DistributedError::InvalidTopology(
                "pipeline_parallel must be greater than zero",
            ));
        }
        if self.expert_parallel == 0 {
            return Err(DistributedError::InvalidTopology(
                "expert_parallel must be greater than zero",
            ));
        }
        Ok(())
    }

    pub fn world_size(self) -> usize {
        self.data_parallel
            .saturating_mul(self.tensor_parallel)
            .saturating_mul(self.pipeline_parallel)
            .saturating_mul(self.expert_parallel)
    }

    pub fn rank_to_coordinate(self, rank: usize) -> DistributedResult<RankCoordinate> {
        self.validate()?;
        let world = self.world_size();
        if rank >= world {
            return Err(DistributedError::RankOutOfRange {
                rank,
                world_size: world,
            });
        }

        let mut remainder = rank;
        let ep = remainder % self.expert_parallel;
        remainder /= self.expert_parallel;

        let pp = remainder % self.pipeline_parallel;
        remainder /= self.pipeline_parallel;

        let tp = remainder % self.tensor_parallel;
        remainder /= self.tensor_parallel;

        let dp = remainder;

        Ok(RankCoordinate { dp, tp, pp, ep })
    }

    pub fn coordinate_to_rank(self, coordinate: RankCoordinate) -> DistributedResult<usize> {
        self.validate()?;

        if coordinate.dp >= self.data_parallel
            || coordinate.tp >= self.tensor_parallel
            || coordinate.pp >= self.pipeline_parallel
            || coordinate.ep >= self.expert_parallel
        {
            return Err(DistributedError::InvalidTopology(
                "coordinate dimension exceeds configured parallelism",
            ));
        }

        let rank = (((coordinate.dp * self.tensor_parallel + coordinate.tp) * self.pipeline_parallel
            + coordinate.pp)
            * self.expert_parallel)
            + coordinate.ep;

        Ok(rank)
    }

    pub fn expert_owner_rank(
        self,
        coordinate_prefix: RankCoordinate,
        expert: ExpertAddress,
        seed: u64,
    ) -> DistributedResult<usize> {
        self.validate()?;

        let mut base = coordinate_prefix;
        base.ep = 0;
        let base_rank = self.coordinate_to_rank(base)?;
        let hash = hash_expert(expert, seed);
        let owner_offset = (hash % self.expert_parallel as u64) as usize;
        Ok(base_rank + owner_offset)
    }
}

fn hash_expert(expert: ExpertAddress, seed: u64) -> u64 {
    let mut payload = [0u8; 18];
    payload[0..8].copy_from_slice(&seed.to_le_bytes());
    payload[8..10].copy_from_slice(&expert.tier.to_le_bytes());
    payload[10..14].copy_from_slice(&expert.group.to_le_bytes());
    payload[14..18].copy_from_slice(&expert.expert.to_le_bytes());
    fnv1a64(0xcbf29ce484222325, &payload)
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
    use super::{ExpertAddress, ParallelismConfig, RankCoordinate};

    #[test]
    fn coordinate_roundtrip_is_lossless() {
        let cfg = ParallelismConfig {
            data_parallel: 2,
            tensor_parallel: 2,
            pipeline_parallel: 2,
            expert_parallel: 2,
        };

        for rank in 0..cfg.world_size() {
            let coord = cfg
                .rank_to_coordinate(rank)
                .expect("coordinate conversion should succeed");
            let restored = cfg
                .coordinate_to_rank(coord)
                .expect("rank restoration should succeed");
            assert_eq!(rank, restored);
        }
    }

    #[test]
    fn expert_owner_is_deterministic() {
        let cfg = ParallelismConfig {
            data_parallel: 1,
            tensor_parallel: 1,
            pipeline_parallel: 1,
            expert_parallel: 8,
        };
        let prefix = RankCoordinate {
            dp: 0,
            tp: 0,
            pp: 0,
            ep: 0,
        };
        let expert = ExpertAddress {
            tier: 1,
            group: 2,
            expert: 3,
        };

        let first = cfg
            .expert_owner_rank(prefix, expert, 55)
            .expect("owner calculation should succeed");
        let second = cfg
            .expert_owner_rank(prefix, expert, 55)
            .expect("owner calculation should succeed");

        assert_eq!(first, second);
    }
}
