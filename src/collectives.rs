use crate::error::{DistributedError, DistributedResult};

pub trait CollectiveOps {
    fn all_reduce_sum(&self, rank_inputs: &[Vec<f32>]) -> DistributedResult<Vec<Vec<f32>>>;
    fn all_to_all(&self, rank_payloads: &[Vec<Vec<u8>>]) -> DistributedResult<Vec<Vec<Vec<u8>>>>;
    fn global_order(&self) -> &[usize];
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterministicCollectives {
    world_size: usize,
    order: Vec<usize>,
}

impl DeterministicCollectives {
    pub fn new(world_size: usize) -> DistributedResult<Self> {
        if world_size == 0 {
            return Err(DistributedError::InvalidTopology(
                "world_size must be greater than zero",
            ));
        }

        let order = (0..world_size).collect();
        Ok(Self { world_size, order })
    }

    pub fn with_global_order(world_size: usize, order: Vec<usize>) -> DistributedResult<Self> {
        if world_size == 0 {
            return Err(DistributedError::InvalidTopology(
                "world_size must be greater than zero",
            ));
        }
        if order.len() != world_size {
            return Err(DistributedError::InvalidCollectiveInput(
                "global order length must equal world_size",
            ));
        }

        let mut seen = vec![false; world_size];
        for rank in &order {
            if *rank >= world_size {
                return Err(DistributedError::RankOutOfRange {
                    rank: *rank,
                    world_size,
                });
            }
            if seen[*rank] {
                return Err(DistributedError::InvalidCollectiveInput(
                    "global order must not repeat ranks",
                ));
            }
            seen[*rank] = true;
        }

        Ok(Self { world_size, order })
    }
}

impl CollectiveOps for DeterministicCollectives {
    fn all_reduce_sum(&self, rank_inputs: &[Vec<f32>]) -> DistributedResult<Vec<Vec<f32>>> {
        if rank_inputs.len() != self.world_size {
            return Err(DistributedError::InvalidCollectiveInput(
                "rank_inputs must contain one tensor per rank",
            ));
        }

        let width = rank_inputs
            .first()
            .map(|row| row.len())
            .ok_or(DistributedError::InvalidCollectiveInput(
                "rank_inputs cannot be empty",
            ))?;

        for row in rank_inputs {
            if row.len() != width {
                return Err(DistributedError::InvalidCollectiveInput(
                    "all input tensors must have the same width",
                ));
            }
        }

        let mut reduced = vec![0.0_f32; width];
        for rank in &self.order {
            for (idx, value) in rank_inputs[*rank].iter().enumerate() {
                reduced[idx] += *value;
            }
        }

        Ok(vec![reduced; self.world_size])
    }

    fn all_to_all(&self, rank_payloads: &[Vec<Vec<u8>>]) -> DistributedResult<Vec<Vec<Vec<u8>>>> {
        if rank_payloads.len() != self.world_size {
            return Err(DistributedError::InvalidCollectiveInput(
                "rank_payloads must contain one row per rank",
            ));
        }

        for row in rank_payloads {
            if row.len() != self.world_size {
                return Err(DistributedError::InvalidCollectiveInput(
                    "each rank row must contain world_size destination payloads",
                ));
            }
        }

        let mut output = vec![vec![Vec::new(); self.world_size]; self.world_size];

        for src in &self.order {
            for dst in &self.order {
                output[*dst][*src] = rank_payloads[*src][*dst].clone();
            }
        }

        Ok(output)
    }

    fn global_order(&self) -> &[usize] {
        &self.order
    }
}

#[cfg(test)]
mod tests {
    use super::{CollectiveOps, DeterministicCollectives};

    #[test]
    fn all_reduce_uses_fixed_global_order() {
        let collectives = DeterministicCollectives::new(3).expect("valid world size");
        let input = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let output = collectives
            .all_reduce_sum(&input)
            .expect("all-reduce should succeed");

        assert_eq!(output.len(), 3);
        assert_eq!(output[0], vec![9.0, 12.0]);
        assert_eq!(output[1], vec![9.0, 12.0]);
        assert_eq!(output[2], vec![9.0, 12.0]);
    }

    #[test]
    fn all_to_all_is_stable() {
        let collectives = DeterministicCollectives::new(2).expect("valid world size");
        let payloads = vec![
            vec![b"0->0".to_vec(), b"0->1".to_vec()],
            vec![b"1->0".to_vec(), b"1->1".to_vec()],
        ];

        let output = collectives
            .all_to_all(&payloads)
            .expect("all-to-all should succeed");

        assert_eq!(output[0][0], b"0->0".to_vec());
        assert_eq!(output[0][1], b"1->0".to_vec());
        assert_eq!(output[1][0], b"0->1".to_vec());
        assert_eq!(output[1][1], b"1->1".to_vec());
    }
}
