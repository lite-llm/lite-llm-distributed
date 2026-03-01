use lite_llm_distributed::{
    CollectiveOps, ConsensusConfig, DeterministicCollectives, ExpertAddress, ParallelismConfig,
    RankCoordinate, RoutingConsensus,
};

#[test]
fn same_seed_and_input_yield_identical_route_and_collectives() {
    let topology = ParallelismConfig {
        data_parallel: 1,
        tensor_parallel: 1,
        pipeline_parallel: 1,
        expert_parallel: 4,
    };
    topology.validate().expect("topology must be valid");

    let prefix = RankCoordinate {
        dp: 0,
        tp: 0,
        pp: 0,
        ep: 0,
    };

    let mut candidates = Vec::new();
    for expert in 0..8_u32 {
        let address = ExpertAddress {
            tier: 1,
            group: expert / 4,
            expert,
        };
        let _owner = topology
            .expert_owner_rank(prefix, address, 4242)
            .expect("owner rank should compute");
        candidates.push(address);
    }

    let consensus = RoutingConsensus::new(ConsensusConfig::new(4242));
    let hidden = vec![0.21, -0.35, 0.87, 0.11, -0.44, 0.19];

    let mut checksums = Vec::new();
    let mut baseline = None;

    for rank in 0..topology.world_size() {
        let route = consensus
            .select_experts(&hidden, 3, 7, &candidates, 4)
            .expect("route selection should succeed");

        if let Some(reference) = &baseline {
            assert_eq!(route, *reference);
        } else {
            baseline = Some(route.clone());
        }

        checksums.push((rank, route.checksum));
    }

    consensus
        .verify_checksums(&checksums)
        .expect("checksums should agree on all ranks");

    let collectives = DeterministicCollectives::new(topology.world_size())
        .expect("collective engine should initialize");
    let rank_inputs: Vec<Vec<f32>> = (0..topology.world_size())
        .map(|rank| vec![rank as f32 + 1.0, 2.0 * rank as f32])
        .collect();

    let reduced_first = collectives
        .all_reduce_sum(&rank_inputs)
        .expect("all-reduce should succeed");
    let reduced_second = collectives
        .all_reduce_sum(&rank_inputs)
        .expect("all-reduce should succeed");

    assert_eq!(reduced_first, reduced_second);
    for rank in 1..topology.world_size() {
        assert_eq!(reduced_first[0], reduced_first[rank]);
    }

    let payloads: Vec<Vec<Vec<u8>>> = (0..topology.world_size())
        .map(|src| {
            (0..topology.world_size())
                .map(|dst| format!("{src}->{dst}").into_bytes())
                .collect()
        })
        .collect();

    let all_to_all_first = collectives
        .all_to_all(&payloads)
        .expect("all-to-all should succeed");
    let all_to_all_second = collectives
        .all_to_all(&payloads)
        .expect("all-to-all should succeed");

    assert_eq!(all_to_all_first, all_to_all_second);

    for src in 0..topology.world_size() {
        for dst in 0..topology.world_size() {
            assert_eq!(
                all_to_all_first[dst][src],
                format!("{src}->{dst}").into_bytes()
            );
        }
    }
}
