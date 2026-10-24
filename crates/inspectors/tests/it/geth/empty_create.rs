//! Empty initcode must retain creation frames without inventing an opcode.

use super::*;

#[test]
fn empty_create_struct_logs_and_call_frames() {
    let caller = address!("00000000000000000000000000000000000a11ce");
    let parent = address!("00000000000000000000000000000000000b0b00");
    for (code, root, gas, expected) in [
        ("", true, 53_000, vec![]),
        ("00", true, 53_006, vec![(0, "STOP", 1)]),
        (
            "5f5f5ff000",
            false,
            53_006,
            vec![
                (0, "PUSH0", 1),
                (1, "PUSH0", 1),
                (2, "PUSH0", 1),
                (3, "CREATE", 1),
                (4, "STOP", 1),
            ],
        ),
        (
            "5f5f5f5ff500",
            false,
            53_008,
            vec![
                (0, "PUSH0", 1),
                (1, "PUSH0", 1),
                (2, "PUSH0", 1),
                (3, "PUSH0", 1),
                (4, "CREATE2", 1),
                (5, "STOP", 1),
            ],
        ),
        (
            "5f5f6000f05f5260205ff3",
            false,
            53_020,
            vec![
                (0, "PUSH0", 1),
                (1, "PUSH0", 1),
                (2, "PUSH1", 1),
                (4, "CREATE", 1),
                (5, "PUSH0", 1),
                (6, "MSTORE", 1),
                (7, "PUSH1", 1),
                (9, "PUSH0", 1),
                (10, "RETURN", 1),
            ],
        ),
        // A single zero byte in memory is real STOP initcode.
        (
            "60015f5ff000",
            false,
            53_012,
            vec![
                (0, "PUSH1", 1),
                (2, "PUSH0", 1),
                (3, "PUSH0", 1),
                (4, "CREATE", 1),
                (0, "STOP", 2),
                (5, "STOP", 1),
            ],
        ),
    ] {
        let code = Bytes::from(hex::decode(code).unwrap());
        let mut inspector = TracingInspector::new(TracingInspectorConfig::default_geth());
        let result = Context::mainnet()
            .with_db(CacheDB::<EmptyDB>::default())
            .modify_cfg_chained(|cfg| cfg.spec = SpecId::OSAKA)
            .modify_db_chained(|db| {
                db.insert_account_info(
                    &caller,
                    AccountInfo { balance: U256::from(1_000_000), ..Default::default() },
                );
                db.insert_account_info(
                    &parent,
                    AccountInfo {
                        code: Some(Bytecode::new_raw(code.clone())),
                        ..Default::default()
                    },
                );
            })
            .build_mainnet()
            .with_inspector(&mut inspector)
            .inspect_tx(TxEnv {
                caller,
                gas_limit: 200_000,
                kind: if root { TxKind::Create } else { TxKind::Call(parent) },
                data: if root { code.clone() } else { Bytes::new() },
                ..Default::default()
            })
            .unwrap()
            .result;
        assert!(result.is_success());
        assert_eq!(result.tx_gas_used(), gas);
        let builder = inspector.geth_builder();
        let trace = builder.geth_traces(
            gas,
            result.output().cloned().unwrap_or_default(),
            GethDefaultTracingOptions::default(),
        );
        assert!(!trace.failed);
        assert_eq!(
            trace
                .struct_logs
                .iter()
                .map(|step| (step.pc, step.opcode(), step.depth))
                .collect::<Vec<_>>(),
            expected
        );
        let calls = builder.geth_call_traces(CallConfig::default(), gas);
        let create = if root {
            &calls
        } else {
            assert_eq!(calls.calls.len(), 1);
            &calls.calls[0]
        };
        let created = if root {
            caller.create(0)
        } else if code[4] == op::CREATE2 {
            parent.create2_from_code([0; 32], [])
        } else {
            parent.create(0)
        };
        assert_eq!(create.to, Some(created));
        assert_eq!(create.from, if root { caller } else { parent });
        assert_eq!(create.typ, if !root && code[4] == op::CREATE2 { "CREATE2" } else { "CREATE" });
        assert_eq!(create.error, None);
        assert_eq!(create.gas_used, U256::from(if root { gas } else { 0 }));
        assert!(create.output.as_ref().is_none_or(|output| output.is_empty()));
        assert!(create.calls.is_empty());
    }
}
