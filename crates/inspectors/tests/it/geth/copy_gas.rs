//! Copy cost regressions pinned to Geth 1df30b18 on Osaka.

use super::*;
use evm2::interpreter::{InstrStop, opcode::OpCode};
use evm2_inspectors::opcode::OpcodeGasInspector;

fn context(code: &[u8]) -> Context {
    Context::mainnet()
        .with_db(CacheDB::<EmptyDB>::default())
        .modify_cfg_chained(|cfg| cfg.spec = SpecId::OSAKA)
        .modify_db_chained(|db| {
            db.insert_account_info(
                &address!("00000000000000000000000000000000000b0b00"),
                AccountInfo {
                    code: Some(Bytecode::new_legacy(code.to_vec().into())),
                    ..Default::default()
                },
            );
        })
}

fn transaction(gas_limit: u64) -> TxEnv {
    TxEnv {
        gas_limit,
        kind: TransactTo::Call(address!("00000000000000000000000000000000000b0b00")),
        ..Default::default()
    }
}

#[test]
fn copy_cost_preserves_charged_gas_and_parity() {
    for (code, gas, expected_geth, expected_parity, charged, status) in [
        (&hex!("60015f5f3700")[..], 21_012, 9, 6, 5, InstrStop::OutOfGas),
        (&hex!("60015f5f3900")[..], 21_012, 9, 6, 5, InstrStop::OutOfGas),
        (&hex!("60015f5f5e00")[..], 21_012, 9, 6, 5, InstrStop::OutOfGas),
        (&hex!("60015f5f3e00")[..], 100_000, 9, 3, 3, InstrStop::OutOfOffset),
        (&hex!("60015f5f3700")[..], 21_013, 9, 9, 6, InstrStop::MemoryOOG),
        (&hex!("60015f5f60443c00")[..], 21_112, 2606, 103, 102, InstrStop::OutOfGas),
        (&hex!("60015f5f60443c00")[..], 21_113, 2606, 106, 103, InstrStop::MemoryOOG),
        (&hex!("60015f5f60443c00")[..], 23_609, 2606, 2599, 2599, InstrStop::OutOfGas),
    ] {
        let mut inspector = TracingInspector::new(TracingInspectorConfig::all());
        let result = context(code)
            .build_mainnet()
            .with_inspector(&mut inspector)
            .inspect_tx(transaction(gas))
            .unwrap();
        assert_eq!(result.result.tx_gas_used(), gas);
        assert_eq!(result.tx_result.result.stop, status);
        let step = inspector.traces().nodes()[0].trace.steps.last().unwrap();
        assert_eq!(step.status, Some(status));
        assert_eq!(step.gas_cost, expected_parity);
        let opcode = step.op;
        let frame = inspector.geth_builder().geth_traces(gas, Bytes::new(), Default::default());
        assert_eq!(frame.struct_logs.last().unwrap().gas_cost, expected_geth);
        assert!(frame.struct_logs.last().unwrap().error.is_some());
        assert_eq!(
            inspector.into_parity_builder().vm_trace().ops.last().unwrap().cost,
            expected_parity
        );

        let mut inspector = OpcodeGasInspector::new();
        let result = context(code)
            .build_mainnet()
            .with_inspector(&mut inspector)
            .inspect_tx(transaction(gas))
            .unwrap();
        assert_eq!(result.result.tx_gas_used(), gas);
        assert_eq!(inspector.opcode_gas().get(&opcode), Some(&charged));
    }
}

#[test]
fn copy_cost_matches_osaka_reference() {
    // Costs and final gas/status are captured from fixed-Osaka Geth RPC with equivalent
    // overridden state. Covers zero/nonzero lengths, existing memory, all charge stages,
    // return-data bounds, cold/warm EXTCODECOPY and overflowing memory ranges.
    let cases = include!("copy_gas_cases.rs");
    for (code, gas, pc, cost, used, failed) in cases {
        let code = alloy_primitives::hex::decode(code).unwrap();
        for disable_stack in [false, true] {
            let options = GethDefaultTracingOptions {
                disable_stack: Some(disable_stack),
                ..Default::default()
            };
            let mut inspector =
                TracingInspector::new(TracingInspectorConfig::from_geth_config(&options));
            let result = context(&code)
                .build_mainnet()
                .with_inspector(&mut inspector)
                .inspect_tx(transaction(gas))
                .unwrap();
            assert_eq!(result.result.tx_gas_used(), used, "{code:x?} gas={gas}");
            assert_eq!(!result.result.is_success(), failed, "{code:x?} gas={gas}");
            assert!(result.result.output().unwrap_or_default().is_empty());
            let frame = inspector.geth_builder().geth_traces(used, Bytes::new(), options);
            let step = frame.struct_logs.iter().find(|step| step.pc == pc).unwrap();
            assert_eq!(step.gas_cost, cost, "{code:x?} gas={gas} pc={pc}");
        }
    }
}

#[test]
fn copy_underflow_and_inactive_opcode_keep_original_cost() {
    for opcode in [op::CALLDATACOPY, op::CODECOPY, op::MCOPY, op::RETURNDATACOPY, op::EXTCODECOPY] {
        let mut inspector = TracingInspector::new(TracingInspectorConfig::all());
        context(&[opcode])
            .build_mainnet()
            .with_inspector(&mut inspector)
            .inspect_tx(transaction(100_000))
            .unwrap();
        let step = &inspector.traces().nodes()[0].trace.steps[0];
        assert_eq!(step.status, Some(InstrStop::StackUnderflow));
        assert_eq!(step.geth_gas_cost, None);
        assert_eq!(step.op, OpCode::new(opcode).unwrap());
    }
    let mut inspector = TracingInspector::new(TracingInspectorConfig::all());
    context(&hex!("6001600060005e00"))
        .modify_cfg_chained(|cfg| cfg.spec = SpecId::SHANGHAI)
        .build_mainnet()
        .with_inspector(&mut inspector)
        .inspect_tx(transaction(100_000))
        .unwrap();
    let step = inspector.traces().nodes()[0].trace.steps.last().unwrap();
    assert_eq!(step.status, Some(InstrStop::NotActivated));
    assert_eq!(step.geth_gas_cost, None);
}
