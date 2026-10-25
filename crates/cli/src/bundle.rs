//! Replays deploy-and-call bundles and times every call on the interpreter and the JIT.
//!
//! A bundle is a JSON list of cases. Each case deploys `creation` from one EOA and then sends
//! `calls` as sequential transactions. Every call runs on the state left by the previous call:
//! the timed repetitions are discarded and one more execution is committed.

use alloy_consensus::{TxLegacy, transaction::Recovered};
use alloy_primitives::{Address, B256, Bytes, TxKind, U256, address};
use evm2::{
    BaseEvmTypes, Evm, ExecutionConfig, InterpreterRunner, Precompiles, SpecId, TxResult, Version,
    env::BlockEnvExt,
    ethereum::{RecoveredTxEnvelope, TxEnvelope, ethereum_tx_registry},
    evm::{AccountInfo, InMemoryDB},
    interpreter::{InstrStop, Interpreter},
};
use evm2_jit_context::EvmCompilerFn;
use evm2_jit_llvm::EvmLlvmBackend;
use evm2_jit_runtime::{EvmCompiler, eyre};
use serde::Deserialize;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

/// Anvil's first default account.
const CALLER: Address = address!("0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266");
const SPEC: SpecId = SpecId::OSAKA;

#[derive(Debug, clap::Args)]
pub(crate) struct BundleArgs {
    /// Bundle JSON files.
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    /// Timed batches per call and engine; the reported cost is the median batch.
    #[arg(long, default_value = "7")]
    batches: usize,
    /// Minimum user-mode cycles per batch; sets how many repetitions a batch runs.
    #[arg(long, default_value = "200000")]
    batch_cycles: u64,
    /// Only run cases whose `test_id` contains this string.
    #[arg(long)]
    filter: Option<String>,
    /// Write per-call results as JSON lines to this file.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Write JIT dumps (`bytecode.txt`, `opt.s`, ...) of every executed contract to this
    /// directory.
    #[arg(long)]
    dump: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Case {
    test_id: String,
    variant: String,
    creation: Bytes,
    deploy_gas: u64,
    calls: Vec<Call>,
}

#[derive(Debug, Deserialize)]
struct Call {
    label: String,
    calldata: Bytes,
    gas: u64,
}

impl BundleArgs {
    pub(crate) fn run(self) -> eyre::Result<()> {
        let counters = Counters::new()?;
        let mut compiler = EvmCompiler::new_llvm(false)?;
        compiler.set_dump_to(self.dump.clone());
        let mut out = self.out.as_ref().map(std::fs::File::create).transpose()?;
        for path in &self.paths {
            let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(path)?)?;
            let mut total = [0.0; 4];
            let mut mismatches = 0;
            eprintln!("{}", path.display());
            eprintln!(
                "{:<30} {:>5} {:>12} {:>12} {:>12} {:>12} {:>6}",
                "case", "calls", "interp cyc", "interp ins", "jit cyc", "jit ins", "gas!="
            );
            for case in &cases {
                if self.filter.as_ref().is_some_and(|f| !case.test_id.contains(f.as_str())) {
                    continue;
                }
                let r = self.run_case(case, &counters, &mut compiler, out.as_mut())?;
                eprintln!(
                    "{:<30} {:>5} {:>12.0} {:>12.0} {:>12.0} {:>12.0} {:>6}",
                    case.test_id,
                    case.calls.len(),
                    r.sums[0],
                    r.sums[1],
                    r.sums[2],
                    r.sums[3],
                    r.gas_mismatches
                );
                for (t, s) in total.iter_mut().zip(r.sums) {
                    *t += s;
                }
                mismatches += r.gas_mismatches;
            }
            let variant = cases.first().map_or("", |c| c.variant.as_str());
            eprintln!(
                "{:<30} {:>5} {:>12.0} {:>12.0} {:>12.0} {:>12.0} {:>6}\n",
                format!("TOTAL {variant}"),
                "",
                total[0],
                total[1],
                total[2],
                total[3],
                mismatches
            );
        }
        Ok(())
    }

    fn run_case(
        &self,
        case: &Case,
        counters: &Counters,
        compiler: &mut EvmCompiler<EvmLlvmBackend>,
        mut out: Option<&mut std::fs::File>,
    ) -> eyre::Result<CaseResult> {
        let mut db = InMemoryDB::default();
        db.insert_account_info(&CALLER, AccountInfo::default().with_balance(U256::MAX >> 8));
        let new_evm = || {
            let block = BlockEnvExt { number: U256::ONE, ..Default::default() };
            let mut evm = Evm::<BaseEvmTypes>::new(
                SPEC,
                block,
                ethereum_tx_registry(SPEC),
                db.clone(),
                Precompiles::base(SPEC),
            );
            // Like anvil's `--disable-code-size-limit`, which the bundles were recorded with.
            let mut version = *Version::base(SPEC);
            version.max_code_size = usize::MAX;
            version.max_initcode_size = usize::MAX;
            evm.set_block_and_execution_config(
                block,
                ExecutionConfig::for_spec_and_version(SPEC, version),
                SPEC,
                ethereum_tx_registry(SPEC),
                Precompiles::base(SPEC),
            );
            evm
        };
        let mut interp = new_evm();
        let mut jit = new_evm();
        let codes = Arc::new(Mutex::new(HashMap::new()));
        let recorder = Recorder { codes: Arc::clone(&codes) };
        let mut functions = HashMap::<B256, EvmCompilerFn>::new();
        let misses = Arc::new(AtomicU64::new(0));

        let deploy = tx(0, TxKind::Create, case.creation.clone());
        interp.set_interpreter_runner(recorder.clone());
        let deployed = interp.transact(&deploy)?.commit();
        interp.clear_interpreter_runner();
        let deployed_jit = jit.transact(&deploy)?.commit();
        let target = deployed.created_address.ok_or_else(|| {
            eyre::eyre!("{}: deployment failed: {:?}", case.test_id, deployed.stop)
        })?;
        eyre::ensure!(deployed_jit.created_address == Some(target), "JIT deployment differs");
        let mut gas_mismatches = usize::from(deployed.tx_gas_used() != case.deploy_gas);
        if gas_mismatches != 0 {
            eprintln!(
                "  {}: deploy gas {} != anvil {}",
                case.test_id,
                deployed.tx_gas_used(),
                case.deploy_gas
            );
        }

        let mut sums = [0.0; 4];
        for (i, call) in case.calls.iter().enumerate() {
            let tx = tx(i as u64 + 1, TxKind::Call(target), call.calldata.clone());

            // The committed interpreter run records every executed bytecode for the JIT.
            let interp_cost =
                self.time(counters, || interp.transact(&tx).map(|r| drop(r.discard())))?;
            interp.set_interpreter_runner(recorder.clone());
            let ires = interp.transact(&tx)?.commit();
            interp.clear_interpreter_runner();

            for (hash, code) in codes.lock().unwrap().iter() {
                if !functions.contains_key(hash) {
                    let name =
                        format!("{}_{}_{}", case.variant, case.test_id, &hash.to_string()[2..10]);
                    compiler.set_module_name(&name);
                    let f = unsafe { compiler.jit(&name, code, SPEC)? };
                    compiler.clear_ir()?;
                    functions.insert(*hash, f);
                }
            }
            jit.set_interpreter_runner(JitRunner {
                functions: Arc::new(functions.clone()),
                misses: Arc::clone(&misses),
            });
            let jit_cost = self.time(counters, || jit.transact(&tx).map(|r| drop(r.discard())))?;
            misses.store(0, Ordering::Relaxed);
            let jres = jit.transact(&tx)?.commit();
            eyre::ensure!(
                misses.load(Ordering::Relaxed) == 0,
                "{} {}: JIT fell back to the interpreter",
                case.test_id,
                call.label
            );
            check_same(&ires, &jres)
                .map_err(|e| eyre::eyre!("{} {}: {e}", case.test_id, call.label))?;
            if ires.tx_gas_used() != call.gas {
                gas_mismatches += 1;
                eprintln!(
                    "  {} {}: gas {} != anvil {} (status {})",
                    case.test_id,
                    call.label,
                    ires.tx_gas_used(),
                    call.gas,
                    ires.status
                );
            }
            let costs = [interp_cost.0, interp_cost.1, jit_cost.0, jit_cost.1];
            for (s, c) in sums.iter_mut().zip(costs) {
                *s += c;
            }
            if let Some(out) = out.as_mut() {
                use std::io::Write;
                let line = serde_json::json!({
                    "variant": case.variant,
                    "test_id": case.test_id,
                    "label": call.label,
                    "gas": ires.tx_gas_used(),
                    "anvil_gas": call.gas,
                    "status": ires.status,
                    "interp_cycles": costs[0],
                    "interp_instructions": costs[1],
                    "jit_cycles": costs[2],
                    "jit_instructions": costs[3],
                });
                writeln!(out, "{line}")?;
            }
        }
        Ok(CaseResult { sums, gas_mismatches })
    }

    /// Returns the median (cycles, instructions) of one execution over `batches` batches.
    fn time<E: std::fmt::Debug>(
        &self,
        counters: &Counters,
        mut f: impl FnMut() -> Result<(), E>,
    ) -> eyre::Result<(f64, f64)> {
        f().map_err(|e| eyre::eyre!("{e:?}"))?;
        let (c, _) = counters.measure(|| f().unwrap());
        let reps = (self.batch_cycles / c.max(1)).clamp(1, 10_000);
        let mut samples = Vec::with_capacity(self.batches);
        for _ in 0..self.batches {
            let (c, i) = counters.measure(|| {
                for _ in 0..reps {
                    f().unwrap();
                }
            });
            samples.push((c as f64 / reps as f64, i as f64 / reps as f64));
        }
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(samples[samples.len() / 2])
    }
}

struct CaseResult {
    sums: [f64; 4],
    gas_mismatches: usize,
}

const fn tx(nonce: u64, to: TxKind, input: Bytes) -> RecoveredTxEnvelope {
    let tx = TxLegacy {
        chain_id: None,
        nonce,
        gas_price: 0,
        gas_limit: Version::base(SPEC).tx_gas_limit_cap,
        to,
        value: U256::ZERO,
        input,
    };
    Recovered::new_unchecked(TxEnvelope::Legacy(tx), CALLER)
}

fn check_same(a: &TxResult, b: &TxResult) -> Result<(), String> {
    if a.status != b.status || a.output != b.output || a.tx_gas_used() != b.tx_gas_used() {
        return Err(format!(
            "interpreter and JIT differ: status {}/{}, gas {}/{}, output {}/{}",
            a.status,
            b.status,
            a.tx_gas_used(),
            b.tx_gas_used(),
            a.output,
            b.output
        ));
    }
    Ok(())
}

/// Records the bytecode of every frame and lets the interpreter run it.
#[derive(Clone, Debug)]
struct Recorder {
    codes: Arc<Mutex<HashMap<B256, Bytes>>>,
}

impl InterpreterRunner<BaseEvmTypes> for Recorder {
    fn run<'frame, 'host>(
        &self,
        interpreter: &mut Interpreter<'frame, 'host, BaseEvmTypes>,
        _host: &mut Evm<'host, BaseEvmTypes>,
    ) -> Option<InstrStop> {
        self.codes
            .lock()
            .unwrap()
            .entry(interpreter.original_bytecode_hash())
            .or_insert_with(|| interpreter.original_bytecode());
        None
    }
}

/// Runs frames with precompiled functions and counts frames it has no function for.
#[derive(Clone, Debug)]
struct JitRunner {
    functions: Arc<HashMap<B256, EvmCompilerFn>>,
    misses: Arc<AtomicU64>,
}

impl InterpreterRunner<BaseEvmTypes> for JitRunner {
    fn run<'frame, 'host>(
        &self,
        interpreter: &mut Interpreter<'frame, 'host, BaseEvmTypes>,
        host: &mut Evm<'host, BaseEvmTypes>,
    ) -> Option<InstrStop> {
        let Some(func) = self.functions.get(&interpreter.original_bytecode_hash()) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        interpreter.prepare_run(host);
        Some(unsafe { func.call_with_interpreter(interpreter) })
    }
}

/// User-mode cycle and instruction counters for the calling thread.
struct Counters {
    cycles: i32,
    instructions: i32,
}

impl Counters {
    fn new() -> eyre::Result<Self> {
        Ok(Self { cycles: perf_counter(0)?, instructions: perf_counter(1)? })
    }

    fn read(fd: i32) -> u64 {
        let mut v = 0u64;
        let n = unsafe { libc::read(fd, (&raw mut v).cast(), 8) };
        assert_eq!(n, 8, "failed to read perf counter");
        v
    }

    fn measure(&self, f: impl FnOnce()) -> (u64, u64) {
        let (c0, i0) = (Self::read(self.cycles), Self::read(self.instructions));
        f();
        let (c1, i1) = (Self::read(self.cycles), Self::read(self.instructions));
        (c1 - c0, i1 - i0)
    }
}

/// Opens a user-mode hardware counter (`PERF_TYPE_HARDWARE`) for the calling thread.
fn perf_counter(config: u64) -> eyre::Result<i32> {
    // `perf_event_attr`, version 8 (136 bytes): type and size, config, then flags at word 6.
    let mut attr = [0u64; 17];
    attr[0] = (136u64) << 32;
    attr[1] = config;
    // exclude_kernel | exclude_hv
    attr[6] = (1 << 5) | (1 << 6);
    let fd = unsafe { libc::syscall(libc::SYS_perf_event_open, attr.as_ptr(), 0, -1, -1, 0) };
    eyre::ensure!(fd >= 0, "perf_event_open failed: {}", std::io::Error::last_os_error());
    Ok(fd as i32)
}
