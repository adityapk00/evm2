//! Geth copy requirements when execution stops before all dynamic charges are reached.

use evm2::{
    EvmFeatures, EvmTypes,
    interpreter::{InstrStop, Interpreter, opcode::op},
    utils::WordExt,
};

/// Copy costs captured independently of optional stack and memory snapshots.
#[derive(Clone, Copy, Debug)]
pub(super) struct PendingCopyGas {
    stack_len: usize,
    gas_before: u64,
    copy: u64,
    memory: Option<u64>,
    cold: u64,
}

impl PendingCopyGas {
    pub(super) fn capture<T: EvmTypes>(interp: &mut Interpreter<'_, '_, T>) -> Option<Self> {
        let opcode = interp.opcode();
        let external = opcode == op::EXTCODECOPY;
        if !matches!(
            opcode,
            op::CALLDATACOPY | op::CODECOPY | op::RETURNDATACOPY | op::MCOPY | op::EXTCODECOPY
        ) {
            return None;
        }
        let stack = interp.stack();
        let stack = stack.as_slice();
        let inputs = if external { 4 } else { 3 };
        if stack.len() < inputs {
            return None;
        }
        let stack_len = stack.len();
        let args = &stack[stack_len - inputs..];
        let len = usize::try_from(args[0]).ok();
        let params = &interp.version().gas_params;
        let copy = len.map_or(0, |len| match opcode {
            op::EXTCODECOPY => params.extcodecopy_cost(len),
            op::MCOPY => params.mcopy_cost(len),
            _ => params.copy_cost(len),
        });
        let memory = len.and_then(|len| {
            if len == 0 {
                return Some(0);
            }
            let offset = if opcode == op::MCOPY { args[2].max(args[1]) } else { args[2] };
            let end = usize::try_from(offset).ok()?.checked_add(len)?;
            // Geth rejects larger ranges before returning any dynamic copy cost.
            let words = end.div_ceil(32);
            if words > 0xffff_ffff {
                return None;
            }
            Some(params.memory_cost(words).saturating_sub(interp.gas().memory().expansion_cost))
        });
        let address = external.then(|| args[3].to_address());
        let cold_cost = params.cold_account_additional_cost();
        let berlin = interp.version().feature(EvmFeatures::EIP2929);
        let cold = if berlin
            && let Some(address) = address
            && !interp.host().state().is_account_warm(&address)
        {
            cold_cost
        } else {
            0
        };
        Some(Self { stack_len, gas_before: interp.gas().remaining(), copy, memory, cold })
    }

    pub(super) fn failed_cost<T: EvmTypes>(self, interp: &Interpreter<'_, '_, T>) -> Option<u64> {
        // A static charge or stack/activation check failed before the operands were popped.
        if interp.stack().len() == self.stack_len {
            return None;
        }
        let paid = self.gas_before.saturating_sub(interp.gas().remaining());
        let cost = match interp.result() {
            Err(InstrStop::OutOfGas) => {
                // Only a true failed deduction reaches here; cold-load guards already have
                // their complete cost recorded separately. The copy charge precedes memory.
                let attempted = interp.failed_charge()?;
                self.memory.map_or(attempted.saturating_sub(self.copy), |memory| {
                    attempted.saturating_add(memory).saturating_add(self.cold)
                })
            }
            Err(InstrStop::MemoryOOG) => {
                let attempted = interp.failed_charge()?;
                self.memory.map_or(attempted.saturating_sub(self.copy), |_| {
                    attempted.saturating_add(self.cold)
                })
            }
            Err(InstrStop::OutOfOffset) => {
                // RETURNDATACOPY checks bounds before copy or memory charges.
                self.memory
                    .map_or(paid, |memory| paid.saturating_add(self.copy).saturating_add(memory))
            }
            Err(InstrStop::InvalidOperandOOG | InstrStop::MemoryLimitOOG) => {
                let paid = interp.failed_charge().unwrap_or(paid);
                self.memory.map_or(paid.saturating_sub(self.copy), |memory| {
                    paid.saturating_add(memory).saturating_add(self.cold)
                })
            }
            _ => return None,
        };
        Some(cost)
    }
}
