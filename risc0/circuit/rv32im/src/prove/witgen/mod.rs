// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub(crate) mod bigint;
pub(crate) mod byte_poly;
pub(crate) mod paged_map;
pub(crate) mod poseidon2;
pub(crate) mod preflight;
pub(crate) mod sha2;
#[cfg(test)]
mod tests;

use std::iter::zip;

use anyhow::{Context, Result};
use preflight::PreflightTrace;
use risc0_binfmt::WordAddr;

use risc0_core::scope;
use risc0_zkp::{
    core::digest::DIGEST_WORDS,
    field::{Elem as _, ExtElem as _},
    hal::Hal,
};

use self::{
    bigint::BigIntState,
    byte_poly::{BigIntAccum, BigIntAccumState},
    preflight::Back,
};
use super::hal::{CircuitAccumulator, CircuitWitnessGenerator, MetaBuffer, StepMode};
use crate::{
    execute::{
        platform::MERKLE_TREE_END_ADDR, poseidon2::Poseidon2State, segment::Segment,
        sha2::Sha2State,
    },
    zirgen::circuit::{
        CircuitField, ExtVal, Val, LAYOUT_GLOBAL, LAYOUT_TOP, REGCOUNT_ACCUM, REGCOUNT_CODE,
        REGCOUNT_DATA, REGCOUNT_GLOBAL, REGCOUNT_MIX,
    },
};

// Lookup table for bit expansion - precomputed for all 256 possible byte values
static BIT_LOOKUP: [[u32; 8]; 256] = {
    let mut table = [[0u32; 8]; 256];
    let mut i = 0;
    while i < 256 {
        let mut j = 0;
        while j < 8 {
            table[i][j] = if i & (1 << j) != 0 { 1 } else { 0 };
            j += 1;
        }
        i += 1;
    }
    table
};

// Fast bit unpacking using lookup table
fn unpack_u32_bits_fast(value: u32) -> [u32; 32] {
    let mut result = [0u32; 32];
    result[0..8].copy_from_slice(&BIT_LOOKUP[(value & 0xFF) as usize]);
    result[8..16].copy_from_slice(&BIT_LOOKUP[((value >> 8) & 0xFF) as usize]);
    result[16..24].copy_from_slice(&BIT_LOOKUP[((value >> 16) & 0xFF) as usize]);
    result[24..32].copy_from_slice(&BIT_LOOKUP[((value >> 24) & 0xFF) as usize]);
    result
}

pub(crate) struct WitnessGenerator<H: Hal> {
    cycles: usize,
    pub global: MetaBuffer<H>,
    pub code: MetaBuffer<H>,
    pub data: MetaBuffer<H>,
    pub accum: MetaBuffer<H>,
    pub trace: PreflightTrace,
}

impl<H> WitnessGenerator<H>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
{
    pub fn new<C: CircuitWitnessGenerator<H>>(
        hal: &H,
        circuit_hal: &C,
        segment: &Segment,
        mode: StepMode,
        rand_z: ExtVal,
    ) -> Result<Self> {
        scope!("witgen");

        let trace = segment.preflight(rand_z)?;
        let cycles = trace.cycles.len();

        tracing::trace!("{segment:#?}");
        tracing::trace!("{trace:#?}");

        // assert_eq!(
        //     segment.suspend_cycle + segment.paging_cycles + LOOKUP_TABLE_CYCLES as u32 + 1,
        //     cycles as u32,
        //     "suspend_cycle: {} + paging_cycles: {} + {LOOKUP_TABLE_CYCLES} + 1 == trace.cycles",
        //     segment.suspend_cycle,
        //     segment.paging_cycles
        // );
        // assert_eq!(cycles, 1 << segment.po2, "cycles == 1 << segment.po2");
        assert!(cycles <= 1 << segment.po2, "cycles <= 1 << segment.po2");
        let cycles = 1 << segment.po2;

        let mut global = vec![Val::INVALID; REGCOUNT_GLOBAL];

        // state in
        for (i, word) in segment.claim.pre_state.as_words().iter().enumerate() {
            let low = word & 0xffff;
            let high = word >> 16;
            global[LAYOUT_GLOBAL.state_in.values[i].low._super.offset] = low.into();
            global[LAYOUT_GLOBAL.state_in.values[i].high._super.offset] = high.into();
        }

        // input digest
        for (i, word) in segment.claim.input.as_words().iter().enumerate() {
            let low = word & 0xffff;
            let high = word >> 16;
            global[LAYOUT_GLOBAL.input.values[i].low._super.offset] = low.into();
            global[LAYOUT_GLOBAL.input.values[i].high._super.offset] = high.into();
        }

        // rand_z
        for (i, &elem) in trace.rand_z.elems().iter().enumerate() {
            global[LAYOUT_GLOBAL.rng._super.offset + i] = elem;
        }

        // is_terminate
        let is_terminate = if segment.claim.terminate_state.is_some() {
            1u32
        } else {
            0u32
        };
        global[LAYOUT_GLOBAL.is_terminate._super.offset] = is_terminate.into();

        // shutdown_cycle
        global[LAYOUT_GLOBAL.shutdown_cycle._super.offset] = segment.segment_threshold.into();

        let global = MetaBuffer {
            buf: hal.copy_from_elem("global", &global),
            rows: 1,
            cols: REGCOUNT_GLOBAL,
            checked: true,
        };

        let code = MetaBuffer::new("code", hal, cycles, REGCOUNT_CODE, false);

        let data = scope!(
            "alloc(data)",
            MetaBuffer::new("data", hal, cycles, REGCOUNT_DATA, true)
        );

                // Set stateful columns from 'top' - optimized sequential processing
        let mut injector = FastInjector::new(cycles);
        for (row, back) in trace.backs.iter().enumerate() {
            let cycle = &trace.cycles[row];
            match back {
                Back::None => {}
                Back::Ecall(s0, s1, s2) => {
                    const ECALL_S0: usize = LAYOUT_TOP.inst_result.arm8.s0._super.offset;
                    const ECALL_S1: usize = LAYOUT_TOP.inst_result.arm8.s1._super.offset;
                    const ECALL_S2: usize = LAYOUT_TOP.inst_result.arm8.s2._super.offset;

                    injector.push_single(row, ECALL_S0, *s0);
                    injector.push_single(row, ECALL_S1, *s1);
                    injector.push_single(row, ECALL_S2, *s2);
                }
                Back::Poseidon2(p2_state) => {
                    let p2_offsets = Poseidon2State::offsets();
                    let p2_values = p2_state.as_array();

                    injector.push_batch(row, &p2_offsets, &p2_values);
                }
                Back::Sha2(sha2_state) => {
                    let fp_offsets = Sha2State::fp_offsets();
                    let fp_values = sha2_state.fp_array();
                    let u32_offsets = Sha2State::u32_offsets();
                    let u32_values = sha2_state.u32_array();

                    injector.push_batch(row, &fp_offsets, &fp_values);

                    for (&col, &value) in zip(&u32_offsets, &u32_values) {
                        injector.push_u32_bits_fast(row, col, value);
                    }
                }
                Back::BigInt(state) => {
                    let bigint_offsets = BigIntState::offsets();
                    let bigint_values = state.as_array();

                    injector.push_batch(row, &bigint_offsets, &bigint_values);
                }
            }

            // Add cycle data
            const CYCLE_COL: usize = LAYOUT_TOP.cycle._super.offset;
            const NEXT_PC_LOW: usize = LAYOUT_TOP.next_pc_low._super.offset;
            const NEXT_PC_HIGH: usize = LAYOUT_TOP.next_pc_high._super.offset;
            const NEXT_STATE: usize = LAYOUT_TOP.next_state_0._super.offset;
            const NEXT_MACHINE_MODE: usize = LAYOUT_TOP.next_machine_mode._super.offset;

            injector.push_single(row, CYCLE_COL, row as u32);
            injector.push_single(row, NEXT_PC_LOW, cycle.pc & 0xffff);
            injector.push_single(row, NEXT_PC_HIGH, cycle.pc >> 16);
            injector.push_single(row, NEXT_STATE, cycle.state);
            injector.push_single(row, NEXT_MACHINE_MODE, cycle.machine_mode as u32);

            injector.push();
        }

        hal.scatter(
            &data.buf,
            &injector.index,
            &injector.offsets,
            &injector.values,
        );

        circuit_hal
            .generate_witness(mode, &trace, &global, &data)
            .context("witness generation failure")?;

        // Zero out 'invalid' entries in data and output.
        scope!("zeroize", {
            hal.eltwise_zeroize_elem(&global.buf);
            hal.eltwise_zeroize_elem(&code.buf);
            hal.eltwise_zeroize_elem(&data.buf);
        });

        // #[cfg(feature = "entropy_finder")]
        // if let Ok(dump_path) = std::env::var("DATA_DUMP") {
        //     let raw = data.buf.to_vec();

        //     let old = if std::fs::exists(&dump_path).unwrap() {
        //         Some(std::fs::read(&dump_path).unwrap())
        //     } else {
        //         None
        //     };

        //     std::fs::write(dump_path, bytemuck::cast_slice(&raw)).unwrap();
        //     if let Some(old) = old {
        //         let old = bytemuck::cast_slice(&old);
        //         for cycle in 0..cycles {
        //             for col in 0..REGCOUNT_DATA {
        //                 assert_eq!(
        //                     H::Elem::new_raw(old[col * cycles + cycle]),
        //                     raw[col * cycles + cycle],
        //                     "cycle: {cycle}, col: {col}",
        //                 );
        //             }
        //         }
        //     }
        // }

        let accum = scope!(
            "alloc(accum)",
            MetaBuffer::new("accum", hal, cycles, REGCOUNT_ACCUM, true)
        );

        Ok(Self {
            cycles,
            global,
            code,
            data,
            accum,
            trace,
        })
    }

    pub fn accum<C: CircuitAccumulator<H>>(
        &self,
        hal: &H,
        circuit_hal: &C,
        mix: &[Val],
    ) -> Result<MetaBuffer<H>> {
        // use final mix to compute BigIntAccumPowers
        let last_mix = ExtVal::from_subelems(mix[mix.len() - 4..].iter().cloned());

        // inject BigIntAccumState backs
        let mut injector = FastInjector::new(self.cycles);
        let mut bigint_accum = BigIntAccum::new(last_mix);

        for (row, back) in self.trace.backs.iter().enumerate() {
            if let Back::BigInt(state) = back {
                bigint_accum.step(state)?;
                for (col, value) in zip(BigIntAccumState::offsets(), bigint_accum.state.as_array())
                {
                    injector.push_single(row, col, value);
                }
                injector.push();
            }
        }

        hal.scatter(
            &self.accum.buf,
            &injector.index,
            &injector.offsets,
            &injector.values,
        );

        let mix = MetaBuffer {
            buf: hal.copy_from_elem("mix", mix),
            rows: 1,
            cols: REGCOUNT_MIX,
            checked: true,
        };

        circuit_hal.step_accum(&self.trace, &self.data, &self.accum, &self.global, &mix)?;

        scope!("zeroize(accum)", {
            hal.eltwise_zeroize_elem(&self.accum.buf);
        });

        Ok(mix)
    }
}

#[derive(Debug)]
struct FastInjector {
    rows: usize,
    offsets: Vec<u32>,
    values: Vec<Val>,
    index: Vec<u32>,
}

impl FastInjector {
    fn new(rows: usize) -> Self {
        let mut index = Vec::with_capacity(rows + 1);
        index.push(0);

        // Pre-allocate much larger capacity based on worst-case analysis:
        // - Poseidon2: 25 elements
        // - SHA2: ~50 FP + up to 1000+ bit operations (32 bits * N u32 values)
        // - BigInt: ~16 elements
        // - Cycle data: 5 elements
        // Estimate: 1500 operations per cycle worst case
        let estimated_ops = rows * 1500;

        Self {
            rows,
            offsets: Vec::with_capacity(estimated_ops),
            values: Vec::with_capacity(estimated_ops),
            index,
        }
    }

    fn push(&mut self) {
        self.index.push(self.offsets.len() as u32);
    }

    fn push_single(&mut self, row: usize, col: usize, value: u32) {
        let idx = col * self.rows + row;
        self.offsets.push(idx as u32);
        self.values.push(value.into());
    }

    fn push_u32_bits_fast(&mut self, row: usize, col: usize, value: u32) {
        // Use lookup table for fast bit unpacking
        let bits = unpack_u32_bits_fast(value);

        // Pre-compute base indices and batch insert
        let base_offset = col * self.rows + row;
        let row_stride = self.rows;

        for (i, &bit) in bits.iter().enumerate() {
            let idx = base_offset + i * row_stride;
            self.offsets.push(idx as u32);
            self.values.push(bit.into());
        }
    }

    fn push_batch(&mut self, row: usize, cols: &[usize], values: &[u32]) {
        // Batch insert multiple values efficiently
        let base_row = row;
        for (&col, &value) in zip(cols, values) {
            let idx = col * self.rows + base_row;
            self.offsets.push(idx as u32);
            self.values.push(value.into());
        }
    }
}

fn node_addr_to_idx(addr: WordAddr) -> u32 {
    (MERKLE_TREE_END_ADDR - addr).0 / DIGEST_WORDS as u32
}

fn node_idx_to_addr(idx: u32) -> WordAddr {
    MERKLE_TREE_END_ADDR - idx * DIGEST_WORDS as u32
}
