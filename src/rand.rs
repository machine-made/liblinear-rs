// SPDX-License-Identifier: BSD-3-Clause
//! The glibc `TYPE_3` random stream used by the frozen reference.

use crate::Error;

/// Additive feedback with degree 31 and separation 3.
pub struct GlibcRand {
    state: [u32; 31],
    front: usize,
    rear: usize,
}

impl GlibcRand {
    /// Start a stream as glibc `srand` does.
    ///
    /// Seed zero selects seed one.
    ///
    /// # Errors
    ///
    /// Returns an error when a seed step falls outside the 31-bit range.
    pub fn new(seed: u32) -> Result<Self, Error> {
        let seed = if seed == 0 { 1 } else { seed };
        let mut state = [0; 31];
        state[0] = seed;
        // glibc reads the seed as a signed 32-bit word.
        let mut word = i64::from(i32::from_ne_bytes(seed.to_ne_bytes()));
        for slot in &mut state[1..] {
            let high = word / 127_773;
            let low = word % 127_773;
            let scaled_low = low.checked_mul(16_807).ok_or_else(Error::overflow)?;
            let scaled_high = high.checked_mul(2_836).ok_or_else(Error::overflow)?;
            word = scaled_low
                .checked_sub(scaled_high)
                .ok_or_else(Error::overflow)?;
            if word < 0 {
                word = word
                    .checked_add(2_147_483_647)
                    .ok_or_else(Error::overflow)?;
            }
            *slot = u32::try_from(word).map_err(|_| {
                Error::input(
                    0,
                    "The random seed step is outside the 31-bit range. Report this failure.",
                )
            })?;
        }
        let mut stream = Self {
            state,
            front: 3,
            rear: 0,
        };
        for _ in 0..310 {
            stream.next_u31();
        }
        Ok(stream)
    }

    /// Return the next value, as glibc `rand` does.
    pub fn next_u31(&mut self) -> u32 {
        #[expect(
            clippy::indexing_slicing,
            reason = "GlibcRand keeps front and rear below 31"
        )]
        let value = {
            let sum = self.state[self.front].wrapping_add(self.state[self.rear]);
            self.state[self.front] = sum;
            sum
        };
        self.front = advance(self.front);
        self.rear = advance(self.rear);
        value >> 1
    }
}

fn advance(index: usize) -> usize {
    index.checked_add(1).map_or(0, |next| next % 31)
}

#[cfg(kani)]
mod kani_proofs {
    // The seed is written into `state` only.
    // `new` sets these indices to 3 and 0, then calls `next_u31`.
    #[kani::proof]
    #[kani::unwind(34)]
    fn prove_glibc_rand_indices_stay_below_31() {
        let mut state = [0u32; 31];
        let mut slot = 0usize;
        while slot < state.len() {
            if let Some(place) = state.get_mut(slot) {
                *place = kani::any();
            }
            slot = match slot.checked_add(1) {
                Some(next) => next,
                None => break,
            };
        }
        let front: usize = kani::any();
        let rear: usize = kani::any();
        kani::assume(front < 31);
        kani::assume(rear < 31);
        let mut stream = super::GlibcRand { state, front, rear };
        assert!(stream.front < 31);
        assert!(stream.rear < 31);
        stream.next_u31();
        assert!(stream.front < 31);
        assert!(stream.rear < 31);
        let mut extra: u8 = kani::any();
        kani::assume(extra <= 3);
        while extra > 0 {
            stream.next_u31();
            assert!(stream.front < 31);
            assert!(stream.rear < 31);
            extra = match extra.checked_sub(1) {
                Some(next) => next,
                None => break,
            };
        }
    }
}
