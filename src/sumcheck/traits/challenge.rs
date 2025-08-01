//! Challenge generation abstraction for Fiat-Shamir heuristic.
//!
//! This module defines the `ChallengeGenerator` trait which provides
//! an abstract interface for generating challenges in the sum-check protocol,
//! enabling different implementations of the Fiat-Shamir heuristic.

use crate::utils::challenger::Challenger;

/// Abstract challenge generation for Fiat-Shamir heuristic.
///
/// This trait provides a unified interface for generating challenges
/// during the sum-check protocol execution. It abstracts away the
/// specific implementation details of the Fiat-Shamir heuristic,
/// allowing for different challenge generation strategies.
///
/// # Examples
///
/// ```rust
/// use deep_fri::sumcheck::traits::ChallengeGenerator;
/// use deep_fri::utils::challenger::Challenger;
///
/// // The Challenger type implements ChallengeGenerator
/// let mut challenger = Challenger::new();
/// // Use challenger to generate challenges...
/// ```
pub trait ChallengeGenerator<F> {
    /// Observe a polynomial by absorbing its coefficients into the challenge generator.
    ///
    /// # Arguments
    ///
    /// * `poly` - The polynomial coefficients to observe
    fn observe_polynomial(&mut self, poly: &[F]);

    /// Generate a challenge value.
    ///
    /// This method produces a pseudorandom challenge based on all previously
    /// observed data.
    ///
    /// # Returns
    ///
    /// A challenge value from the base field
    fn get_challenge(&mut self) -> F;

    /// Reset the challenge generator to its initial state.
    ///
    /// This method clears all observed data and resets the internal state,
    /// allowing the generator to be reused for a new protocol execution.
    fn reset(&mut self);

    /// Observe a single field element.
    ///
    /// # Arguments
    ///
    /// * `element` - The field element to observe
    fn observe_element(&mut self, element: F) {
        self.observe_polynomial(&[element]);
    }

    /// Observe multiple field elements.
    ///
    /// # Arguments
    ///
    /// * `elements` - The field elements to observe
    fn observe_elements(&mut self, elements: &[F]) {
        self.observe_polynomial(elements);
    }
}

/// Default implementation of ChallengeGenerator using the Challenger type.
impl<F> ChallengeGenerator<F> for Challenger {
    fn observe_polynomial(&mut self, poly: &[F]) {
        for &coeff in poly {
            self.observe_element(coeff);
        }
    }

    fn get_challenge(&mut self) -> F {
        self.sample()
    }

    fn reset(&mut self) {
        // Challenger doesn't have a reset method, so we do nothing
        // This is acceptable as Challenger is designed to be used once per protocol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;

    type F = BabyBear;

    #[test]
    fn test_challenge_generator_trait() {
        let mut challenger = Challenger::<F>::new();

        // Test observing elements
        challenger.observe_element(F::from_u32(42));
        challenger.observe_elements(&[F::from_u32(1), F::from_u32(2)]);

        // Test polynomial observation
        let poly = vec![F::from_u32(1), F::from_u32(2), F::from_u32(3)];
        challenger.observe_polynomial(&poly);

        // Test challenge generation
        let challenge = challenger.get_challenge();
        assert_ne!(challenge, F::zero());
    }
}
