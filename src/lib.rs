use crate::gfloat::GFloat;
use num::Float;
use std::ops::{Add, Div, Mul, Sub};

pub mod gfloat;

pub trait UnaryDOp<F: Float> {
    // the actual operation
    fn op(a: F) -> F;
    // gradient/derivative of the operation
    fn grad(a: DFloat<F>) -> F;
}

pub trait BinaryDOp<F: Float> {
    // the actual operation
    fn op(a: F, b: F) -> F;
    // the gradient vector of the operation for a and b
    fn grad(a: DFloat<F>, b: DFloat<F>) -> (F, F);
}

pub trait NaryDOp<F: Float, const WIDTH: usize> {
    // the actual operation
    fn op(args: [F;WIDTH]) -> F;
    // the gradient vector of the operation for a and b
    fn grad(args: [DFloat<F>;WIDTH]) -> [F;WIDTH];
}

impl<F: Float, const ARGWIDTH: usize> Add<F> for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        GFloat {
            value: self.value + rhs,
            grad: self.grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Mul<F> for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * rhs;
        }
        GFloat {
            value: self.value * rhs,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Sub<F> for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self::Output {
        GFloat {
            value: self.value - rhs,
            grad: self.grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Div<F> for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn div(self, rhs: F) -> Self::Output {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / rhs;
        }
        GFloat {
            value: self.value / rhs,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> WrappedFloat<F> for GFloat<F,ARGWIDTH> {
    fn get(value: F) -> Self {
        GFloat {
            value,
            grad: [F::zero();ARGWIDTH]
        }
    }
}

pub trait WrappedFloat<F: Float>: Float
    + Add<F,Output=Self>
    + Mul<F,Output=Self>
    + Sub<F,Output=Self>
    + Div<F,Output=Self> {
    fn get(value: F) -> Self;
}

impl<F: Float> WrappedFloat<F> for F {
    fn get(value: F) -> Self {
        value
    }
}

// A struct representing a value, and its derivative along some argument
// DFloat doesn't have helper functions because the only current point is to feed to DOp::grad
pub struct DFloat<F: Float> {
    pub value: F,
    pub derivative: F
}
