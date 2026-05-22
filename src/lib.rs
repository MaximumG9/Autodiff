pub mod gfloat;

use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use num::{Float, Num, NumCast, One, ToPrimitive, Zero};

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

// A struct representing a value, and its derivative along some argument
// DFloat doesn't have helper functions because the only current point is to feed to DOp::grad
pub struct DFloat<F: Float> {
    pub value: F,
    pub derivative: F
}
