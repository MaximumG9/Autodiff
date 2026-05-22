use num::Float;

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

// Floats should have conversion from f64 so this forces that behaviour
pub trait FloatCast<F: Float> {
    fn get(value: F) -> Self;
}

impl<F: Float> FloatCast<f64> for F {
    fn get(value: f64) -> Self {
        F::from(value).expect("All floats should be able to convert from f64")
    }
}

// A struct representing a value, and its derivative along some argument
// DFloat doesn't have helper functions because the only current point is to feed to DOp::grad
pub struct DFloat<F: Float> {
    pub value: F,
    pub derivative: F
}
