use std::cmp::Ordering;
use std::num::FpCategory;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use num::{Float, Num, NumCast, One, ToPrimitive, Zero};
use crate::{BinaryDOp, DFloat, NaryDOp, UnaryDOp, WrappedFloat};

#[derive(Debug)]
pub struct FwdGFloat<F: Float, const ARGWIDTH: usize> {
    pub value: F,
    pub grad: [F;ARGWIDTH]
}

impl<F: Float, const ARGWIDTH: usize> FwdGFloat<F, ARGWIDTH> {
    fn nan_grad(&self, new_value: F) -> FwdGFloat<F,ARGWIDTH> {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            if self.grad[i] != F::zero() {
                grad[i] = F::nan();
            }
        }
        FwdGFloat {
            value: new_value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> FwdGFloat<F, ARGWIDTH> {
    pub fn new_var(value: F, index: usize) -> FwdGFloat<F, ARGWIDTH> {
        let mut grad = [F::zero(); ARGWIDTH];
        grad[index] = F::one();
        FwdGFloat {
            value,
            grad
        }
    }

    // Returns a DFloat with the value and the partial derivative along the index
    pub fn partial(self, index: usize) -> DFloat<F> {
        DFloat {
            value: self.value,
            derivative: self.grad[index]
        }
    }

    pub fn apply_u<O: UnaryDOp<F>>(self) -> FwdGFloat<F, ARGWIDTH> {
        let mut result_grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            result_grad[i] = result_grad[i] * O::grad(self.partial(i));
        }
        FwdGFloat {
            value: O::op(self.value),
            grad: result_grad
        }
    }

    pub fn apply_b<O: BinaryDOp<F>>(self, other: FwdGFloat<F,ARGWIDTH>) -> FwdGFloat<F, ARGWIDTH> {
        let mut result_grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            let op_grad = O::grad(self.partial(i),other.partial(i));
            result_grad[i] = self.grad[i] * op_grad.0;
            result_grad[i] = result_grad[i] + other.grad[i] * op_grad.1;
        }
        FwdGFloat {
            value: O::op(self.value, other.value),
            grad: result_grad
        }
    }

    pub fn apply_n<O: NaryDOp<F,WIDTH>, const WIDTH: usize>(args: [FwdGFloat<F,ARGWIDTH>;WIDTH]) -> FwdGFloat<F, ARGWIDTH> {
        let mut result_grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            let op_grad = O::grad(args.map(|f| f.partial(i)));
            for j in 0..WIDTH {
                result_grad[i] = result_grad[i] + args[i].grad[j] * op_grad[j];
            }
        }
        FwdGFloat {
            value: O::op(args.map(|f| f.value)),
            grad: result_grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Num for FwdGFloat<F, ARGWIDTH> {
    type FromStrRadixErr = F::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(FwdGFloat::get)
    }
}

impl<F: Float, const ARGWIDTH: usize> PartialEq for FwdGFloat<F, ARGWIDTH> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F: Float, const ARGWIDTH: usize> Zero for FwdGFloat<F, ARGWIDTH> {
    fn zero() -> Self {
        FwdGFloat::get(F::zero())
    }

    fn is_zero(&self) -> bool {
        self.value == F::zero()
    }
}

impl<F: Float, const ARGWIDTH: usize> One for FwdGFloat<F, ARGWIDTH> {
    fn one() -> Self {
        FwdGFloat::get(F::one())
    }
}

impl<F: Float, const ARGWIDTH: usize> Add<Self> for FwdGFloat<F, ARGWIDTH> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = self.grad[i] + rhs.grad[i];
        }
        FwdGFloat {
            value: self.value + rhs.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Mul<Self> for FwdGFloat<F, ARGWIDTH> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = self.value * rhs.grad[i] + self.grad[i] * rhs.value;
        }
        FwdGFloat {
            value: self.value * rhs.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Sub<Self> for FwdGFloat<F, ARGWIDTH> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = self.grad[i] - rhs.grad[i];
        }
        FwdGFloat {
            value: self.value - rhs.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Div<Self> for FwdGFloat<F, ARGWIDTH> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        if rhs.is_zero() {
            for i in 0..ARGWIDTH {
                if !self.grad[i].is_zero() {
                    grad[i] = -self.grad[i] / (rhs.value * rhs.value);
                }
                if !rhs.grad[i].is_zero() {
                    grad[i] = grad[i] - rhs.grad[i] / (rhs.value * rhs.value);
                }
            }
        } else {
            for i in 0..ARGWIDTH {
                grad[i] = ((self.grad[i] * rhs.value) - (rhs.grad[i] * self.value))
                    / (rhs.value * rhs.value);
            }
        }
        FwdGFloat {
            value: self.value / rhs.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Rem<Self> for FwdGFloat<F, ARGWIDTH> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {

        let new_value = self.value % rhs.value;

        if (new_value == F::zero() && !self.value.is_zero()) || new_value.is_nan() {
            let mut grad = [F::zero(); ARGWIDTH];
            for i in 0..ARGWIDTH {
                if !rhs.grad[i].is_zero() || !self.grad[i].is_zero() {
                    grad[i] = F::nan();
                }
            }
            FwdGFloat {
                value: new_value,
                grad
            }
        } else {
            let mut grad = [F::zero(); ARGWIDTH];
            for i in 0..ARGWIDTH {
                grad[i] = self.grad[i];
                grad[i] = grad[i] - (rhs.grad[i] * (self.value / rhs.value).floor());
            }

            FwdGFloat {
                value: new_value,
                grad
            }
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Copy for FwdGFloat<F, ARGWIDTH> {}

impl<F: Float, const ARGWIDTH: usize> Clone for FwdGFloat<F, ARGWIDTH> {
    fn clone(&self) -> Self {
        FwdGFloat {
            value: self.value,
            grad: self.grad,
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> NumCast for FwdGFloat<F, ARGWIDTH> {
    fn from<P: ToPrimitive>(n: P) -> Option<Self> {
        F::from(n).map(FwdGFloat::get)
    }
}

// Minimal implementation to keep exact behaviour of running self.value.to_whatever() I think
impl<F: Float, const ARGWIDTH: usize> ToPrimitive for FwdGFloat<F, ARGWIDTH> {
    fn to_i64(&self) -> Option<i64> { self.value.to_i64() }
    fn to_i128(&self) -> Option<i128> { self.value.to_i128() }
    fn to_u64(&self) -> Option<u64> { self.value.to_u64() }
    fn to_f32(&self) -> Option<f32> { self.value.to_f32() }
    fn to_f64(&self) -> Option<f64> { self.value.to_f64() }
}

impl<F: Float, const ARGWIDTH: usize> PartialOrd for FwdGFloat<F, ARGWIDTH> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<F: Float, const ARGWIDTH: usize> Neg for FwdGFloat<F, ARGWIDTH> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = -self.grad[i];
        }
        FwdGFloat {
            value: -self.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Float for FwdGFloat<F, ARGWIDTH> {
    fn nan() -> Self { FwdGFloat::get(F::nan()) }
    fn infinity() -> Self { FwdGFloat::get(F::infinity()) }
    fn neg_infinity() -> Self { FwdGFloat::get(F::neg_infinity()) }
    fn neg_zero() -> Self { FwdGFloat::get(F::neg_zero()) }
    fn min_value() -> Self { FwdGFloat::get(F::min_value()) }
    fn min_positive_value() -> Self { FwdGFloat::get(F::min_positive_value()) }
    fn max_value() -> Self { FwdGFloat::get(F::max_value()) }

    fn is_nan(self) -> bool { self.value.is_nan() }
    fn is_infinite(self) -> bool { self.value.is_infinite() }
    fn is_finite(self) -> bool { self.value.is_finite() }
    fn is_normal(self) -> bool { self.value.is_normal() }
    fn classify(self) -> FpCategory { self.value.classify() }

    fn floor(self) -> Self {
        let result = self.value.floor();
        if result == self.value {
            self.nan_grad(result)
        } else {
            FwdGFloat {
                value: result,
                grad: [F::zero(); ARGWIDTH]
            }
        }
    }
    fn ceil(self) -> Self {
        let result = self.value.ceil();
        if result == self.value {
            self.nan_grad(result)
        } else {
            FwdGFloat {
                value: result,
                grad: [F::zero(); ARGWIDTH]
            }
        }
    }
    fn round(self) -> Self {
        let result = self.value.round();
        if result == (self.value + F::from(0.5).unwrap()) {
            self.nan_grad(result)
        } else {
            FwdGFloat {
                value: result,
                grad: [F::zero(); ARGWIDTH]
            }
        }
    }
    fn trunc(self) -> Self {
        let result = self.value.trunc();
        if result == self.value {
            self.nan_grad(result)
        } else {
            FwdGFloat {
                value: result,
                grad: [F::zero(); ARGWIDTH]
            }
        }
    }

    fn fract(self) -> Self {
        let result = self.value.fract();
        if result == F::zero() {
            self.nan_grad(result)
        } else {
            FwdGFloat {
                value: result,
                grad: self.grad
            }
        }
    }
    fn abs(self) -> Self {
        if self.is_sign_negative() {
            self.neg()
        } else if self.is_zero() {
            self.nan_grad(F::zero())
        } else {
            self
        }
    }

    fn signum(self) -> Self {
        if self.is_zero() {
            self.nan_grad(F::zero())
        } else {
            FwdGFloat::get(self.value.signum())
        }
    }

    fn is_sign_positive(self) -> bool {
        self.value.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.value.is_sign_negative()
    }

    fn mul_add(self, mult: Self, add: Self) -> Self {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = self.value * mult.grad[i] + self.grad[i] * mult.value + add.grad[i];
        }
        FwdGFloat {
            value: self.value.mul_add(mult.value, add.value),
            grad
        }
    }

    fn recip(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value * self.value);
        }
        FwdGFloat {
            value: self.value.recip(),
            grad
        }
    }

    fn powi(self, n: i32) -> Self {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = self.grad[i] * F::from(n).expect("Float type must support n") * self.value.powi(n-1);
        }
        FwdGFloat {
            value: self.value.powi(n),
            grad
        }
    }

    fn powf(self, n: Self) -> Self {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = self.value.powf(n.value - F::one()) *
                (n.value * self.grad[i] + self.value * n.grad[i] * self.value.ln());
        }
        FwdGFloat {
            value: self.value.powf(n.value),
            grad
        }
    }

    fn sqrt(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (F::from(2).expect("Float should implement 2") * self.value.sqrt());
        }
        FwdGFloat {
            value: self.value.sqrt(),
            grad
        }
    }

    fn exp(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.exp();
        }
        FwdGFloat {
            value: self.value.exp(),
            grad
        }
    }

    fn exp2(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            // If the base float can't support 2, I don't think it deserves having 2^x
            grad[i] = grad[i] * F::from(2).expect("Float type should support 2").ln() * self.value.exp2();
        }
        FwdGFloat {
            value: self.value.exp2(),
            grad
        }
    }

    fn ln(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / self.value;
        }
        FwdGFloat {
            value: self.value.ln(),
            grad
        }
    }

    fn log(self, base: Self) -> Self {
        let log = self.value.log(base.value);

        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = (self.grad[i]/self.value - (base.grad[i] * log)/base.value)/base.value.ln();
        }
        FwdGFloat {
            value: log,
            grad
        }
    }

    fn log2(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            // pretty much same comment as for exp2
            grad[i] = grad[i] / (F::from(2).expect("Float type should support 2").ln() * self.value);
        }
        FwdGFloat {
            value: self.value.log2(),
            grad
        }
    }

    fn log10(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            // pretty much same comment as for exp2
            grad[i] = grad[i] / (F::from(10).expect("Float type should support 10").ln() * self.value);
        }
        FwdGFloat {
            value: self.value.log10(),
            grad
        }
    }

    fn max(self, other: Self) -> Self {
        if self.value == other.value {
            let mut grad = [F::zero(); ARGWIDTH];
            for i in 0..ARGWIDTH {
                if self.grad[i] != F::zero() || other.grad[i] != F::zero() {
                    grad[i] = F::nan();
                }
            }
            return FwdGFloat {
                value: self.value,
                grad
            }
        }
        if self.value > other.value {
            FwdGFloat {
                value: self.value,
                grad: self.grad
            }
        } else {
            FwdGFloat {
                value: other.value,
                grad: other.grad
            }
        }
    }

    fn min(self, other: Self) -> Self {
        if self.value == other.value {
            let mut grad = [F::zero(); ARGWIDTH];
            for i in 0..ARGWIDTH {
                if self.grad[i] != F::zero() || other.grad[i] != F::zero() {
                    grad[i] = F::nan();
                }
            }
            FwdGFloat {
                value: self.value,
                grad
            }
        } else if self.value < other.value {
            FwdGFloat {
                value: self.value,
                grad: self.grad
            }
        } else {
            FwdGFloat {
                value: other.value,
                grad: other.grad
            }
        }
    }

    fn abs_sub(self, other: Self) -> Self {
        if self.value == other.value {
            let mut grad = [F::zero(); ARGWIDTH];
            for i in 0..ARGWIDTH {
                if self.grad[i] != F::zero() || other.grad[i] != F::zero() {
                    grad[i] = F::nan();
                }
            }
            FwdGFloat {
                value: F::zero(),
                grad
            }
        } else if self.value < other.value {
            FwdGFloat {
                value: F::zero(),
                grad: [F::zero();ARGWIDTH]
            }
        } else {
            self - other
        }
    }

    fn cbrt(self) -> Self {
        let cbrt = self.value.cbrt();
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (cbrt.powi(2) * F::from(3).expect("Float type should support 3"));
        }
        FwdGFloat {
            value: cbrt,
            grad
        }
    }

    fn hypot(self, other: Self) -> Self {
        let hypot = self.value.hypot(other.value);
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = F::from(2).expect("Float type should support 2") * grad[i]
                * (self.value * self.grad[i] + other.grad[i] * other.value) / hypot;
        }
        FwdGFloat {
            value: hypot,
            grad
        }
    }

    fn sin(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.cos();
        }
        FwdGFloat {
            value: self.value.sin(),
            grad
        }
    }

    fn cos(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * -self.value.sin();
        }
        FwdGFloat {
            value: self.value.cos(),
            grad
        }
    }

    fn tan(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value.cos() * self.value.cos());
        }
        FwdGFloat {
            value: self.value.tan(),
            grad
        }
    }

    fn asin(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (F::one() - self.value * self.value).sqrt();
        }
        FwdGFloat {
            value: self.value.asin(),
            grad
        }
    }

    fn acos(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = - grad[i] / (F::one() - self.value * self.value).sqrt();
        }
        FwdGFloat {
            value: self.value.acos(),
            grad
        }
    }

    fn atan(self) -> Self {
        let mut grad = [F::zero(); ARGWIDTH];

        let denominator =  (F::one() + self.value * self.value).recip();

        for i in 0..ARGWIDTH {
            grad[i] = self.grad[i] * denominator;
        }
        FwdGFloat {
            value: self.value.atan(),
            grad
        }
    }

    fn atan2(self, other: Self) -> Self {
        let mut grad = [F::zero(); ARGWIDTH];

        let denominator = (self.value * self.value + other.value * other.value).recip();

        for i in 0..ARGWIDTH {
            grad[i] = self.grad[i] * denominator - other.grad[i] * denominator
        }

        FwdGFloat {
            value: self.value.atan2(other.value),
            grad
        }
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.value.sin_cos();

        let mut grads: [F; ARGWIDTH] = self.grad;
        let mut gradc: [F; ARGWIDTH] = self.grad;
        for i in 0..ARGWIDTH {
            grads[i] = grads[i] * cos;
            gradc[i] = -gradc[i] * sin;
        }

        (
            FwdGFloat {
                value: sin,
                grad: grads
            },
            FwdGFloat {
                value: cos,
                grad: gradc
            }
        )
    }

    fn exp_m1(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.exp();
        }
        FwdGFloat {
            value: self.value.exp_m1(),
            grad
        }
    }

    fn ln_1p(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value + F::one());
        }
        FwdGFloat {
            value: self.value.ln_1p(),
            grad
        }
    }

    fn sinh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.cosh();
        }
        FwdGFloat {
            value: self.value.sinh(),
            grad
        }
    }

    fn cosh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.sinh();
        }
        FwdGFloat {
            value: self.value.cosh(),
            grad
        }
    }

    fn tanh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value.cosh() * self.value.cosh());
        }
        FwdGFloat {
            value: self.value.tanh(),
            grad
        }
    }

    fn asinh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (F::one() + self.value * self.value).sqrt();
        }
        FwdGFloat {
            value: self.value.asinh(),
            grad
        }
    }

    fn acosh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value * self.value - F::one()).sqrt();
        }
        FwdGFloat {
            value: self.value.acosh(),
            grad
        }
    }

    fn atanh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (F::one() - self.value * self.value);
        }
        FwdGFloat {
            value: self.value.atanh(),
            grad
        }
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.value.integer_decode()
    }
}