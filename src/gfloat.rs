use std::cmp::Ordering;
use std::num::FpCategory;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use num::{Float, Num, NumCast, One, ToPrimitive, Zero};
use crate::{BinaryDOp, DFloat, NaryDOp, UnaryDOp};

#[derive(Debug)]
pub struct GFloat<F: Float, const ARGWIDTH: usize> {
    pub value: F,
    pub grad: [F;ARGWIDTH]
}

impl<F: Float, const ARGWIDTH: usize> GFloat<F, ARGWIDTH> {
    fn nan_grad(&self, new_value: F) -> GFloat<F,ARGWIDTH> {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            if self.grad[i] != F::zero() {
                grad[i] = F::nan();
            }
        }
        GFloat {
            value: new_value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> GFloat<F, ARGWIDTH> {
    pub fn from(value: F) -> GFloat<F, ARGWIDTH> {
        GFloat {
            value,
            grad: [F::zero();ARGWIDTH]
        }
    }

    pub fn new_var(value: F, index: usize) -> GFloat<F, ARGWIDTH> {
        let mut grad = [F::zero(); ARGWIDTH];
        grad[index] = F::one();
        GFloat {
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

    pub fn apply_u<O: UnaryDOp<F>>(self) -> GFloat<F, ARGWIDTH> {
        let mut result_grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            result_grad[i] = result_grad[i] * O::grad(self.partial(i));
        }
        GFloat {
            value: O::op(self.value),
            grad: result_grad
        }
    }

    pub fn apply_b<O: BinaryDOp<F>>(self, other: GFloat<F,ARGWIDTH>) -> GFloat<F, ARGWIDTH> {
        let mut result_grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            let op_grad = O::grad(self.partial(i),other.partial(i));
            result_grad[i] = self.grad[i] * op_grad.0;
            result_grad[i] = result_grad[i] + other.grad[i] * op_grad.1;
        }
        GFloat {
            value: O::op(self.value, other.value),
            grad: result_grad
        }
    }

    pub fn apply_n<O: NaryDOp<F,WIDTH>, const WIDTH: usize>(args: [GFloat<F,ARGWIDTH>;WIDTH]) -> GFloat<F, ARGWIDTH> {
        let mut result_grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            let op_grad = O::grad(args.map(|f| f.partial(i)));
            for j in 0..WIDTH {
                result_grad[i] = result_grad[i] + args[i].grad[j] * op_grad[j];
            }
        }
        GFloat {
            value: O::op(args.map(|f| f.value)),
            grad: result_grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Num for GFloat<F, ARGWIDTH> {
    type FromStrRadixErr = F::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(GFloat::from)
    }
}

impl<F: Float, const ARGWIDTH: usize> PartialEq for GFloat<F, ARGWIDTH> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F: Float, const ARGWIDTH: usize> Zero for GFloat<F, ARGWIDTH> {
    fn zero() -> Self {
        GFloat::from(F::zero())
    }

    fn is_zero(&self) -> bool {
        self.value == F::zero()
    }
}

impl<F: Float, const ARGWIDTH: usize> One for GFloat<F, ARGWIDTH> {
    fn one() -> Self {
        GFloat::from(F::one())
    }
}

impl<F: Float, const ARGWIDTH: usize> Add<Self> for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = self.grad[i] + rhs.grad[i];
        }
        GFloat {
            value: self.value + rhs.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Mul<Self> for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = self.value * rhs.grad[i] + self.grad[i] * rhs.value;
        }
        GFloat {
            value: self.value * rhs.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Sub<Self> for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = self.grad[i] - rhs.grad[i];
        }
        GFloat {
            value: self.value - rhs.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Div<Self> for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = (self.grad[i] * rhs.value - rhs.grad[i] * self.value)
                / (rhs.value * rhs.value);
        }
        GFloat {
            value: self.value / rhs.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Rem<Self> for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {

        let new_value = self.value % rhs.value;

        if new_value == F::zero() {
            let mut grad = [F::zero(); ARGWIDTH];
            for i in 0..ARGWIDTH {
                if self.grad[i] != F::zero() {
                    grad[i] = F::nan();
                }
            }
            return GFloat {
                value: new_value,
                grad
            }
        } else {
            let mut grad = [F::zero(); ARGWIDTH];
            for i in 0..ARGWIDTH {
                grad[i] = self.grad[i];
                grad[i] = grad[i] - rhs.grad[i] * (self.value / rhs.value).floor();
            }

            GFloat {
                value: new_value,
                grad
            }
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Copy for GFloat<F, ARGWIDTH> {}

impl<F: Float, const ARGWIDTH: usize> Clone for GFloat<F, ARGWIDTH> {
    fn clone(&self) -> Self {
        GFloat {
            value: self.value,
            grad: self.grad,
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> NumCast for GFloat<F, ARGWIDTH> {
    fn from<P: ToPrimitive>(n: P) -> Option<Self> {
        F::from(n).map(GFloat::from)
    }
}

// Minimal implementation to keep exact behaviour of running self.value.to_whatever() I think
impl<F: Float, const ARGWIDTH: usize> ToPrimitive for GFloat<F, ARGWIDTH> {
    fn to_i64(&self) -> Option<i64> { self.value.to_i64() }
    fn to_i128(&self) -> Option<i128> { self.value.to_i128() }
    fn to_u64(&self) -> Option<u64> { self.value.to_u64() }
    fn to_f32(&self) -> Option<f32> { self.value.to_f32() }
    fn to_f64(&self) -> Option<f64> { self.value.to_f64() }
}

impl<F: Float, const ARGWIDTH: usize> PartialOrd for GFloat<F, ARGWIDTH> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<F: Float, const ARGWIDTH: usize> Neg for GFloat<F, ARGWIDTH> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = -self.grad[i];
        }
        GFloat {
            value: -self.value,
            grad
        }
    }
}

impl<F: Float, const ARGWIDTH: usize> Float for GFloat<F, ARGWIDTH> {
    fn nan() -> Self { GFloat::from(F::nan()) }
    fn infinity() -> Self { GFloat::from(F::infinity()) }
    fn neg_infinity() -> Self { GFloat::from(F::neg_infinity()) }
    fn neg_zero() -> Self { GFloat::from(F::neg_zero()) }
    fn min_value() -> Self { GFloat::from(F::min_value()) }
    fn min_positive_value() -> Self { GFloat::from(F::min_positive_value()) }
    fn max_value() -> Self { GFloat::from(F::max_value()) }

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
            GFloat {
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
            GFloat {
                value: result,
                grad: [F::zero(); ARGWIDTH]
            }
        }
    }
    fn round(self) -> Self {
        let result = self.value.round();
        if result == self.value {
            self.nan_grad(result)
        } else {
            GFloat {
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
            GFloat {
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
            GFloat {
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
            GFloat::from(self.value.signum())
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
        GFloat {
            value: self.value.mul_add(mult.value, add.value),
            grad
        }
    }

    fn recip(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value * self.value);
        }
        GFloat {
            value: self.value.recip(),
            grad
        }
    }

    fn powi(self, n: i32) -> Self {
        let mut grad = [F::zero(); ARGWIDTH];
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.powi(n-1);
        }
        GFloat {
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
        GFloat {
            value: self.value.powf(n.value),
            grad
        }
    }

    fn sqrt(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / self.value.sqrt();
        }
        GFloat {
            value: self.value.sqrt(),
            grad
        }
    }

    fn exp(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.exp();
        }
        GFloat {
            value: self.value.exp(),
            grad
        }
    }

    fn exp2(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            // If the base float can't support 2, I don't think it deserves having 2^x
            grad[i] = grad[i] * F::from(2).expect("Float type should support 2").ln() * self.value.exp();
        }
        GFloat {
            value: self.value.exp2(),
            grad
        }
    }

    fn ln(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / self.value;
        }
        GFloat {
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
        GFloat {
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
        GFloat {
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
        GFloat {
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
            return GFloat {
                value: self.value,
                grad
            }
        }
        if self.value > other.value {
            GFloat {
                value: self.value,
                grad: self.grad
            }
        } else {
            GFloat {
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
            GFloat {
                value: self.value,
                grad
            }
        } else if self.value < other.value {
            GFloat {
                value: self.value,
                grad: self.grad
            }
        } else {
            GFloat {
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
            GFloat {
                value: F::zero(),
                grad
            }
        } else if self.value < other.value {
            GFloat {
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
        GFloat {
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
        GFloat {
            value: hypot,
            grad
        }
    }

    fn sin(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.cos();
        }
        GFloat {
            value: self.value.sin(),
            grad
        }
    }

    fn cos(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * -self.value.sin();
        }
        GFloat {
            value: self.value.cos(),
            grad
        }
    }

    fn tan(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value.cos() * self.value.cos());
        }
        GFloat {
            value: self.value.tan(),
            grad
        }
    }

    fn asin(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (F::one() - self.value * self.value).sqrt();
        }
        GFloat {
            value: self.value.asin(),
            grad
        }
    }

    fn acos(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = - grad[i] / (F::one() - self.value * self.value).sqrt();
        }
        GFloat {
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
        GFloat {
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

        GFloat {
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
            GFloat {
                value: sin,
                grad: grads
            },
            GFloat {
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
        GFloat {
            value: self.value.exp_m1(),
            grad
        }
    }

    fn ln_1p(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value + F::one());
        }
        GFloat {
            value: self.value.ln_1p(),
            grad
        }
    }

    fn sinh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.cosh();
        }
        GFloat {
            value: self.value.sinh(),
            grad
        }
    }

    fn cosh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] * self.value.sinh();
        }
        GFloat {
            value: self.value.cosh(),
            grad
        }
    }

    fn tanh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value.cosh() * self.value.cosh());
        }
        GFloat {
            value: self.value.tanh(),
            grad
        }
    }

    fn asinh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (F::one() + self.value * self.value).sqrt();
        }
        GFloat {
            value: self.value.asinh(),
            grad
        }
    }

    fn acosh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (self.value * self.value - F::one()).sqrt();
        }
        GFloat {
            value: self.value.acosh(),
            grad
        }
    }

    fn atanh(self) -> Self {
        let mut grad = self.grad;
        for i in 0..ARGWIDTH {
            grad[i] = grad[i] / (F::one() - self.value * self.value);
        }
        GFloat {
            value: self.value.atanh(),
            grad
        }
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.value.integer_decode()
    }
}