use num::Float;
use autodiff::gfloat::GFloat;
use autodiffmacros::grad;
use autodiff::FloatCast;

fn paraboloid<F: Float>(x: F, y: F) -> F {
    x * x + y * y
}

fn quadratic_formula<F: Float>(a: F, b: F, c: F) -> (F,F) {
    (
        (-b + (b * b - F::get(4.0) * a * c).sqrt()) / (F::get(2.0) * a),
        (-b - (b * b - F::get(4.0) * a * c).sqrt()) / (F::get(2.0) * a)
    )
}

fn main() {
    let result: GFloat<f64,2> = grad!(paraboloid(5f64,2f64));
    println!("x^2 + y^2 paraboloid at (5,2): {:?}",result);
    let (pos, neg) = grad!(quadratic_formula(1f64,0f64,-1f64));
    println!("quadratic formula for 1x^2 + 0x - 1, +: {:?} -: {:?}",pos,neg);
}