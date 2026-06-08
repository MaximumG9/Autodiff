use autodiff::gfloat::GFloat;
use autodiff::WrappedFloat;
use autodiffmacros::grad;

fn paraboloid<W: WrappedFloat<f64>>(x: W, y: W) -> W {
    x * x + y * y
}

fn quadratic_formula<W: WrappedFloat<f64>>(a: W, b: W, c: W) -> (W,W) {
    (
        (-b + (b * b - a * c * 4.0).sqrt()) / (a * 2.0),
        (-b - (b * b - a * c * 4.0).sqrt()) / (a * 2.0)
    )
}

fn main() {
    let result: GFloat<f64,2> = grad!(paraboloid(5f64,2f64));
    println!("x^2 + y^2 paraboloid at (5,2): {:?}",result);
    let (pos, neg) = grad!(quadratic_formula(1f64,0f64,-1f64));
    println!("quadratic formula for 1x^2 + 0x - 1, +: {:?} -: {:?}",pos,neg);
}