# Autodiff
A rust crate for automatic differentiation/gradients. \
For an example, look at [the example crate](autodiffExamples/src/main.rs) or see the example below.

```rust
use num::Float;
use autodiff::gfloat::GFloat;
use autodiffmacros::grad;

fn paraboloid<F: Float>(x: F, y: F) -> F {
    x * x + y * y
}

fn main() {
    // Evaluate normally
    let result_normal: f64 = paraboloid(5f64, 2f64);
    println!("x^2 + y^2 paraboloid at (5,2): {:?}", result_normal);

    // Evaluate with gradient
    let result_w_grad: GFloat<f64, 2> = grad!(paraboloid(5f64,2f64));
    println!("x^2 + y^2 paraboloid at (5,2): {:?}", result_w_grad);
}
```