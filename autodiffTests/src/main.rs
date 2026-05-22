fn main() {
}

#[cfg(test)]
mod tests {
    use num::Float;
    use autodiffmacros::grad;

    const EPSILON : f64 = 0.0001f64;
    fn similar_f64(a : f64, b : f64) -> bool {
        (a - b).abs() < EPSILON
    }
    fn similar_array<const WIDTH: usize>(a : [f64;WIDTH], b : [f64;WIDTH]) -> bool {
        a.iter()
            .zip(b)
            .all(
                |(a,b)|
                    (a - b).abs() < EPSILON
            )
    }

    fn paraboloid<F : Float>(a: F, b: F) -> F {
        a * a + b * b
    }

    #[test]
    fn simple_macros() {
        let result = grad!(paraboloid(5f64,2f64));
        assert!(similar_f64(result.value, 29f64));
        assert!(similar_array(result.grad, [10f64, 4f64]));
    }
}
