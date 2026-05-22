fn main() {
}

#[cfg(test)]
mod tests {
    use num::Float;
    use autodiff::GFloat;
    use autodiffmacros::grad;

    fn test<F : Float>(a: F, b: F) -> F {
        a * a + b * b
    }

    #[test]
    fn simple_macros() {
        let result = grad!(test(1f64,2f64));

        assert_eq!(result, GFloat {
            value: 10f64,
            grad: [1f64,1f64]
        });
    }
}
