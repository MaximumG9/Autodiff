
#[cfg(test)]
mod tests {
    use autodiff::gfloat::GFloat;
    use autodiffmacros::grad;
    use num::traits::Inv;
    use num::{Float, Zero};

    const CASES: [f64; 10] = [0.,1.,-1.,0.5,0.2,10000.,-17.59,285.32,-9.5,88.5];

    const fn build_case_arr(cases: [f64;10]) -> [(f64,f64);100] {
        let mut result = [(0.0, 0.0);100];
        let mut i = 0;
        while i < 10 {
            let mut j = 0;
            while j < 10 {
                result[i * 10 + j] = (cases[i],cases[j]);
                j += 1;
            }
            i += 1;
        }
        result
    }
    const CASE_PAIR: [(f64,f64);100] = build_case_arr(CASES);
    const EPSILON : f64 = 1E-11;

    fn assert_similar(a: f64,b: f64, case : (f64,f64)) {
        if a.is_finite() && a.is_finite() {
            assert!(a - b < EPSILON, "{} !~= {} {:?}", a,b,case);
        }
    }

    fn assert_similars<const SIZE: usize>(a : [f64;SIZE],b : [f64;SIZE], case : (f64,f64)) {
        for i in 0..SIZE {
            assert_similar(a[i],b[i], case);
        }
    }

    #[test]
    fn test_add() {

        let f = |a : GFloat<f64,2>,b : GFloat<f64,2>| a + b;
        for (a,b) in CASE_PAIR {
            assert_similar(grad!(f(a,b)).grad[0],1.0, (a,b));
            assert_similar(grad!(f(a,b)).grad[1],1.0, (a,b));
            assert_similar(grad!(f(a,b)).value,a + b, (a,b));
        }
    }
    #[test]
    fn test_eq() {

        for (a_f,b_f) in CASE_PAIR {
            if a_f == b_f {continue;}
            let a : GFloat<_,3> = GFloat::new_var(a_f, 1);
            let b : GFloat<_,3> = GFloat::new_var(a_f, 0);
            let c : GFloat<_,3> = GFloat::new_var(b_f, 0);

            assert_eq!(a, b);
            assert_ne!(a, c);
            assert_ne!(b, c);
        }
    }

    #[test]
    fn test_mul() {

        let f = |a : GFloat<f64,2>,b : GFloat<f64,2>| a * b;
        for (a,b) in CASE_PAIR {
            let gresult = grad!(f(a,b));
            assert_similars(
                gresult.grad,
                [b,a],
                (a,b)
            );
            assert_similar(
                gresult.value,
                a * b,
                (a,b)
            );
        }
    }

    #[test]
    fn test_sub() {

        let f = |a : GFloat<f64,2>,b : GFloat<f64,2>| a - b;
        for (a,b) in CASE_PAIR {
            let gresult = grad!(f(a,b));
            assert_similars(
                gresult.grad,
                [1.0,-1.0],
                (a,b)
            );
            assert_similar(
                gresult.value,
                a - b,
                (a,b)
            );
        }
    }

    #[test]
    fn test_div() {
        let f = |a : GFloat<f64,2>,b : GFloat<f64,2>| a / b;
        for (a,b) in CASE_PAIR {
            let gresult = grad!(f(a,b));
            assert_similars(
                gresult.grad,
                [
                    if a.is_zero() && b.is_zero() {f64::NAN} else {b.inv()},
                    a/(b*b)
                ],
                (a,b)
            );
            assert_similar(
                gresult.value,
                a / b,
                (a,b)
            );
        }
    }

    #[test]
    fn test_rem() {

        let f = |a : GFloat<f64,2>,b : GFloat<f64,2>| a % b;
        for (a,b) in CASE_PAIR {
            let gresult = grad!(f(a,b));
            assert_similars(
                gresult.grad,
                [
                    if ((a % b).is_zero() && !a.is_zero()) || (a % b).is_nan() {
                        f64::NAN
                    } else {1.0} ,
                    if ((a % b).is_zero() && !a.is_zero()) || (a % b).is_nan()
                    {f64::NAN} else
                    {-(a/b).floor()}
                ],
                (a,b)
            );
            assert_similar(
                gresult.value,
                a % b,
                (a,b)
            );
        }
    }

    #[test]
    fn test_neg() {
        let f = |a : GFloat<f64,2>,b : GFloat<f64,2>| a + -b;
        for (a,b) in CASE_PAIR {
            let gresult = grad!(f(a,b));
            assert_similars(
                gresult.grad,
                [1.0,-1.0],
                (a,b)
            );
            assert_similar(
                gresult.value,
                a - b,
                (a,b)
            );
        }
    }

    #[test]
    fn test_floor() {

        let f = |a : GFloat<f64,1>| a.floor();
        for a in CASES {
            let gresult = grad!(f(a));
            assert_similars(
                gresult.grad,
                [if a.fract() == 0.0 {f64::NAN} else {0.0}],
                (a,0.0)
            );
            assert_similar(
                gresult.value,
                a.floor(),
                (a,0.0)
            );
        }
    }
    #[test]
    fn test_ceil() {
        let f = |a : GFloat<f64,1>| a.ceil();
        for a in CASES {
            let gresult = grad!(f(a));
            assert_similars(
                gresult.grad,
                [if a.fract() == 0.0 {f64::NAN} else {0.0}],
                (a,0.0)
            );
            assert_similar(
                gresult.value,
                a.ceil(),
                (a,0.0)
            );
        }
    }
    #[test]
    fn test_round() {

        let f = |a : GFloat<f64,1>| a.round();
        for a in CASES {
            let gresult = grad!(f(a));
            assert_similars(
                gresult.grad,
                [if a.fract() == 0.5 {f64::NAN} else {0.0}],
                (a,0.0)
            );
            assert_similar(
                gresult.value,
                a.ceil(),
                (a,0.0)
            );
        }
    }
    #[test]
    fn test_trunc() {

        let f = |a : GFloat<f64,1>| a.ceil();
        for a in CASES {
            let gresult = grad!(f(a));
            assert_similars(
                gresult.grad,
                [if a.fract() == 0.0 {f64::NAN} else {0.0}],
                (a,0.0)
            );
            assert_similar(
                gresult.value,
                a.ceil(),
                (a,0.0)
            );
        }
    }
}