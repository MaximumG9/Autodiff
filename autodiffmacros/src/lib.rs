use proc_macro::TokenStream;
use quote::{quote};

#[proc_macro]
pub fn grad(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate.
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation.
    impl_hello_macro(&ast)
}

fn impl_hello_macro(ast: &syn::ExprCall) -> TokenStream {
    let mut result = proc_macro2::TokenStream::new();
    let arglen = ast.args.len();
    ast.args.iter().enumerate()
        .map(|(i,a)| {
            return quote! {autodiff::fwdgfloat::FwdGFloat::<_,#arglen>::new_var(#a, #i),}
        })
        .for_each(
            |a| result.extend(a)
        );

    let call = &ast.func;

    quote! {
        #call (#result)
    }.into()
}