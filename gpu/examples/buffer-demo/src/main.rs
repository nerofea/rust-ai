fn main() {
    let output = rust_ai_gpu_runtime::add_f32(&[1., 2.], &[3., 4.]).expect("valid vectors");
    println!("CPU reference: {output:?}");
}
