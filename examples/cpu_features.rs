fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        println!("CPU Feature Detection:");
        println!("  AVX2: {}", std::arch::is_x86_feature_detected!("avx2"));
        println!(
            "  SSE4.1: {}",
            std::arch::is_x86_feature_detected!("sse4.1")
        );
        println!("  SHA-NI: {}", std::arch::is_x86_feature_detected!("sha"));
        println!("  AVX: {}", std::arch::is_x86_feature_detected!("avx"));
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("Not x86_64 architecture");
    }
}
