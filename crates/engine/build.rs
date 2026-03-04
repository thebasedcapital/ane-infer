fn main() {
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // Compile Metal shaders if .metal source is newer than .metallib
    let metal_src = "metal/q8_gemv.metal";
    let metal_lib = "metal/q8_gemv.metallib";
    println!("cargo:rerun-if-changed={metal_src}");

    if std::path::Path::new(metal_src).exists() {
        let air = "metal/q8_gemv.air";
        let status = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c", metal_src, "-o", air])
            .status();
        if let Ok(s) = status {
            if s.success() {
                let _ = std::process::Command::new("xcrun")
                    .args(["-sdk", "macosx", "metallib", air, "-o", metal_lib])
                    .status();
            }
        }
    }
}
