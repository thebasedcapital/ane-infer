fn main() {
    println!("cargo:rerun-if-changed=objc/ane_runtime.m");
    println!("cargo:rerun-if-changed=objc/ane_runtime.h");

    cc::Build::new()
        .file("objc/ane_runtime.m")
        .include("objc")
        .flag("-fno-objc-arc") // Manual retain/release — ARC can't handle id fields in C structs
        .flag("-fmodules")
        .compile("ane_runtime");

    // Link required frameworks at the Rust level
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOSurface");
    println!("cargo:rustc-link-lib=dylib=objc");
}
