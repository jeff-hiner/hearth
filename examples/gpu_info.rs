//! GPU detection and capability report.
//!
//! Enumerates all GPU adapters and prints their capabilities.
//!
//! Usage:
//!   cargo run --example gpu_info

#![cfg(any(feature = "gpu-vulkan-f16", feature = "gpu-vulkan-bf16"))]

use wgpu::{Backends, Device, DeviceType, Instance, InstanceDescriptor, InstanceFlags, Queue};

fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    }
}

fn device_type_str(device_type: DeviceType) -> &'static str {
    match device_type {
        DeviceType::Other => "Other",
        DeviceType::IntegratedGpu => "Integrated GPU",
        DeviceType::DiscreteGpu => "Discrete GPU",
        DeviceType::VirtualGpu => "Virtual GPU",
        DeviceType::Cpu => "CPU",
    }
}

fn backend_str(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Noop => "Noop",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "DirectX 12",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::BrowserWebGpu => "WebGPU (Browser)",
    }
}

#[tokio::main]
async fn main() {
    let sep = "=".repeat(60);
    let sep_short = "-".repeat(40);

    // Create wgpu instance with all backends
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        flags: InstanceFlags::default(),
        backend_options: Default::default(),
        memory_budget_thresholds: Default::default(),
    });

    // Enumerate all adapters
    let adapters = instance.enumerate_adapters(Backends::all());

    println!("\n{sep}");
    println!("GPU Detection Report");
    println!("{sep}\n");

    if adapters.is_empty() {
        eprintln!("No GPU adapters found! Vulkan/GPU drivers may not be installed.");
        std::process::exit(1);
    }

    println!("Found {} adapter(s):\n", adapters.len());

    let mut vulkan_found = false;
    let mut discrete_gpu_found = false;

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        let limits = adapter.limits();

        println!("Adapter #{i}");
        println!("{sep_short}");
        println!("  Name:         {}", info.name);
        println!("  Vendor:       0x{:04X}", info.vendor);
        println!("  Device:       0x{:04X}", info.device);
        println!("  Type:         {}", device_type_str(info.device_type));
        println!("  Backend:      {}", backend_str(info.backend));
        println!("  Driver:       {}", info.driver);
        println!("  Driver Info:  {}", info.driver_info);
        println!();
        println!("  Limits:");
        println!(
            "    Max Texture 2D:       {}x{}",
            limits.max_texture_dimension_2d, limits.max_texture_dimension_2d
        );
        println!(
            "    Max Buffer Size:      {}",
            format_bytes(limits.max_buffer_size)
        );
        println!(
            "    Max Storage Buffer:   {}",
            format_bytes(limits.max_storage_buffer_binding_size as u64)
        );
        println!(
            "    Max Compute Workgroup Size: {}x{}x{}",
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z
        );
        println!(
            "    Max Compute Invocations:    {}",
            limits.max_compute_invocations_per_workgroup
        );
        println!();

        if info.backend == wgpu::Backend::Vulkan {
            vulkan_found = true;
        }
        if info.device_type == DeviceType::DiscreteGpu {
            discrete_gpu_found = true;
        }
    }

    println!("{sep}");
    println!("Summary:");
    println!(
        "  Vulkan support: {}",
        if vulkan_found { "Yes" } else { "No" }
    );
    println!(
        "  Discrete GPU:   {}",
        if discrete_gpu_found { "Yes" } else { "No" }
    );
    println!("{sep}");

    // Try to create a device with the best adapter
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await;

    let Ok(adapter) = adapter else {
        eprintln!("\nFailed to find a suitable GPU adapter.");
        std::process::exit(1);
    };

    let info = adapter.get_info();
    println!(
        "\nSelected adapter: {} ({})",
        info.name,
        backend_str(info.backend)
    );

    let result: Result<(Device, Queue), _> = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("hearth"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
        })
        .await;

    let Ok((device, queue)) = result else {
        eprintln!("Failed to create GPU device.");
        std::process::exit(1);
    };

    println!("Successfully created device and queue.");

    // Verify we can submit an empty command buffer (basic sanity check)
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });
    queue.submit(std::iter::once(encoder.finish()));

    println!("Successfully submitted command buffer.");
}
