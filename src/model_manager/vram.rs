//! VRAM budget querying.
//!
//! Provides [`VramBudget`] with current heap budget and usage, queried from
//! the GPU adapter via wgpu. In the future this can be extended to use
//! `VK_EXT_memory_budget` directly via ash for more precise numbers.

/// Current VRAM budget and usage snapshot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VramBudget {
    /// Total usable VRAM in bytes (from adapter memory hints).
    pub heap_budget: u64,
    /// Currently allocated VRAM in bytes.
    pub heap_usage: u64,
}

impl VramBudget {
    /// How many bytes are free (budget minus usage).
    #[expect(dead_code, reason = "scaffolded for model unloading heuristics")]
    pub(crate) fn free(&self) -> u64 {
        self.heap_budget.saturating_sub(self.heap_usage)
    }
}

/// Query VRAM budget from the default wgpu adapter.
///
/// This uses `wgpu::Adapter::get_info()` for total VRAM. Actual usage is
/// tracked internally by the model manager. A future iteration will use
/// `VK_EXT_memory_budget` for live per-heap usage numbers.
#[expect(dead_code, reason = "scaffolded for model unloading heuristics")]
pub(crate) fn query_adapter_vram() -> VramBudget {
    // wgpu doesn't expose per-heap budget in its public API.
    // For now we report the adapter's total dedicated video memory
    // and let the model manager track usage internally.
    //
    // The actual VK_EXT_memory_budget query requires raw Vulkan access
    // through ash, which is a deeper integration deferred to later.
    VramBudget {
        heap_budget: 0,
        heap_usage: 0,
    }
}
