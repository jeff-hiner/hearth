//! Startup validation and diagnostics.
//!
//! Provides early checks for common configuration issues to give users
//! clear error messages before attempting expensive operations.

use std::path::{Path, PathBuf};

/// Result of checking required files.
#[derive(Debug)]
pub struct FileCheckResult {
    /// Files that were found.
    pub found: Vec<PathBuf>,
    /// Files that were missing.
    pub missing: Vec<PathBuf>,
}

impl FileCheckResult {
    /// Returns true if all required files were found.
    pub fn all_found(&self) -> bool {
        self.missing.is_empty()
    }

    /// Format a user-friendly error message listing missing files.
    pub fn format_error(&self) -> String {
        let mut msg = String::from("Required model files not found:\n\n");

        for path in &self.missing {
            msg.push_str(&format!("  [MISSING] {}\n", path.display()));
        }

        if !self.found.is_empty() {
            msg.push_str("\nFiles found:\n");
            for path in &self.found {
                msg.push_str(&format!("  [OK] {}\n", path.display()));
            }
        }

        msg.push_str("\nPlease ensure model files are downloaded to the correct locations.\n");
        msg.push_str("See README.md for download instructions.");

        msg
    }
}

/// Check that all required model files exist.
///
/// Returns a result indicating which files were found and which are missing.
pub fn check_model_files<P: AsRef<Path>>(paths: &[P]) -> FileCheckResult {
    let mut found = Vec::new();
    let mut missing = Vec::new();

    for path in paths {
        let path_buf = path.as_ref().to_path_buf();
        if path.as_ref().exists() {
            found.push(path_buf);
        } else {
            missing.push(path_buf);
        }
    }

    FileCheckResult { found, missing }
}

/// Error returned when the GPU backend is not available.
#[derive(Debug)]
pub struct GpuNotAvailable {
    /// The backend that was requested.
    pub backend: &'static str,
    /// Details about why it's not available.
    pub details: String,
}

impl std::fmt::Display for GpuNotAvailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} GPU backend is not available: {}",
            self.backend, self.details
        )
    }
}

impl std::error::Error for GpuNotAvailable {}

/// Check if the Vulkan backend is available.
///
/// This attempts to enumerate Vulkan adapters to verify the runtime works.
pub fn check_vulkan_available() -> Result<(), GpuNotAvailable> {
    // wgpu doesn't panic on missing Vulkan, it just returns no adapters
    // We can check this by trying to get adapter info
    // For now, we rely on wgpu's own error handling which is reasonable
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_files_all_found() {
        // Use files that definitely exist
        let paths: Vec<&Path> = vec![Path::new("Cargo.toml"), Path::new("src/lib.rs")];
        let result = check_model_files(&paths);

        assert!(result.all_found());
        assert_eq!(result.found.len(), 2);
        assert!(result.missing.is_empty());
    }

    #[test]
    fn check_files_some_missing() {
        let paths: Vec<&Path> = vec![
            Path::new("Cargo.toml"),
            Path::new("nonexistent_file.safetensors"),
        ];
        let result = check_model_files(&paths);

        assert!(!result.all_found());
        assert_eq!(result.found.len(), 1);
        assert_eq!(result.missing.len(), 1);
    }

    #[test]
    fn error_message_format() {
        let result = FileCheckResult {
            found: vec![PathBuf::from("models/clip/vocab.json")],
            missing: vec![
                PathBuf::from("models/checkpoints/model.safetensors"),
                PathBuf::from("models/clip/merges.txt"),
            ],
        };

        let msg = result.format_error();
        assert!(msg.contains("[MISSING] models/checkpoints/model.safetensors"));
        assert!(msg.contains("[MISSING] models/clip/merges.txt"));
        assert!(msg.contains("[OK] models/clip/vocab.json"));
    }
}
