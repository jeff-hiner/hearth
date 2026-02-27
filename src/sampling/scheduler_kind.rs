//! Scheduler selection enum for CLI and ComfyUI integration.

use super::{DiffusionSchedule, NoiseSchedule};
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString, VariantArray};

/// Available sigma schedules.
///
/// Used for CLI argument parsing and ComfyUI workflow deserialization.
/// Each variant selects a different sigma spacing strategy.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display, EnumString, VariantArray,
)]
#[strum(ascii_case_insensitive)]
#[serde(rename_all = "lowercase")]
pub enum SchedulerKind {
    /// Normal schedule (linear timestep spacing).
    #[strum(serialize = "Normal")]
    Normal,
    /// Karras schedule (exponential sigma spacing, often better quality).
    #[strum(serialize = "Karras")]
    Karras,
}

impl SchedulerKind {
    /// Build a [`DiffusionSchedule`] from this scheduler kind.
    ///
    /// # Arguments
    /// * `noise_schedule` - The noise schedule to derive sigmas from
    /// * `num_steps` - Number of sampling steps
    pub fn schedule(&self, noise_schedule: &NoiseSchedule, num_steps: usize) -> DiffusionSchedule {
        match self {
            Self::Normal => noise_schedule.schedule_for_steps(num_steps),
            Self::Karras => {
                let (sigma_min, sigma_max) = noise_schedule.sigma_range();
                noise_schedule.schedule_karras(num_steps, sigma_min, sigma_max, 7.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_variants() {
        assert_eq!(
            "normal".parse::<SchedulerKind>().unwrap(),
            SchedulerKind::Normal
        );
        assert_eq!(
            "karras".parse::<SchedulerKind>().unwrap(),
            SchedulerKind::Karras
        );
    }

    #[test]
    fn display_roundtrip() {
        for kind in [SchedulerKind::Normal, SchedulerKind::Karras] {
            let s = kind.to_string();
            let parsed: SchedulerKind = s.parse().unwrap();
            assert_eq!(parsed, kind);
        }
    }

    #[test]
    fn schedule_builds() {
        let noise_schedule = NoiseSchedule::linear(1000, 0.00085, 0.012);
        for kind in [SchedulerKind::Normal, SchedulerKind::Karras] {
            let schedule = kind.schedule(&noise_schedule, 20);
            assert_eq!(schedule.sigmas.len(), 21);
            assert_eq!(schedule.timesteps.len(), 20);
        }
    }

    #[test]
    fn parse_unknown() {
        assert!("unknown".parse::<SchedulerKind>().is_err());
    }

    #[test]
    fn parse_case_insensitive() {
        assert_eq!(
            "Normal".parse::<SchedulerKind>().unwrap(),
            SchedulerKind::Normal
        );
        assert_eq!(
            "KARRAS".parse::<SchedulerKind>().unwrap(),
            SchedulerKind::Karras
        );
    }

    #[test]
    fn variants_count() {
        assert_eq!(SchedulerKind::VARIANTS.len(), 2);
    }
}
