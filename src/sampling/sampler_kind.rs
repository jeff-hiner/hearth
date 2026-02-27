//! Sampler selection enum for CLI and ComfyUI integration.

use serde::{Deserialize, Serialize};
use strum::{Display, EnumString, VariantArray};

/// Available sampling algorithms.
///
/// Used for CLI argument parsing and ComfyUI workflow deserialization.
/// Each variant corresponds to a sampler implementation in the [`super`] module.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Display, EnumString, VariantArray,
)]
#[strum(ascii_case_insensitive)]
#[serde(rename_all = "snake_case")]
pub enum SamplerKind {
    /// Euler method (first-order deterministic).
    #[strum(serialize = "Euler")]
    Euler,
    /// Euler Ancestral (first-order stochastic).
    #[strum(serialize = "Euler_a", serialize = "Euler a")]
    EulerA,
    /// DPM++ 2M (second-order deterministic).
    #[serde(rename = "dpm++_2m")]
    #[strum(serialize = "DPM++_2M", serialize = "DPM++ 2M")]
    DpmPp2m,
    /// DPM++ SDE (second-order stochastic).
    #[serde(rename = "dpm++_sde")]
    #[strum(serialize = "DPM++_SDE", serialize = "DPM++ SDE")]
    DpmPpSde,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_variants() {
        assert_eq!("euler".parse::<SamplerKind>().unwrap(), SamplerKind::Euler);
        assert_eq!(
            "euler_a".parse::<SamplerKind>().unwrap(),
            SamplerKind::EulerA
        );
        assert_eq!(
            "dpm++_2m".parse::<SamplerKind>().unwrap(),
            SamplerKind::DpmPp2m
        );
        assert_eq!(
            "dpm++_sde".parse::<SamplerKind>().unwrap(),
            SamplerKind::DpmPpSde
        );
    }

    #[test]
    fn display_roundtrip() {
        for kind in [
            SamplerKind::Euler,
            SamplerKind::EulerA,
            SamplerKind::DpmPp2m,
            SamplerKind::DpmPpSde,
        ] {
            let s = kind.to_string();
            let parsed: SamplerKind = s.parse().unwrap();
            assert_eq!(parsed, kind);
        }
    }

    #[test]
    fn parse_unknown() {
        assert!("unknown".parse::<SamplerKind>().is_err());
    }

    #[test]
    fn parse_case_insensitive() {
        assert_eq!("Euler".parse::<SamplerKind>().unwrap(), SamplerKind::Euler);
        assert_eq!("EULER".parse::<SamplerKind>().unwrap(), SamplerKind::Euler);
        assert_eq!(
            "Euler a".parse::<SamplerKind>().unwrap(),
            SamplerKind::EulerA
        );
        assert_eq!(
            "Euler A".parse::<SamplerKind>().unwrap(),
            SamplerKind::EulerA
        );
        assert_eq!(
            "DPM++ SDE".parse::<SamplerKind>().unwrap(),
            SamplerKind::DpmPpSde
        );
        assert_eq!(
            "DPM++ 2M".parse::<SamplerKind>().unwrap(),
            SamplerKind::DpmPp2m
        );
    }

    #[test]
    fn variants_count() {
        assert_eq!(SamplerKind::VARIANTS.len(), 4);
    }
}
