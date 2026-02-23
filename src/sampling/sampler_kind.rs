//! Sampler selection enum for CLI and ComfyUI integration.

use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

/// Available sampling algorithms.
///
/// Used for CLI argument parsing and ComfyUI workflow deserialization.
/// Each variant corresponds to a sampler implementation in the [`super`] module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerKind {
    /// Euler method (first-order deterministic).
    Euler,
    /// Euler Ancestral (first-order stochastic).
    EulerA,
    /// DPM++ 2M (second-order deterministic).
    #[serde(rename = "dpm++_2m")]
    DpmPp2m,
    /// DPM++ SDE (second-order stochastic).
    #[serde(rename = "dpm++_sde")]
    DpmPpSde,
}

impl FromStr for SamplerKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "euler" => Ok(Self::Euler),
            "euler_a" => Ok(Self::EulerA),
            "dpm++_2m" => Ok(Self::DpmPp2m),
            "dpm++_sde" => Ok(Self::DpmPpSde),
            _ => Err(format!(
                "unknown sampler '{s}', expected one of: euler, euler_a, dpm++_2m, dpm++_sde"
            )),
        }
    }
}

impl fmt::Display for SamplerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Euler => write!(f, "euler"),
            Self::EulerA => write!(f, "euler_a"),
            Self::DpmPp2m => write!(f, "dpm++_2m"),
            Self::DpmPpSde => write!(f, "dpm++_sde"),
        }
    }
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
}
