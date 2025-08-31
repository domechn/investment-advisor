from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    max_single_name: float = 0.06
    sector_cap: float = 0.20
    var_soft_cap: float = 0.035


def _get_float(env_key: str, default: float) -> float:
    try:
        v = os.getenv(env_key)
        if v is None:
            return default
        fv = float(v)
        if fv <= 0:
            return default
        return fv
    except Exception:
        return default


def load_policy_from_env() -> PolicyConfig:
    """Load policy baselines from environment variables with sane defaults.

    Environment variables:
    - POLICY_MAX_SINGLE_NAME (default 0.06)
    - POLICY_SECTOR_CAP (default 0.20)
    - POLICY_VAR_SOFT_CAP (default 0.035)
    """
    return PolicyConfig(
        max_single_name=_get_float("POLICY_MAX_SINGLE_NAME", 0.06),
        sector_cap=_get_float("POLICY_SECTOR_CAP", 0.20),
        var_soft_cap=_get_float("POLICY_VAR_SOFT_CAP", 0.035),
    )


__all__ = ["PolicyConfig", "load_policy_from_env"]


