from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

_FACTORY_BRIDGE_WIRE_KEYS = {
    "router_overrides": "_router_overrides",
    "perceptor_overrides": "_perceptor_overrides",
    "bridge_env_profile": "_env_profile",
}

_CONFIG_POLICY_WIRE_KEYS = {
    "sequence_length": "_policy_batch_length",
    "imagination_horizon": "_policy_horizon",
    "curiosity_enabled": "_policy_curiosity_enabled",
    "mastery_enabled": "_policy_mastery_enabled",
    "router_overrides": "_router_overrides",
    "perceptor_overrides": "_perceptor_overrides",
    "policy_env_profile": "_env_profile",
}

_CONFIG_POLICY_CANONICAL_KEYS = {
    *_CONFIG_POLICY_WIRE_KEYS.values(),
}

_NONCANONICAL_RESERVED_ALIASES: Tuple[str, ...] = (
    "router_overrides",
    "perceptor_overrides",
    "env_profile",
)


@dataclass(frozen=True)
class ConfigPolicyOverrides:
    sequence_length: Optional[int] = None
    imagination_horizon: Optional[int] = None
    curiosity_enabled: Optional[bool] = None
    mastery_enabled: Optional[bool] = None
    router_overrides: Dict[str, Any] = field(default_factory=dict)
    perceptor_overrides: Dict[str, Any] = field(default_factory=dict)
    policy_env_profile: Optional[Any] = None
    custom_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactoryBridgeOverrides:
    custom_overrides: Optional[Dict[str, Any]] = None
    router_overrides: Optional[Dict[str, Any]] = None
    perceptor_overrides: Optional[Dict[str, Any]] = None
    bridge_env_profile: Optional[Any] = None


def _reject_noncanonical_reserved_aliases(
    raw: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for alias in _NONCANONICAL_RESERVED_ALIASES:
        if alias in raw:
            raise ValueError(
                f"Non-canonical {context} '{alias}' is no longer supported. "
                "Use the underscored canonical key."
            )


def _decode_wire_sections(
    raw: Optional[Mapping[str, Any]],
    *,
    wire_keys: Mapping[str, str],
    context: str,
    reject_unknown_reserved: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    empty_sections = {section: None for section in wire_keys}
    if raw is None:
        return empty_sections, {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{context} must be a mapping.")

    data = dict(raw)
    _reject_noncanonical_reserved_aliases(data, context=context)

    sections = {
        section: data.pop(wire_key, None)
        for section, wire_key in wire_keys.items()
    }
    if reject_unknown_reserved:
        unexpected_keys = sorted(key for key in data if key.startswith("_"))
        if unexpected_keys:
            raise ValueError(
                f"Unexpected {context} reserved overrides: {unexpected_keys}"
            )
    return sections, data


def _encode_wire_sections(
    sections: Mapping[str, Any],
    *,
    wire_keys: Mapping[str, str],
    custom_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(custom_overrides or {})
    for section, wire_key in wire_keys.items():
        value = sections.get(section)
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        payload[wire_key] = value
    return payload


def parse_factory_bridge_overrides(
    custom_overrides: Optional[Mapping[str, Any]],
) -> FactoryBridgeOverrides:
    """Parse reserved factory bridge keys from a mixed overrides mapping."""
    sections, cleaned = _decode_wire_sections(
        custom_overrides,
        wire_keys=_FACTORY_BRIDGE_WIRE_KEYS,
        context="reserved override",
        reject_unknown_reserved=False,
    )

    return FactoryBridgeOverrides(
        custom_overrides=cleaned or None,
        router_overrides=sections["router_overrides"],
        perceptor_overrides=sections["perceptor_overrides"],
        bridge_env_profile=sections["bridge_env_profile"],
    )


def export_factory_bridge_overrides(
    bridge: FactoryBridgeOverrides,
) -> Dict[str, Any]:
    """Encode typed factory bridge overrides into reserved wire keys."""
    return _encode_wire_sections(
        {
            "router_overrides": dict(bridge.router_overrides or {}),
            "perceptor_overrides": dict(bridge.perceptor_overrides or {}),
            "bridge_env_profile": bridge.bridge_env_profile,
        },
        wire_keys=_FACTORY_BRIDGE_WIRE_KEYS,
        custom_overrides=bridge.custom_overrides,
    )


def parse_config_policy_overrides(
    raw: Optional[Mapping[str, Any]],
) -> ConfigPolicyOverrides:
    sections, custom_overrides = _decode_wire_sections(
        raw,
        wire_keys=_CONFIG_POLICY_WIRE_KEYS,
        context="ConfigPolicy",
        reject_unknown_reserved=True,
    )

    sequence_length = sections["sequence_length"]
    if sequence_length is not None:
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError(
                f"ConfigPolicy {_CONFIG_POLICY_WIRE_KEYS['sequence_length']} must be positive, got {sequence_length}"
            )

    imagination_horizon = sections["imagination_horizon"]
    if imagination_horizon is not None:
        imagination_horizon = int(imagination_horizon)
        if imagination_horizon <= 0:
            raise ValueError(
                f"ConfigPolicy {_CONFIG_POLICY_WIRE_KEYS['imagination_horizon']} must be positive, got {imagination_horizon}"
            )

    curiosity_enabled = sections["curiosity_enabled"]
    if curiosity_enabled is not None:
        curiosity_enabled = bool(curiosity_enabled)

    mastery_enabled = sections["mastery_enabled"]
    if mastery_enabled is not None:
        mastery_enabled = bool(mastery_enabled)

    router_overrides = sections["router_overrides"]
    if router_overrides is None:
        router_overrides = {}
    elif not isinstance(router_overrides, dict):
        raise ValueError(
            f"ConfigPolicy {_CONFIG_POLICY_WIRE_KEYS['router_overrides']} must be a mapping."
        )

    perceptor_overrides = sections["perceptor_overrides"]
    if perceptor_overrides is None:
        perceptor_overrides = {}
    elif not isinstance(perceptor_overrides, dict):
        raise ValueError(
            f"ConfigPolicy {_CONFIG_POLICY_WIRE_KEYS['perceptor_overrides']} must be a mapping."
        )

    return ConfigPolicyOverrides(
        sequence_length=sequence_length,
        imagination_horizon=imagination_horizon,
        curiosity_enabled=curiosity_enabled,
        mastery_enabled=mastery_enabled,
        router_overrides=dict(router_overrides),
        perceptor_overrides=dict(perceptor_overrides),
        policy_env_profile=sections["policy_env_profile"],
        custom_overrides=dict(custom_overrides),
    )


def export_config_policy_wire_overrides(
    policy: ConfigPolicyOverrides,
) -> Dict[str, Any]:
    """Encode canonical ConfigPolicy overrides into reserved wire keys."""
    return _encode_wire_sections(
        {
            "sequence_length": None if policy.sequence_length is None else int(policy.sequence_length),
            "imagination_horizon": None
            if policy.imagination_horizon is None
            else int(policy.imagination_horizon),
            "curiosity_enabled": None
            if policy.curiosity_enabled is None
            else bool(policy.curiosity_enabled),
            "mastery_enabled": None
            if policy.mastery_enabled is None
            else bool(policy.mastery_enabled),
            "router_overrides": dict(policy.router_overrides),
            "perceptor_overrides": dict(policy.perceptor_overrides),
            "policy_env_profile": policy.policy_env_profile,
        },
        wire_keys=_CONFIG_POLICY_WIRE_KEYS,
        custom_overrides=policy.custom_overrides,
    )
