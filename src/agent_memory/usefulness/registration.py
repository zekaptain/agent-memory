"""Freeze registration, verification, registered MCP tool wording, and deviations (MEM-23).

Writes ``registration.json`` once at the freeze, naming every item required under
"The registration" in ``docs/memory-utility-evaluation-spec.md``, refusing when
any required item is missing or disagrees with the run's task manifest or memory
set. Enforces registration presence for held-out trials and checks that the task
manifest, memory set, environment manifest, and memory-tool wording still match
the registration before any trial starts. Deviations are append-only in
``deviations.jsonl``.
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config, ConfigError
from ..judge.store import utc_now_iso
from ..mcp_server import load_tool_wording_file, validate_tool_wording
from .memory_set import memory_set_manifest_path, verify_memory_set
from .store import (
    UsefulnessError,
    UsefulnessRunStore,
    ensure_experiment_home,
    sha256_file,
    sha256_json,
)

REGISTRATION_SCHEMA_VERSION = 1
EXACT_VALUE_STRATUM = "exact_value"
VALID_LIMIT_NAMES: tuple[str, ...] = ("tokens", "wall_time_s", "tool_calls", "retries")


def _require_mapping(parent: Mapping[str, Any], key: str, *, path: str = "") -> Mapping[str, Any]:
    full = f"{path}.{key}" if path else key
    if key not in parent or parent[key] is None:
        raise UsefulnessError(f"registration is missing required field: {full}")
    val = parent[key]
    if not isinstance(val, Mapping):
        raise UsefulnessError(f"registration field {full} must be a JSON object")
    return val


def _require_non_empty_str(parent: Mapping[str, Any], key: str, *, path: str = "") -> str:
    full = f"{path}.{key}" if path else key
    if key not in parent or parent[key] is None:
        raise UsefulnessError(f"registration is missing required field: {full}")
    val = parent[key]
    if not isinstance(val, str) or not val.strip():
        raise UsefulnessError(f"registration field {full} must be a non-empty string")
    return val


def _require_int(
    parent: Mapping[str, Any],
    key: str,
    *,
    path: str = "",
    minimum: int | None = None,
) -> int:
    full = f"{path}.{key}" if path else key
    if key not in parent or parent[key] is None:
        raise UsefulnessError(f"registration is missing required field: {full}")
    val = parent[key]
    if isinstance(val, bool) or not isinstance(val, int):
        raise UsefulnessError(f"registration field {full} must be an integer")
    if minimum is not None and val < minimum:
        raise UsefulnessError(f"registration field {full} must be >= {minimum}, got {val}")
    return val


def _require_positive_number(parent: Mapping[str, Any], key: str, *, path: str = "") -> float:
    full = f"{path}.{key}" if path else key
    if key not in parent or parent[key] is None:
        raise UsefulnessError(f"registration is missing required field: {full}")
    val = parent[key]
    if isinstance(val, bool) or not isinstance(val, int | float):
        raise UsefulnessError(f"registration field {full} must be a number")
    fval = float(val)
    if not math.isfinite(fval) or fval <= 0:
        raise UsefulnessError(f"registration field {full} must be finite and > 0, got {val!r}")
    return fval


def _require_str_list(
    parent: Mapping[str, Any],
    key: str,
    *,
    path: str = "",
    allow_empty: bool = False,
) -> list[str]:
    full = f"{path}.{key}" if path else key
    if key not in parent or parent[key] is None:
        raise UsefulnessError(f"registration is missing required field: {full}")
    val = parent[key]
    if not isinstance(val, list | tuple):
        raise UsefulnessError(f"registration field {full} must be a list of strings")
    items: list[str] = []
    for item in val:
        if not isinstance(item, str) or not item.strip():
            raise UsefulnessError(f"registration field {full} must contain non-empty strings")
        items.append(item)
    if not allow_empty and not items:
        raise UsefulnessError(f"registration field {full} must be non-empty")
    return items


def _require_non_empty_text_or_mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    path: str = "",
) -> str | dict[str, Any]:
    full = f"{path}.{key}" if path else key
    if key not in parent or parent[key] is None:
        raise UsefulnessError(f"registration is missing required field: {full}")
    val = parent[key]
    if isinstance(val, str):
        if not val.strip():
            raise UsefulnessError(f"registration field {full} must be non-empty")
        return val
    if isinstance(val, Mapping):
        if not val:
            raise UsefulnessError(f"registration field {full} must be a non-empty object")
        return dict(val)
    raise UsefulnessError(f"registration field {full} must be a non-empty string or object")


# ---------------------------------------------------------------------------
# Environment manifest & memory-tool wording helpers
# ---------------------------------------------------------------------------


def detect_repo_commit(repo_root: Path | None = None) -> str:
    """Return the current git HEAD commit SHA of this repository."""
    cwd = repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UsefulnessError(f"could not determine repository commit: {exc}") from None
    sha = proc.stdout.strip()
    if not sha:
        raise UsefulnessError("git rev-parse HEAD returned an empty commit SHA")
    return sha


def detect_hardware_class() -> str:
    """Return a stable descriptor of the local machine's hardware class."""
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    cpus = os.cpu_count() or 0
    return f"{system}-{machine}-{cpus}cpu"


def default_trial_config_overrides(auto_level: str) -> dict[str, Any]:
    """Canonical description of what an isolated trial reads instead of user config."""
    return {
        "auto_level": auto_level,
        "hooks_disabled": True,
        "cloud_session_sync": False,
        "builtin_skills_disabled": True,
        "isolated_factory_home": True,
    }


def environment_manifest_body(
    *,
    harness: str,
    harness_version: str,
    repo_commit: str,
    allowed_base_tools: Sequence[str],
    memory_server_name: str,
    hardware_class: str,
    config_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical unhashed body of an environment manifest."""
    if not isinstance(harness, str) or not harness.strip():
        raise UsefulnessError("environment_manifest.harness must be a non-empty string")
    if not isinstance(harness_version, str) or not harness_version.strip():
        raise UsefulnessError("environment_manifest.harness_version must be a non-empty string")
    if not isinstance(repo_commit, str) or not repo_commit.strip():
        raise UsefulnessError("environment_manifest.repo_commit must be a non-empty string")
    tools = [t for t in allowed_base_tools if isinstance(t, str) and t.strip()]
    if not tools or len(tools) != len(allowed_base_tools):
        raise UsefulnessError(
            "environment_manifest.allowed_base_tools must be a non-empty list of strings"
        )
    if not isinstance(memory_server_name, str) or not memory_server_name.strip():
        raise UsefulnessError("environment_manifest.memory_server_name must be a non-empty string")
    if not isinstance(hardware_class, str) or not hardware_class.strip():
        raise UsefulnessError("environment_manifest.hardware_class must be a non-empty string")
    if not isinstance(config_overrides, Mapping) or not config_overrides:
        raise UsefulnessError("environment_manifest.config_overrides must be a non-empty object")
    return {
        "harness": harness,
        "harness_version": harness_version,
        "repo_commit": repo_commit,
        "allowed_base_tools": list(tools),
        "memory_server_name": memory_server_name,
        "hardware_class": hardware_class,
        "config_overrides": dict(config_overrides),
    }


def build_environment_manifest(
    *,
    harness: str,
    harness_version: str,
    repo_commit: str,
    allowed_base_tools: Sequence[str],
    memory_server_name: str,
    hardware_class: str,
    config_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and hash an environment manifest for registration."""
    body = environment_manifest_body(
        harness=harness,
        harness_version=harness_version,
        repo_commit=repo_commit,
        allowed_base_tools=allowed_base_tools,
        memory_server_name=memory_server_name,
        hardware_class=hardware_class,
        config_overrides=config_overrides,
    )
    return {**body, "sha256": sha256_json(body)}


def memory_tool_wording_sha256(
    instructions: str,
    descriptions: Mapping[str, str],
) -> str:
    """Canonical SHA-256 of MCP server instructions and tool descriptions."""
    try:
        norm_instructions, norm_descriptions = validate_tool_wording(
            {"instructions": instructions, "descriptions": descriptions}
        )
    except ConfigError as exc:
        raise UsefulnessError(str(exc)) from None
    return sha256_json({"instructions": norm_instructions, "descriptions": norm_descriptions})


def build_memory_tool_wording(
    instructions: str,
    descriptions: Mapping[str, str],
) -> dict[str, Any]:
    """Validate and hash MCP server instructions and per-tool descriptions."""
    try:
        norm_instructions, norm_descriptions = validate_tool_wording(
            {"instructions": instructions, "descriptions": descriptions}
        )
    except ConfigError as exc:
        raise UsefulnessError(str(exc)) from None
    return {
        "instructions": norm_instructions,
        "descriptions": norm_descriptions,
        "sha256": sha256_json(
            {"instructions": norm_instructions, "descriptions": norm_descriptions}
        ),
    }


# ---------------------------------------------------------------------------
# Registration schema validation, freeze, loading, and enforcement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistrationResult:
    """A written or loaded ``registration.json`` together with its file SHA-256."""

    registration: dict[str, Any]
    registration_path: Path
    registration_sha256: str


@dataclass(frozen=True)
class VerifiedRegistration:
    """Registration verified against the live run artifacts and trial environment."""

    registration: dict[str, Any]
    registration_sha256: str
    memory_tools_instructions: str
    memory_tools_descriptions: dict[str, str]
    memory_tools_sha256: str
    memory_set_dir: Path


def _validate_trials_per_task(raw: Any, task_ids: set[str]) -> int | dict[str, int]:
    if isinstance(raw, bool):
        raise UsefulnessError(
            "registration field task_manifest.trials_per_task must be an int or mapping"
        )
    if isinstance(raw, int):
        if raw < 1:
            raise UsefulnessError("registration field task_manifest.trials_per_task must be >= 1")
        return raw
    if isinstance(raw, Mapping) and raw:
        out: dict[str, int] = {}
        for tid, count in sorted(raw.items()):
            if not isinstance(tid, str) or tid not in task_ids:
                raise UsefulnessError(
                    f"registration task_manifest.trials_per_task names unknown task {tid!r}"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise UsefulnessError(
                    f"registration task_manifest.trials_per_task[{tid!r}] must be an integer >= 1"
                )
            out[tid] = count
        missing = sorted(task_ids - set(out))
        if missing:
            raise UsefulnessError(
                "registration task_manifest.trials_per_task is missing task(s): "
                + ", ".join(missing)
            )
        return out
    raise UsefulnessError(
        "registration is missing or has invalid field: task_manifest.trials_per_task"
    )


def _validate_limits(raw_limits: Mapping[str, Any]) -> dict[str, Any]:
    for required_key in (*VALID_LIMIT_NAMES, "enforced"):
        if required_key not in raw_limits:
            raise UsefulnessError(f"registration is missing required field: limits.{required_key}")

    wall_time_s = _require_positive_number(raw_limits, "wall_time_s", path="limits")

    tokens = raw_limits["tokens"]
    if tokens is not None and (
        isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1
    ):
        raise UsefulnessError("registration field limits.tokens must be a positive int or null")

    tool_calls = raw_limits["tool_calls"]
    if tool_calls is not None and (
        isinstance(tool_calls, bool) or not isinstance(tool_calls, int) or tool_calls < 1
    ):
        raise UsefulnessError("registration field limits.tool_calls must be a positive int or null")

    retries = raw_limits["retries"]
    if isinstance(retries, bool):
        raise UsefulnessError("registration field limits.retries must be an int >= 0 or object")
    if isinstance(retries, int):
        if retries < 0:
            raise UsefulnessError("registration field limits.retries must be >= 0")
        norm_retries: int | dict[str, Any] = retries
    elif isinstance(retries, Mapping) and retries:
        norm_retries = dict(retries)
    else:
        raise UsefulnessError("registration field limits.retries must be an int >= 0 or object")

    enforced = _require_str_list(raw_limits, "enforced", path="limits")
    unknown_enforced = sorted(set(enforced) - set(VALID_LIMIT_NAMES))
    if unknown_enforced:
        raise UsefulnessError(
            "registration field limits.enforced names unknown limit(s): "
            + ", ".join(unknown_enforced)
        )

    return {
        "tokens": tokens,
        "wall_time_s": wall_time_s,
        "tool_calls": tool_calls,
        "retries": norm_retries,
        "enforced": list(dict.fromkeys(enforced)),
    }


def _validate_models_section(raw_models: Mapping[str, Any]) -> dict[str, Any]:
    entries = raw_models.get("models")
    if not isinstance(entries, list) or not entries:
        raise UsefulnessError("registration is missing required non-empty list: models.models")
    normalized_models: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise UsefulnessError(f"registration field models.models[{idx}] must be an object")
        model_name = _require_non_empty_str(entry, "model", path=f"models.models[{idx}]")
        norm_entry: dict[str, Any] = {"model": model_name}
        if "reasoning_effort" in entry:
            re_val = entry["reasoning_effort"]
            if re_val is not None and (not isinstance(re_val, str) or not re_val.strip()):
                raise UsefulnessError(
                    f"registration field models.models[{idx}].reasoning_effort must be a string "
                    "or null"
                )
            norm_entry["reasoning_effort"] = re_val
        if "model_base_url" in entry:
            url_val = entry["model_base_url"]
            if url_val is not None and (not isinstance(url_val, str) or not url_val.strip()):
                raise UsefulnessError(
                    f"registration field models.models[{idx}].model_base_url must be a string "
                    "or null"
                )
            norm_entry["model_base_url"] = url_val
        normalized_models.append(norm_entry)

    harness_settings = _require_mapping(raw_models, "harness_settings", path="models")
    if not harness_settings:
        raise UsefulnessError("registration field models.harness_settings must be non-empty")
    pooling = _require_non_empty_text_or_mapping(raw_models, "pooling", path="models")

    return {
        "models": normalized_models,
        "harness_settings": dict(harness_settings),
        "pooling": pooling,
    }


def _validate_registration_document(
    spec: Mapping[str, Any],
    *,
    run_id: str,
    task_manifest: Mapping[str, Any] | None = None,
    task_manifest_sha256: str | None = None,
    memory_set_manifest: Mapping[str, Any] | None = None,
    memory_set_manifest_sha256: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize a full registration document against all spec sections."""
    if not isinstance(spec, Mapping):
        raise UsefulnessError("registration must be a JSON object")

    # 1. Primary endpoint
    pe = _require_mapping(spec, "primary_endpoint")
    primary_endpoint = {
        "endpoint": _require_non_empty_str(pe, "endpoint", path="primary_endpoint"),
        "practical_benefit_threshold": _require_positive_number(
            pe, "practical_benefit_threshold", path="primary_endpoint"
        ),
    }

    # 2. Task manifest section
    tm = _require_mapping(spec, "task_manifest")
    population = _require_non_empty_str(tm, "population", path="task_manifest")
    raw_rules = tm.get("inclusion_exclusion_rules")
    if isinstance(raw_rules, str) and raw_rules.strip():
        inclusion_exclusion_rules: list[str] = [raw_rules]
    elif isinstance(raw_rules, list | tuple) and raw_rules:
        inclusion_exclusion_rules = _require_str_list(
            tm, "inclusion_exclusion_rules", path="task_manifest"
        )
    else:
        raise UsefulnessError(
            "registration is missing required field: task_manifest.inclusion_exclusion_rules"
        )
    sampling_seed = _require_int(tm, "sampling_seed", path="task_manifest")
    sample_size = _require_int(tm, "sample_size", path="task_manifest", minimum=1)

    if task_manifest is not None:
        manifest_tasks = task_manifest.get("tasks")
        if not isinstance(manifest_tasks, Mapping) or not manifest_tasks:
            raise UsefulnessError("task-manifest.json has no tasks")
        task_ids = {str(k) for k in manifest_tasks}
        manifest_strata = {
            str(s)
            for entry in manifest_tasks.values()
            if isinstance(entry, Mapping)
            for s in (entry.get("strata") or ())
        }
        families_bound = dict(task_manifest.get("families") or {})
        split_bound = dict(task_manifest.get("split") or {})
        if not families_bound or not split_bound:
            raise UsefulnessError("task-manifest.json is missing families or split")
        if task_manifest_sha256 is None:
            raise UsefulnessError("task_manifest_sha256 is required when binding task_manifest")
        if "sha256" in tm and tm["sha256"] is not None and tm["sha256"] != task_manifest_sha256:
            raise UsefulnessError(
                f"registration task_manifest.sha256 ({tm['sha256']}) does not match "
                f"run's task-manifest.json ({task_manifest_sha256})"
            )
        if (
            "families" in tm
            and tm["families"] is not None
            and dict(tm["families"]) != families_bound
        ):
            raise UsefulnessError(
                "registration task_manifest.families does not match run's task-manifest.json"
            )
        if "split" in tm and tm["split"] is not None and dict(tm["split"]) != split_bound:
            raise UsefulnessError(
                "registration task_manifest.split does not match run's task-manifest.json"
            )
        bound_tm_sha = task_manifest_sha256
    else:
        bound_tm_sha = _require_non_empty_str(tm, "sha256", path="task_manifest")
        families_bound = dict(_require_mapping(tm, "families", path="task_manifest"))
        split_bound = dict(_require_mapping(tm, "split", path="task_manifest"))
        if not families_bound:
            raise UsefulnessError("registration field task_manifest.families must be non-empty")
        if not split_bound:
            raise UsefulnessError("registration field task_manifest.split must be non-empty")
        task_ids = set()
        for fam_entry in families_bound.values():
            if isinstance(fam_entry, Mapping):
                for tid in fam_entry.get("task_ids") or ():
                    task_ids.add(str(tid))
        manifest_strata = set()

    trials_per_task = _validate_trials_per_task(tm.get("trials_per_task"), task_ids)

    raw_strata_rules = _require_mapping(tm, "strata_rules", path="task_manifest")
    if not raw_strata_rules:
        raise UsefulnessError("registration field task_manifest.strata_rules must be non-empty")
    strata_rules: dict[str, str] = {}
    for s_name, s_rule in sorted(raw_strata_rules.items()):
        if not isinstance(s_name, str) or not s_name.strip():
            raise UsefulnessError("registration task_manifest.strata_rules has empty stratum name")
        if not isinstance(s_rule, str) or not s_rule.strip():
            raise UsefulnessError(
                f"registration task_manifest.strata_rules[{s_name!r}] must be a non-empty string"
            )
        strata_rules[s_name] = s_rule
    if EXACT_VALUE_STRATUM not in strata_rules:
        raise UsefulnessError(
            "registration task_manifest.strata_rules must include a rule for the "
            f"{EXACT_VALUE_STRATUM!r} stratum (tasks that hinge on an exact value)"
        )
    missing_strata = sorted(manifest_strata - set(strata_rules))
    if missing_strata:
        raise UsefulnessError(
            "registration task_manifest.strata_rules is missing rule(s) for manifest strata: "
            + ", ".join(missing_strata)
        )

    task_manifest_section = {
        "sha256": bound_tm_sha,
        "population": population,
        "inclusion_exclusion_rules": inclusion_exclusion_rules,
        "sampling_seed": sampling_seed,
        "sample_size": sample_size,
        "trials_per_task": trials_per_task,
        "families": families_bound,
        "split": split_bound,
        "strata_rules": strata_rules,
    }

    # 3. Memory set section
    raw_ms = spec.get("memory_set")
    if memory_set_manifest is not None:
        if memory_set_manifest_sha256 is None:
            raise UsefulnessError("memory_set_manifest_sha256 is required when binding memory_set")
        ms_sha = str(memory_set_manifest["memory_set_sha256"])
        ms_cutoff = str(memory_set_manifest["cutoff"])
        ms_last_day = str(memory_set_manifest["last_day"])
        if raw_ms is not None:
            if not isinstance(raw_ms, Mapping):
                raise UsefulnessError("registration field memory_set must be a JSON object")
            if raw_ms.get("sha256") is not None and raw_ms["sha256"] != ms_sha:
                raise UsefulnessError(
                    f"registration memory_set.sha256 ({raw_ms['sha256']}) does not match "
                    f"verified memory set ({ms_sha})"
                )
            if (
                raw_ms.get("manifest_sha256") is not None
                and raw_ms["manifest_sha256"] != memory_set_manifest_sha256
            ):
                raise UsefulnessError(
                    "registration memory_set.manifest_sha256 does not match "
                    "memory-set/manifest.json"
                )
            if raw_ms.get("cutoff") is not None and raw_ms["cutoff"] != ms_cutoff:
                raise UsefulnessError(
                    f"registration memory_set.cutoff ({raw_ms['cutoff']}) does not match "
                    f"memory set cutoff ({ms_cutoff})"
                )
            if raw_ms.get("last_day") is not None and raw_ms["last_day"] != ms_last_day:
                raise UsefulnessError(
                    f"registration memory_set.last_day ({raw_ms['last_day']}) does not match "
                    f"memory set last_day ({ms_last_day})"
                )
        memory_set_section = {
            "sha256": ms_sha,
            "manifest_sha256": memory_set_manifest_sha256,
            "cutoff": ms_cutoff,
            "last_day": ms_last_day,
        }
    else:
        ms = _require_mapping(spec, "memory_set")
        memory_set_section = {
            "sha256": _require_non_empty_str(ms, "sha256", path="memory_set"),
            "manifest_sha256": _require_non_empty_str(ms, "manifest_sha256", path="memory_set"),
            "cutoff": _require_non_empty_str(ms, "cutoff", path="memory_set"),
            "last_day": _require_non_empty_str(ms, "last_day", path="memory_set"),
        }

    # 4. Models section
    models_section = _validate_models_section(_require_mapping(spec, "models"))

    # 5. Memory tools wording & calibration settings
    raw_mt = _require_mapping(spec, "memory_tools")
    raw_mt_instructions = _require_non_empty_str(raw_mt, "instructions", path="memory_tools")
    raw_mt_descriptions = _require_mapping(raw_mt, "descriptions", path="memory_tools")
    memory_tools_section = build_memory_tool_wording(
        raw_mt_instructions,
        dict(raw_mt_descriptions),
    )
    if (
        "sha256" in raw_mt
        and raw_mt["sha256"] is not None
        and raw_mt["sha256"] != memory_tools_section["sha256"]
    ):
        raise UsefulnessError(
            "registration memory_tools.sha256 does not match its instructions and descriptions"
        )

    if "calibration_settings" not in spec or spec["calibration_settings"] is None:
        raise UsefulnessError("registration is missing required field: calibration_settings")
    if not isinstance(spec["calibration_settings"], Mapping):
        raise UsefulnessError("registration field calibration_settings must be a JSON object")
    calibration_settings = dict(spec["calibration_settings"])

    # 6. Environment manifest
    raw_env = _require_mapping(spec, "environment_manifest")
    env_section = build_environment_manifest(
        harness=_require_non_empty_str(raw_env, "harness", path="environment_manifest"),
        harness_version=_require_non_empty_str(
            raw_env, "harness_version", path="environment_manifest"
        ),
        repo_commit=_require_non_empty_str(raw_env, "repo_commit", path="environment_manifest"),
        allowed_base_tools=_require_str_list(
            raw_env, "allowed_base_tools", path="environment_manifest"
        ),
        memory_server_name=_require_non_empty_str(
            raw_env, "memory_server_name", path="environment_manifest"
        ),
        hardware_class=_require_non_empty_str(
            raw_env, "hardware_class", path="environment_manifest"
        ),
        config_overrides=_require_mapping(raw_env, "config_overrides", path="environment_manifest"),
    )
    if (
        "sha256" in raw_env
        and raw_env["sha256"] is not None
        and raw_env["sha256"] != env_section["sha256"]
    ):
        raise UsefulnessError(
            "registration environment_manifest.sha256 does not match its contents"
        )

    # 7. Limits
    limits_section = _validate_limits(_require_mapping(spec, "limits"))

    # 8. Invalidation rule
    invalidation_rule = _require_non_empty_text_or_mapping(spec, "invalidation_rule")

    # 9. Analysis
    raw_analysis = _require_mapping(spec, "analysis")
    analysis_section = {
        "bootstrap_seed": _require_int(raw_analysis, "bootstrap_seed", path="analysis"),
        "bootstrap_resamples": _require_int(
            raw_analysis, "bootstrap_resamples", path="analysis", minimum=1
        ),
        "secondary_measures": _require_non_empty_text_or_mapping(
            raw_analysis, "secondary_measures", path="analysis"
        ),
        "exploratory_slices": _require_str_list(
            raw_analysis, "exploratory_slices", path="analysis", allow_empty=True
        ),
    }

    # 10. Stopping rule
    raw_stop = _require_mapping(spec, "stopping_rule")
    stopping_rule = {
        "stop_at": _require_non_empty_text_or_mapping(raw_stop, "stop_at", path="stopping_rule"),
        "safety_stop_harmful_uses": _require_int(
            raw_stop, "safety_stop_harmful_uses", path="stopping_rule", minimum=1
        ),
        "sequential_error_control": _require_non_empty_str(
            raw_stop, "sequential_error_control", path="stopping_rule"
        ),
    }

    existing_created = spec.get("created_at")
    effective_created = (
        created_at
        if created_at is not None
        else (
            existing_created
            if isinstance(existing_created, str) and existing_created.strip()
            else utc_now_iso()
        )
    )

    return {
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "created_at": effective_created,
        "run_id": run_id,
        "primary_endpoint": primary_endpoint,
        "task_manifest": task_manifest_section,
        "memory_set": memory_set_section,
        "models": models_section,
        "memory_tools": memory_tools_section,
        "calibration_settings": calibration_settings,
        "environment_manifest": env_section,
        "limits": limits_section,
        "invalidation_rule": invalidation_rule,
        "analysis": analysis_section,
        "stopping_rule": stopping_rule,
    }


def freeze_registration(
    experiment_home: Config | Path,
    run_id: str,
    spec: Mapping[str, Any],
    *,
    created_at: str | None = None,
) -> RegistrationResult:
    """Validate and write ``registration.json`` once at the freeze.

    Requires ``task-manifest.json`` in the run directory and a verified memory
    set under ``<experiment_home>/memory-set``. Refuses if any required item
    under "The registration" is missing, if any hash/field disagrees with the
    run's task manifest or memory set, or if ``registration.json`` already exists.
    """
    store = UsefulnessRunStore(experiment_home, run_id)
    task_manifest = store.read_task_manifest()
    task_manifest_sha = store.task_manifest_sha256()

    ms_manifest = verify_memory_set(experiment_home)
    ms_manifest_sha = sha256_file(memory_set_manifest_path(experiment_home))

    doc = _validate_registration_document(
        spec,
        run_id=run_id,
        task_manifest=task_manifest,
        task_manifest_sha256=task_manifest_sha,
        memory_set_manifest=ms_manifest,
        memory_set_manifest_sha256=ms_manifest_sha,
        created_at=created_at,
    )
    reg_sha = store.write_registration(doc)
    return RegistrationResult(
        registration=doc,
        registration_path=store.registration_path,
        registration_sha256=reg_sha,
    )


def load_registration(experiment_home: Config | Path, run_id: str) -> RegistrationResult:
    """Load and internally validate ``registration.json`` for ``run_id``."""
    store = UsefulnessRunStore(experiment_home, run_id)
    raw = store.read_registration()
    version = raw.get("schema_version")
    if isinstance(version, bool) or version != REGISTRATION_SCHEMA_VERSION:
        raise UsefulnessError(
            f"unsupported registration schema_version {version!r}; "
            f"expected {REGISTRATION_SCHEMA_VERSION}"
        )
    if raw.get("run_id") != run_id:
        raise UsefulnessError(
            f"registration run_id {raw.get('run_id')!r} does not match requested run {run_id!r}"
        )
    validated = _validate_registration_document(raw, run_id=run_id)
    return RegistrationResult(
        registration=validated,
        registration_path=store.registration_path,
        registration_sha256=store.registration_sha256(),
    )


def verify_registration_for_trial(
    experiment_home: Config | Path,
    run_id: str,
    *,
    harness: str,
    harness_version: str | None,
    repo_commit: str,
    hardware_class: str,
    allowed_base_tools: Sequence[str],
    memory_server_name: str,
    config_overrides: Mapping[str, Any],
    model: str,
    reasoning_effort: str | None,
    wall_time_limit_s: float,
    arm: str,
    source_home: Path | None = None,
    wording_override: Mapping[str, Any] | None = None,
    wording_file: Path | None = None,
) -> VerifiedRegistration:
    """Verify the live run state and trial settings against the frozen registration.

    Refuses if any of the following no longer matches ``registration.json``:
    - the run's ``task-manifest.json``
    - the experiment home's memory set (and any explicit ``source_home``)
    - the environment manifest (harness, version, repo commit, tools, hardware, overrides)
    - the registered model/reasoning_effort or wall-time limit
    - the memory tools' instructions and descriptions wording
    """
    store = UsefulnessRunStore(experiment_home, run_id)
    loaded = load_registration(experiment_home, run_id)
    reg = loaded.registration

    # 1. Task manifest match
    actual_tm_sha = store.task_manifest_sha256()
    expected_tm_sha = reg["task_manifest"]["sha256"]
    if actual_tm_sha != expected_tm_sha:
        raise UsefulnessError(
            f"task manifest no longer matches registration: actual SHA-256 {actual_tm_sha} "
            f"!= registered {expected_tm_sha}"
        )

    # 2. Memory set match (re-hashes memory.duckdb and blobs/)
    ms_manifest = verify_memory_set(experiment_home)
    actual_ms_manifest_sha = sha256_file(memory_set_manifest_path(experiment_home))
    reg_ms = reg["memory_set"]
    if ms_manifest["memory_set_sha256"] != reg_ms["sha256"]:
        raise UsefulnessError(
            f"memory set no longer matches registration: actual SHA-256 "
            f"{ms_manifest['memory_set_sha256']} != registered {reg_ms['sha256']}"
        )
    if actual_ms_manifest_sha != reg_ms["manifest_sha256"]:
        raise UsefulnessError(
            f"memory set manifest no longer matches registration: actual SHA-256 "
            f"{actual_ms_manifest_sha} != registered {reg_ms['manifest_sha256']}"
        )
    if ms_manifest["cutoff"] != reg_ms["cutoff"] or ms_manifest["last_day"] != reg_ms["last_day"]:
        raise UsefulnessError("memory set cutoff/last_day no longer matches registration")
    if arm == "memory" and source_home is not None:
        resolved_src = ensure_experiment_home(source_home)
        expected_ms_dir = store.memory_set_dir.resolve()
        if resolved_src != expected_ms_dir:
            raise UsefulnessError(
                f"trial source_home ({resolved_src}) does not match the registered memory set "
                f"({expected_ms_dir})"
            )

    # 3. Environment manifest match
    reg_env = reg["environment_manifest"]
    if harness != reg_env["harness"]:
        raise UsefulnessError(
            f"environment harness {harness!r} does not match registration {reg_env['harness']!r}"
        )
    if harness_version is None or harness_version != reg_env["harness_version"]:
        raise UsefulnessError(
            f"environment harness_version {harness_version!r} does not match registration "
            f"{reg_env['harness_version']!r}"
        )
    if repo_commit != reg_env["repo_commit"]:
        raise UsefulnessError(
            f"environment repo_commit {repo_commit!r} does not match registration "
            f"{reg_env['repo_commit']!r}"
        )
    if list(allowed_base_tools) != reg_env["allowed_base_tools"]:
        raise UsefulnessError(
            f"environment allowed_base_tools {list(allowed_base_tools)!r} does not match "
            f"registration {reg_env['allowed_base_tools']!r}"
        )
    if memory_server_name != reg_env["memory_server_name"]:
        raise UsefulnessError(
            f"environment memory_server_name {memory_server_name!r} does not match registration "
            f"{reg_env['memory_server_name']!r}"
        )
    if hardware_class != reg_env["hardware_class"]:
        raise UsefulnessError(
            f"environment hardware_class {hardware_class!r} does not match registration "
            f"{reg_env['hardware_class']!r}"
        )
    if dict(config_overrides) != reg_env["config_overrides"]:
        raise UsefulnessError(
            f"environment config_overrides {dict(config_overrides)!r} does not match "
            f"registration {reg_env['config_overrides']!r}"
        )

    live_env = build_environment_manifest(
        harness=harness,
        harness_version=harness_version,
        repo_commit=repo_commit,
        allowed_base_tools=allowed_base_tools,
        memory_server_name=memory_server_name,
        hardware_class=hardware_class,
        config_overrides=config_overrides,
    )
    if live_env["sha256"] != reg_env["sha256"]:
        raise UsefulnessError(
            f"environment manifest SHA-256 ({live_env['sha256']}) does not match registration "
            f"({reg_env['sha256']})"
        )

    # Model and wall-time limit match
    matching_model = [m for m in reg["models"]["models"] if m["model"] == model]
    if not matching_model:
        raise UsefulnessError(f"model {model!r} is not listed in registration models")
    if not any(
        ("reasoning_effort" not in m) or (m["reasoning_effort"] == reasoning_effort)
        for m in matching_model
    ):
        raise UsefulnessError(
            f"reasoning_effort {reasoning_effort!r} for model {model!r} does not match registration"
        )
    if float(wall_time_limit_s) != float(reg["limits"]["wall_time_s"]):
        raise UsefulnessError(
            f"wall_time_limit_s ({wall_time_limit_s}) does not match registration "
            f"({reg['limits']['wall_time_s']})"
        )

    # 4. Memory tools wording match
    reg_mt = reg["memory_tools"]
    reg_instructions = str(reg_mt["instructions"])
    reg_descriptions = dict(reg_mt["descriptions"])
    reg_mt_sha = str(reg_mt["sha256"])

    if wording_override is not None and wording_file is not None:
        raise UsefulnessError("specify at most one of wording_override or wording_file")

    if wording_file is not None:
        try:
            file_instr, file_descs = load_tool_wording_file(wording_file)
        except ConfigError as exc:
            raise UsefulnessError(str(exc)) from None
        actual_sha = memory_tool_wording_sha256(file_instr, file_descs)
        if actual_sha != reg_mt_sha:
            raise UsefulnessError(
                f"memory tools wording in {wording_file} no longer matches registration "
                f"(actual SHA-256 {actual_sha} != registered {reg_mt_sha})"
            )
    elif wording_override is not None:
        try:
            ov_instr, ov_descs = validate_tool_wording(wording_override)
        except ConfigError as exc:
            raise UsefulnessError(str(exc)) from None
        actual_sha = memory_tool_wording_sha256(ov_instr, ov_descs)
        if actual_sha != reg_mt_sha:
            raise UsefulnessError(
                f"memory tools wording no longer matches registration "
                f"(actual SHA-256 {actual_sha} != registered {reg_mt_sha})"
            )

    return VerifiedRegistration(
        registration=reg,
        registration_sha256=loaded.registration_sha256,
        memory_tools_instructions=reg_instructions,
        memory_tools_descriptions=reg_descriptions,
        memory_tools_sha256=reg_mt_sha,
        memory_set_dir=store.memory_set_dir.resolve(),
    )


def verify_registration_artifacts(
    experiment_home: Config | Path,
    run_id: str,
    *,
    wording_file: Path | None = None,
) -> RegistrationResult:
    """Verify ``registration.json`` against the live task manifest, memory set, and wording."""
    store = UsefulnessRunStore(experiment_home, run_id)
    loaded = load_registration(experiment_home, run_id)
    reg = loaded.registration

    actual_tm_sha = store.task_manifest_sha256()
    if actual_tm_sha != reg["task_manifest"]["sha256"]:
        raise UsefulnessError(
            f"task manifest no longer matches registration: actual SHA-256 {actual_tm_sha} "
            f"!= registered {reg['task_manifest']['sha256']}"
        )

    ms_manifest = verify_memory_set(experiment_home)
    actual_ms_manifest_sha = sha256_file(memory_set_manifest_path(experiment_home))
    if ms_manifest["memory_set_sha256"] != reg["memory_set"]["sha256"]:
        raise UsefulnessError(
            f"memory set no longer matches registration: actual SHA-256 "
            f"{ms_manifest['memory_set_sha256']} != registered {reg['memory_set']['sha256']}"
        )
    if actual_ms_manifest_sha != reg["memory_set"]["manifest_sha256"]:
        raise UsefulnessError(
            f"memory set manifest no longer matches registration: actual SHA-256 "
            f"{actual_ms_manifest_sha} != registered {reg['memory_set']['manifest_sha256']}"
        )

    if wording_file is not None:
        try:
            file_instr, file_descs = load_tool_wording_file(wording_file)
        except ConfigError as exc:
            raise UsefulnessError(str(exc)) from None
        actual_sha = memory_tool_wording_sha256(file_instr, file_descs)
        if actual_sha != reg["memory_tools"]["sha256"]:
            raise UsefulnessError(
                f"memory tools wording in {wording_file} no longer matches registration "
                f"(actual SHA-256 {actual_sha} != registered {reg['memory_tools']['sha256']})"
            )

    return loaded


def append_deviation(
    experiment_home: Config | Path,
    run_id: str,
    reason: str,
    *,
    date: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a dated deviation to ``deviations.jsonl`` without rewriting earlier rows."""
    store = UsefulnessRunStore(experiment_home, run_id)
    return store.append_deviation(reason, date=date, metadata=metadata)


def list_deviations(experiment_home: Config | Path, run_id: str) -> list[dict[str, Any]]:
    """Return every recorded deviation for ``run_id`` in append order."""
    store = UsefulnessRunStore(experiment_home, run_id)
    return store.deviations()
