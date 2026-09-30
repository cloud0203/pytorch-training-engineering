"""Check skill packaging without importing PyTorch or executing training."""

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml


SKILL_NAME = "pytorch-training-engineering"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_mapping(path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected YAML mapping: {path}")
    return value


def positive_integer(value):
    return type(value) is int and value > 0


def check_example_config(config):
    """Validate this repository's step-budget example, not arbitrary trainers."""
    require(config.get("schema_version") == 2, "Example schema must be version 2")
    training = config["training"]
    budget = training["budget"]
    require(budget["unit"] == "optimizer_step", "Example budget must use updates")
    limit = budget["limit"]
    require(positive_integer(limit), "Budget limit must be a positive integer")
    require("epochs" not in training, "Example mixes epoch and update budgets")
    for field in (
        "accumulation_steps",
        "target_effective_batch_size",
        "max_consecutive_skipped_updates",
    ):
        require(positive_integer(training[field]), f"Invalid training.{field}")
    require(training["accumulation_tail"] == "flush", "Example must flush tails")
    batch = config["data"]["batch_size_per_device"]
    require(positive_integer(batch), "Invalid per-device batch")
    require(
        batch * training["accumulation_steps"]
        == training["target_effective_batch_size"],
        "Example's default single-process effective batch does not match target",
    )

    scheduler = config["scheduler"]
    require(
        scheduler["interval"] == "optimizer_step",
        "Example scheduler must use successful updates",
    )
    require(
        positive_integer(scheduler["total_steps"])
        and scheduler["total_steps"] == limit,
        "Fresh example scheduler horizon must match the update budget",
    )
    warmup = scheduler["warmup_steps"]
    require(
        type(warmup) is int and 0 <= warmup < limit,
        "Warmup must be an integer below the total update budget",
    )
    require(
        not {"total_epochs", "warmup_epochs"}.intersection(scheduler),
        "Example scheduler contains legacy epoch fields",
    )
    for name in ("validation", "logging", "checkpoint"):
        section = config[name]
        require(section["interval"] == "optimizer_step", f"Wrong {name} unit")
        require(
            positive_integer(section["every_n_steps"]),
            f"Invalid {name} frequency",
        )
        require("every_n_epochs" not in section, f"Legacy epoch field in {name}")
    for name in ("validation", "checkpoint"):
        require(config[name]["at_end"] is True, f"Missing final {name} event")
    require(
        config["checkpoint"]["resume_granularity"] == "optimizer_step",
        "Step checkpoints require an explicit compatible recovery contract",
    )


def check_markdown(path, root):
    content = path.read_text(encoding="utf-8")
    require("[TODO:" not in content, f"Unfinished scaffold: {path}")
    fence = None
    prose = []
    for line in content.splitlines():
        require(line == line.rstrip(), f"Trailing whitespace: {path}")
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if match:
            marker, suffix = match.groups()
            if fence is None:
                fence = marker
            elif (
                marker[0] == fence[0]
                and len(marker) >= len(fence)
                and not suffix.strip()
            ):
                fence = None
            continue
        if fence is None:
            prose.append(line)
    require(fence is None, f"Unclosed code fence: {path}")
    for target in re.findall(r"\]\(([^)]+)\)", "\n".join(prose)):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = (path.parent / unquote(parsed.path)).resolve()
        require(destination.is_relative_to(root), f"Nonportable link: {target}")
        require(destination.exists(), f"Broken link in {path}: {target}")


def validate(root):
    skill = root / "skills" / SKILL_NAME
    required = (
        "SKILL.md",
        "agents/openai.yaml",
        "references/project-layout.md",
        "references/training-contract.md",
        "references/training-budget.md",
        "references/acceptance.md",
        "assets/config.example.yaml",
        "assets/standalone-prompt.md",
    )
    for relative in required:
        require((skill / relative).is_file(), f"Missing skill file: {relative}")

    content = (skill / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---(?:\n|$)", content, re.DOTALL)
    require(match is not None, "Missing SKILL.md frontmatter")
    frontmatter = yaml.safe_load(match.group(1))
    require(isinstance(frontmatter, dict), "Frontmatter must be a mapping")
    require(frontmatter.get("name") == SKILL_NAME, "Skill name mismatch")
    description = frontmatter.get("description")
    require(
        isinstance(description, str) and 0 < len(description) <= 1024,
        "Description must be a nonempty string of at most 1024 characters",
    )

    metadata = read_mapping(skill / "agents/openai.yaml")
    policy = metadata.get("policy")
    require(isinstance(policy, dict), "Invocation policy must be a mapping")
    require(
        policy.get("allow_implicit_invocation") is False,
        "Explicit-only invocation must remain enabled",
    )
    interface = metadata.get("interface", {})
    require(isinstance(interface, dict), "Interface metadata must be a mapping")
    for key in ("display_name", "short_description", "default_prompt"):
        require(isinstance(interface.get(key), str), f"Missing interface {key}")
        require(interface[key].strip(), f"Empty interface {key}")
    require(
        f"${SKILL_NAME}" in interface["default_prompt"],
        "Default prompt must explicitly invoke this skill",
    )

    documents = [root / "README.md", root / "CONTRIBUTING.md"]
    documents.extend((root / "docs").rglob("*.md"))
    documents.extend(skill.rglob("*.md"))
    for path in documents:
        check_markdown(path, root)
    for path in skill.rglob("*.yaml"):
        read_mapping(path)
    check_example_config(read_mapping(skill / "assets/config.example.yaml"))
    read_mapping(root / ".github/workflows/validate.yml")
    print(
        f"PASS: {len(documents)} documents, YAML, metadata, local links "
        "and example budget consistency"
    )
    print("No training, GPU calls or network requests were performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    try:
        validate(args.root.resolve())
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
