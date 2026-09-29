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
    read_mapping(root / ".github/workflows/validate.yml")
    print(f"PASS: {len(documents)} documents, YAML, metadata and local file links")
    print("No training, GPU calls or network requests were performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    try:
        validate(args.root.resolve())
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
