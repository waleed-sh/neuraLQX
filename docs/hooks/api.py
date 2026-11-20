"""Generate API pages from public source namespaces without importing neuraLQX."""

from __future__ import annotations

import ast
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File
from mkdocs.structure.files import Files


def public_namespaces(package: Path) -> dict[str, Path]:
    """Include public packages and modules that explicitly declare ``__all__``.

    Re-exported classes/functions are rendered at their package namespace by
    mkdocstrings. Private implementation paths never get separate API pages.
    """
    namespaces = {}
    for source in sorted(package.rglob("*.py")):
        relative = source.relative_to(package.parent).with_suffix("")
        parts = relative.parts
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        if any(part.startswith("_") for part in parts):
            continue
        if not is_package:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            declares_exports = any(
                (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "__all__"
                        for target in node.targets
                    )
                )
                or (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "__all__"
                )
                for node in tree.body
            )
            if not declares_exports:
                continue
        namespaces[".".join(parts)] = source
    return namespaces


def namespace_nav(pages: dict[str, str], root: str) -> list[dict]:
    """Nest namespace pages, keeping the package landing page first."""
    tree: dict = {}
    for namespace, uri in pages.items():
        if namespace == root:
            continue
        node = tree
        for part in namespace.removeprefix(root + ".").split("."):
            node = node.setdefault(part, {})
        node["_page"] = uri

    def entries(node: dict) -> list[dict]:
        children = []
        if "_page" in node:
            children.append({"Overview": node["_page"]})
        for name, child in sorted(node.items()):
            if name == "_page":
                continue
            children.append({name: child["_page"] if set(child) == {"_page"} else entries(child)})
        return children

    return ([{root: pages[root]}] if root in pages else []) + entries(tree)


def replace_api_nav(nav: list, overview: str, generated: list[dict]) -> bool:
    """Replace a generated submenu on every build, including live reloads."""
    for item in nav:
        for child in item.values():
            if not isinstance(child, list):
                continue
            if child and child[0] == {"Overview": overview}:
                child[:] = [{"Overview": overview}, *generated]
                return True
            if replace_api_nav(child, overview, generated):
                return True
    return False


def on_files(files: Files, *, config: MkDocsConfig) -> Files:
    root = Path(config.config_file_path).resolve().parent
    groups: dict[str, dict[str, str]] = {"public": {}, "experimental": {}}
    for namespace, source in public_namespaces(root / "neuralqx").items():
        group = (
            "experimental"
            if namespace == "neuralqx.experimental"
            or namespace.startswith("neuralqx.experimental.")
            else "public"
        )
        uri = f"api/{group}/{namespace.replace('.', '/')}/index.md"
        content = f"---\ntitle: {namespace}\n---\n\n# {namespace}\n\n::: {namespace}\n"
        generated = File.generated(config, uri, content=content)
        generated.edit_uri = "../" + source.relative_to(root).as_posix()
        files.append(generated)
        groups[group][namespace] = uri

    for group, pages in groups.items():
        package = "neuralqx.experimental" if group == "experimental" else "neuralqx"
        overview = f"api/{group}/index.md"
        if not replace_api_nav(config.nav, overview, namespace_nav(pages, package)):
            raise PluginError(f"Missing API navigation anchor: {overview}")
    return files
