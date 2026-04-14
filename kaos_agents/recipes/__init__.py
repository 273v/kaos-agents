"""Built-in recipe library for kaos-agents.

Recipes are structured playbooks that guide agent planning. They are loaded
into the PLAN_EXAMPLES memory section at session creation time, giving the
planning LLM concrete examples of how to decompose goals into steps.

Each recipe is a JSON file with:
- name: human-readable recipe name
- description: when to use this recipe
- tools: list of tool names the recipe uses
- steps: ordered list of step descriptions

Use :func:`load_builtin_recipes` to get all built-in recipes, or
:func:`load_recipe` to load a specific one by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaos_core.logging import get_logger

logger = get_logger(__name__)


def load_builtin_recipes() -> list[dict[str, Any]]:
    """Load all built-in recipe JSON files.

    Returns:
        List of parsed recipe dicts, each with name, description, tools, steps.
    """
    recipes = []
    recipe_dir = Path(__file__).parent
    for path in sorted(recipe_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "name" in data:
                recipes.append(data)
        except Exception as exc:
            logger.warning("Failed to load recipe %s: %s", path.name, exc)
    return recipes


def load_recipe(name: str) -> dict[str, Any] | None:
    """Load a specific recipe by name.

    Args:
        name: Recipe name (e.g., "contract-extraction", "corpus-qa").

    Returns:
        The recipe dict, or None if not found.
    """
    for recipe in load_builtin_recipes():
        if recipe.get("name") == name:
            return recipe
    return None


def recipe_names() -> list[str]:
    """List available recipe names."""
    return [r["name"] for r in load_builtin_recipes()]


def format_recipe_for_memory(recipe: dict[str, Any]) -> str:
    """Format a recipe as a text string for loading into PLAN_EXAMPLES.

    Args:
        recipe: Parsed recipe dict.

    Returns:
        Formatted string suitable for SessionMemory.load_explicit().
    """
    parts = [f"Recipe: {recipe.get('name', 'unnamed')}"]
    if recipe.get("description"):
        parts.append(f"When: {recipe['description']}")
    if recipe.get("tools"):
        parts.append(f"Tools: {', '.join(recipe['tools'])}")
    if recipe.get("steps"):
        parts.append("Steps:")
        for i, step in enumerate(recipe["steps"], 1):
            if isinstance(step, dict):
                parts.append(f"  {i}. {step.get('description', step)}")
            else:
                parts.append(f"  {i}. {step}")
    return "\n".join(parts)
