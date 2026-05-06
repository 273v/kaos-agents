"""Unit tests for the recipe library."""

from __future__ import annotations

from kaos_agents.memory.session import SessionMemory
from kaos_agents.recipes import (
    format_recipe_for_memory,
    load_builtin_recipes,
    load_recipe,
    recipe_names,
)
from kaos_agents.types.memory import MemoryType


class TestRecipeLoading:
    def test_load_builtin_recipes(self):
        recipes = load_builtin_recipes()
        assert len(recipes) >= 5, f"Expected at least 5 recipes, got {len(recipes)}"

    def test_recipe_names(self):
        names = recipe_names()
        assert "contract-extraction" in names
        assert "corpus-qa" in names
        assert "federal-register-research" in names
        assert "edgar-research" in names
        assert "summarization" in names

    def test_load_specific_recipe(self):
        recipe = load_recipe("contract-extraction")
        assert recipe is not None
        assert recipe["name"] == "contract-extraction"
        assert "tools" in recipe
        assert "steps" in recipe
        assert len(recipe["steps"]) >= 3

    def test_load_nonexistent_recipe(self):
        recipe = load_recipe("nonexistent-recipe")
        assert recipe is None

    def test_recipe_has_required_fields(self):
        for recipe in load_builtin_recipes():
            assert "name" in recipe, f"Recipe missing name: {recipe}"
            assert "description" in recipe, f"Recipe {recipe['name']} missing description"
            assert "steps" in recipe, f"Recipe {recipe['name']} missing steps"


class TestRecipeFormatting:
    def test_format_for_memory(self):
        recipe = load_recipe("contract-extraction")
        assert recipe is not None
        text = format_recipe_for_memory(recipe)
        assert "Recipe: contract-extraction" in text
        assert "When:" in text
        assert "Tools:" in text
        assert "Steps:" in text
        assert "1." in text

    def test_format_all_recipes(self):
        for recipe in load_builtin_recipes():
            text = format_recipe_for_memory(recipe)
            assert len(text) > 20, f"Recipe {recipe['name']} formatted too short"


class TestRecipeMemoryIntegration:
    def test_load_recipes_into_plan_examples(self):
        """Recipes should load into PLAN_EXAMPLES via load_explicit."""
        memory = SessionMemory("test-recipes")
        recipes = load_builtin_recipes()
        formatted = [format_recipe_for_memory(r) for r in recipes]
        n = memory.load_explicit(MemoryType.PLAN_EXAMPLES, formatted)
        assert n == len(recipes)
        items = memory.get(MemoryType.PLAN_EXAMPLES)
        assert len(items) == len(recipes)
