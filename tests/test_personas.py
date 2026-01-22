"""Tests for persona generation."""

from salesbench.core.types import LeadTemperature
from salesbench.envs.sales_mvp.personas import (
    HiddenState,
    Persona,
    PersonaGenerator,
)


class TestPersonaGenerator:
    """Tests for PersonaGenerator."""

    def test_seeded_generation_is_deterministic(self, default_seed: int):
        """Test that same seed produces identical personas."""
        gen1 = PersonaGenerator(seed=default_seed)
        gen2 = PersonaGenerator(seed=default_seed)

        personas1 = gen1.generate_batch(100)
        personas2 = gen2.generate_batch(100)

        for p1, p2 in zip(personas1, personas2):
            assert p1.name == p2.name
            assert p1.age == p2.age
            assert p1.job == p2.job
            assert p1.annual_income == p2.annual_income
            assert p1.temperature == p2.temperature
            assert p1.hidden.trust == p2.hidden.trust

    def test_different_seeds_produce_different_personas(self):
        """Test that different seeds produce different personas."""
        gen1 = PersonaGenerator(seed=42)
        gen2 = PersonaGenerator(seed=123)

        p1 = gen1.generate_one()
        p2 = gen2.generate_one()

        # With different seeds, at least some attributes should differ
        differences = 0
        if p1.name != p2.name:
            differences += 1
        if p1.age != p2.age:
            differences += 1
        if p1.job != p2.job:
            differences += 1

        assert differences > 0, "Different seeds should produce different personas"

    def test_reset_restores_initial_state(self, persona_generator: PersonaGenerator):
        """Test that reset() restores the generator to initial state."""
        first_batch = persona_generator.generate_batch(10)
        persona_generator.reset()
        second_batch = persona_generator.generate_batch(10)

        for p1, p2 in zip(first_batch, second_batch):
            assert p1.name == p2.name

    def test_all_archetypes_represented(self, default_seed: int):
        """Test that all 10 archetypes appear in large batch."""
        gen = PersonaGenerator(seed=default_seed)
        personas = gen.generate_batch(1000)

        # Check that we see variety in jobs (proxy for archetypes)
        jobs = set(p.job for p in personas)

        # We expect to see jobs from multiple archetypes
        assert len(jobs) >= 20, "Should see jobs from multiple archetypes"

    def test_hidden_state_bounds(self, sample_personas: list[Persona]):
        """Test that hidden state values are within valid bounds."""
        for persona in sample_personas:
            hidden = persona.hidden

            assert 0.0 <= hidden.trust <= 1.0, f"Trust {hidden.trust} out of bounds"
            assert 0.0 <= hidden.interest <= 1.0, f"Interest {hidden.interest} out of bounds"
            assert 0.0 <= hidden.patience <= 1.0, f"Patience {hidden.patience} out of bounds"
            assert (
                0.01 <= hidden.close_threshold <= 0.15
            ), f"Close threshold {hidden.close_threshold} out of bounds"

    def test_temperature_distribution(self, default_seed: int):
        """Test that temperature distribution roughly matches config."""
        gen = PersonaGenerator(seed=default_seed)
        personas = gen.generate_batch(1000)

        temp_counts = {t: 0 for t in LeadTemperature}
        for p in personas:
            temp_counts[p.temperature] += 1

        # Check rough distribution (with some tolerance)
        # Expected: HOT 3%, WARM 12%, LUKEWARM 35%, COLD 40%, HOSTILE 10%
        assert temp_counts[LeadTemperature.HOT] < 100, "Too many HOT leads"
        assert temp_counts[LeadTemperature.HOSTILE] < 200, "Too many HOSTILE leads"
        assert temp_counts[LeadTemperature.COLD] > 200, "Too few COLD leads"

    def test_age_within_archetype_range(self, sample_personas: list[Persona]):
        """Test that generated ages are within valid range."""
        for persona in sample_personas:
            assert 18 <= persona.age <= 80, f"Age {persona.age} out of valid range"

    def test_income_is_positive(self, sample_personas: list[Persona]):
        """Test that income is always positive."""
        for persona in sample_personas:
            assert persona.annual_income > 0

    def test_household_size_is_valid(self, sample_personas: list[Persona]):
        """Test that household size is at least 1."""
        for persona in sample_personas:
            assert persona.household_size >= 1
            expected_size = 1 + (1 if persona.has_spouse else 0) + persona.num_dependents
            assert persona.household_size == expected_size


class TestPersona:
    """Tests for Persona dataclass."""

    def test_to_public_dict_excludes_hidden(self, sample_personas: list[Persona]):
        """Test that public dict doesn't include hidden state."""
        for persona in sample_personas:
            public = persona.to_public_dict()
            assert "hidden" not in public
            assert "trust" not in public

    def test_to_full_dict_includes_hidden(self, sample_personas: list[Persona]):
        """Test that full dict includes hidden state."""
        for persona in sample_personas:
            full = persona.to_full_dict()
            assert "hidden" in full
            assert "trust" in full["hidden"]

    def test_monthly_income_calculation(self, sample_personas: list[Persona]):
        """Test monthly income calculation."""
        for persona in sample_personas:
            expected = persona.annual_income / 12
            assert persona.monthly_income() == expected

    def test_affordable_premium_calculation(self, sample_personas: list[Persona]):
        """Test affordable premium calculation."""
        for persona in sample_personas:
            monthly = persona.monthly_income()
            threshold = persona.hidden.close_threshold
            expected = monthly * threshold
            assert persona.affordable_premium() == expected


class TestHiddenState:
    """Tests for HiddenState dataclass."""

    def test_to_dict(self):
        """Test HiddenState serialization."""
        hidden = HiddenState(
            trust=0.5,
            interest=0.6,
            patience=0.7,
            close_threshold=0.05,
        )

        d = hidden.to_dict()

        assert d["trust"] == 0.5
        assert d["interest"] == 0.6
        assert d["patience"] == 0.7
        assert d["close_threshold"] == 0.05
