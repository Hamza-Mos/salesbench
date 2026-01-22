"""Inspection commands for CLI.

Includes: seed-leads, inspect-products, quote, list-models, list-domains
"""

import argparse
import json

from salesbench.cli.commands import register_command
from salesbench.cli.formatting import OutputFormatter


@register_command("seed-leads")
def seed_leads_command(args: argparse.Namespace) -> int:
    """Generate and display personas from a seed."""
    from salesbench.envs.sales_mvp.personas import PersonaGenerator

    generator = PersonaGenerator(seed=args.seed)
    leads = generator.generate_batch(args.n)
    formatter = OutputFormatter(args.format)

    if formatter.is_json:
        output = {
            "seed": args.seed,
            "count": len(leads),
            "leads": [lead.to_public_dict() for lead in leads],
        }
        if args.full:
            output["leads"] = [lead.to_full_dict() for lead in leads]
        formatter.print_json(output)
    else:
        print(f"Generated {len(leads)} leads with seed {args.seed}")
        print("-" * 80)

        # Count by temperature
        temps = {}
        for lead in leads:
            t = lead.temperature.value
            temps[t] = temps.get(t, 0) + 1

        print("\nTemperature distribution:")
        for temp, count in sorted(temps.items()):
            pct = count / len(leads) * 100
            print(f"  {temp:10s}: {count:3d} ({pct:5.1f}%)")

        print("\nSample leads:")
        print(f"{'Name':<25} {'Age':>4} {'Temperature':<10} {'Income':>10} {'Job':<25}")
        print("-" * 80)
        for lead in leads[:10]:
            print(
                f"{lead.name:<25} {lead.age:>4} {lead.temperature.value:<10} "
                f"${lead.annual_income:>9,} {lead.job:<25}"
            )

        if args.full:
            print("\nHidden state (first 5 leads):")
            for lead in leads[:5]:
                print(f"  {lead.name}:")
                print(f"    trust={lead.hidden.trust:.2f}, interest={lead.hidden.interest:.2f}")
                print(f"    patience={lead.hidden.patience:.2f}")
                print(f"    close_threshold={lead.hidden.close_threshold:.2%}")

    return 0


@register_command("inspect-products")
def inspect_products_command(args: argparse.Namespace) -> int:
    """Display product catalog."""
    from salesbench.envs.sales_mvp.products import ProductCatalog

    catalog = ProductCatalog()
    products = catalog.list_products()
    formatter = OutputFormatter(args.format)

    if formatter.is_json:
        formatter.print_json(products)
    else:
        print("Insurance Product Catalog")
        print("=" * 80)
        for product in products:
            print(f"\n{product['name']} ({product['plan_id']})")
            print("-" * 40)
            print(f"  {product['description'][:70]}...")
            print(f"  Coverage: ${product['min_coverage']:,} - ${product['max_coverage']:,}")
            print(f"  Ages: {product['min_age']} - {product['max_age']}")
            print("  Features:")
            for feature in product["features"][:3]:
                print(f"    - {feature}")

    return 0


@register_command("quote")
def quote_command(args: argparse.Namespace) -> int:
    """Get a premium quote."""
    from salesbench.core.types import PlanType, RiskClass
    from salesbench.envs.sales_mvp.products import ProductCatalog

    catalog = ProductCatalog()

    try:
        plan_type = PlanType(args.plan)
    except ValueError:
        print(f"Invalid plan: {args.plan}")
        print(f"Valid plans: {[p.value for p in PlanType]}")
        return 1

    risk_class = RiskClass.STANDARD_PLUS
    if args.risk:
        try:
            risk_class = RiskClass(args.risk)
        except ValueError:
            print(f"Invalid risk class: {args.risk}")
            print(f"Valid: {[r.value for r in RiskClass]}")
            return 1

    quote = catalog.quote_premium(
        plan_id=plan_type,
        age=args.age,
        coverage_amount=args.coverage,
        risk_class=risk_class,
        term_years=args.term,
    )

    if "error" in quote:
        print(f"Error: {quote['error']}")
        return 1

    formatter = OutputFormatter(args.format)

    if formatter.is_json:
        formatter.print_json(quote)
    else:
        print("\nPremium Quote")
        print("=" * 40)
        print(f"  Plan: {quote['plan_name']}")
        print(f"  Age: {quote['age']}")
        print(f"  Risk Class: {quote['risk_class']}")
        if "term_years" in quote:
            print(f"  Term: {quote['term_years']} years")
        print(f"  Coverage: ${quote.get('coverage_amount', 0):,.0f}")
        print("-" * 40)
        print(f"  Monthly Premium: ${quote['monthly_premium']:,.2f}")
        print(f"  Annual Premium: ${quote['annual_premium']:,.2f}")
        if "projected_cash_value_year_10" in quote:
            print(
                f"  Projected Cash Value (Year 10): ${quote['projected_cash_value_year_10']:,.2f}"
            )

    return 0


@register_command("list-models")
def list_models_command(args: argparse.Namespace) -> int:
    """List known models and providers."""
    from salesbench.models import DEFAULT_BENCHMARK_MODELS, list_known_models

    models_by_provider = list_known_models()
    formatter = OutputFormatter(args.format)

    if formatter.is_json:
        output = {
            "models_by_provider": models_by_provider,
            "default_benchmark_set": DEFAULT_BENCHMARK_MODELS,
        }
        formatter.print_json(output)
    else:
        print("Known Models by Provider")
        print("=" * 60)
        for provider, models in sorted(models_by_provider.items()):
            print(f"\n{provider.upper()}")
            print("-" * 40)
            for model in models:
                marker = " *" if model in DEFAULT_BENCHMARK_MODELS else ""
                print(f"  {model}{marker}")

        print("\n" + "=" * 60)
        print("Default benchmark set (* above):")
        for model in DEFAULT_BENCHMARK_MODELS:
            print(f"  - {model}")

        print("\nUsage:")
        print("  salesbench run-benchmark                           # Runs default set")
        print("  salesbench run-benchmark --models openai/gpt-4o    # Single model")
        print(
            "  salesbench run-benchmark --models openai/gpt-4o,anthropic/claude-sonnet-4-20250514"
        )

    return 0


@register_command("list-domains")
def list_domains_command(args: argparse.Namespace) -> int:
    """List available sales domains."""
    # Import domains package to trigger registration
    import salesbench.domains.insurance  # noqa: F401
    from salesbench.domains import get_domain, list_domains

    domains = list_domains()
    formatter = OutputFormatter(args.format)

    if formatter.is_json:
        output = []
        for name in domains:
            domain = get_domain(name)
            output.append(
                {
                    "name": domain.config.name,
                    "display_name": domain.config.display_name,
                    "description": domain.config.description,
                    "product_types": domain.config.product_types,
                    "tools": domain.config.tools,
                }
            )
        formatter.print_json(output)
    else:
        print("Available Sales Domains")
        print("=" * 60)
        for name in domains:
            domain = get_domain(name)
            print(f"\n{domain.config.display_name} [{name}]")
            print(f"  {domain.config.description}")
            print(f"  Products: {', '.join(domain.config.product_types)}")
            print(f"  Tools: {len(domain.config.tools)} available")

    return 0
