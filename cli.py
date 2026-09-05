"""Command-line entry point: run the full pipeline without the API or the UI.

Useful for a scripted demo, for debugging a target, and for checking that the
whole graph runs end to end on a machine where you would rather not start two
servers.

    python cli.py https://books.toscrape.com/
    python cli.py https://example.com --intent "focus on checkout" --prd prd.md
    python cli.py https://app.example.com --username qa@example.com --password-env QA_PW

Credentials
-----------
``--password`` is accepted but discouraged: a password typed on a command line
lands in the shell history and in the process table. Prefer ``--password-env``,
which names an environment variable to read, or leave it out entirely and let
the agent test what is publicly reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from config import Settings, ensure_dirs, get_settings, set_settings
from target_policy import TargetPolicy, evaluate_target, log_decision
from graph.state import new_run_id
from logging_setup import configure_logging, get_logger
from security import SECRET_BOX, Credentials, insecure_for_credentials

log = get_logger("aivor.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ato-run",
        description="Autonomous Test Orchestration Agent - plan, generate, run, heal, report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", nargs="?", help="target application URL")
    parser.add_argument("--intent", default=None, help="natural-language scope hint")
    parser.add_argument("--prd", type=Path, default=None, help="path to a PRD (.txt or .md)")

    auth = parser.add_argument_group("authentication (use a throwaway test account)")
    auth.add_argument("--username", default=None)
    auth.add_argument("--password", default=None, help="discouraged: use --password-env instead")
    auth.add_argument(
        "--password-env",
        default=None,
        metavar="VAR",
        help="name of an environment variable holding the password",
    )
    auth.add_argument("--token", default=None, help="bearer token (or use LOGIN_TOKEN)")
    auth.add_argument("--login-url", default=None)

    run = parser.add_argument_group("run options")
    run.add_argument("--headed", action="store_true", help="show the browser window")
    run.add_argument("--max-pages", type=int, default=None, help="crawl page budget")
    run.add_argument("--parallel", action="store_true", help="enable parallel flow execution")
    run.add_argument("--offline", action="store_true", help="LLM_OFFLINE_MODE: stub, no model")
    run.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    safety = parser.add_argument_group("target and safety policy")
    safety.add_argument(
        "--allow-private",
        action="store_true",
        help=(
            "permit a loopback/private/link-local target such as "
            "http://localhost:3000. Cloud metadata endpoints stay blocked."
        ),
    )
    safety.add_argument(
        "--allow-insecure-tls",
        action="store_true",
        help="ignore TLS certificate errors (self-signed staging certificates only)",
    )
    safety.add_argument(
        "--allowlist",
        default=None,
        help="comma-separated hosts this run may test, e.g. '*.staging.example.com'",
    )
    safety.add_argument(
        "--unsafe",
        action="store_true",
        help="DISABLE safe mode: permits payment, checkout, delete and other irreversible actions",
    )
    safety.add_argument(
        "--authorize",
        default=None,
        help=(
            "comma-separated destructive categories to authorise while keeping safe "
            "mode on: payment, checkout, delete, account_cancellation, "
            "password_reset, email_send, irreversible_submit"
        ),
    )
    return parser


def resolve_credentials(args: argparse.Namespace) -> Credentials:
    """Assemble credentials from flags then environment. Values are never echoed."""
    password = args.password
    if args.password_env:
        password = os.environ.get(args.password_env)
        if not password:
            print(
                f"warning: ${args.password_env} is empty; continuing without a password",
                file=sys.stderr,
            )
    return Credentials(
        username=args.username or os.environ.get("LOGIN_USERNAME") or None,
        password=password or os.environ.get("LOGIN_PASSWORD") or None,
        token=args.token or os.environ.get("LOGIN_TOKEN") or None,
        login_url=args.login_url or os.environ.get("LOGIN_URL") or None,
    )


def apply_overrides(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {}
    if args.headed:
        overrides["headless"] = False
    if args.max_pages:
        overrides["crawl_max_pages"] = args.max_pages
    if args.parallel:
        overrides["enable_parallel_execution"] = True
    if args.offline:
        overrides["llm_offline_mode"] = True
    if args.log_level:
        overrides["log_level"] = args.log_level
    if args.allow_private:
        overrides["allow_private_targets"] = True
    if args.allow_insecure_tls:
        overrides["allow_insecure_tls"] = True
    if args.allowlist:
        overrides["target_allowlist"] = tuple(
            chunk.strip() for chunk in args.allowlist.replace(";", ",").split(",") if chunk.strip()
        )
    if args.unsafe:
        overrides["safe_mode"] = False
    if args.authorize:
        overrides["authorized_destructive_actions"] = tuple(
            chunk.strip() for chunk in args.authorize.replace(";", ",").split(",") if chunk.strip()
        )
    settings = get_settings().with_overrides(**overrides) if overrides else get_settings()
    set_settings(settings)
    return settings


async def run(args: argparse.Namespace) -> int:
    from agents.orchestrator import cleanup_run, execute_run

    settings = apply_overrides(args)
    configure_logging(settings.log_level)
    ensure_dirs()

    target = args.url or settings.default_target_url
    if not target.startswith(("http://", "https://")):
        print(f"error: target must be an http(s) URL, got {target!r}", file=sys.stderr)
        return 2

    # Pre-flight target admission. The navigation guard would catch a blocked
    # target anyway, but doing it here turns "the crawl mysteriously found no
    # pages" into one clear sentence naming the rule and how to override it.
    decision = evaluate_target(target, TargetPolicy.from_settings(settings))
    log_decision(decision, context="cli")
    if not decision.allowed:
        print(f"error: {decision.detail}", file=sys.stderr)
        return 4
    if decision.overridden:
        print(
            f"warning: testing a {decision.category} target because --allow-private "
            "was given; cloud metadata endpoints remain blocked.",
            file=sys.stderr,
        )
    if not settings.safe_mode:
        print(
            "warning: SAFE MODE IS OFF. Payments, orders, deletions and account "
            "closures on the target may be performed for real.",
            file=sys.stderr,
        )

    if not settings.llm_available:
        print(
            "error: no LLM provider configured.\n"
            "  Put a free key in .env as GROQ_API_KEY (https://console.groq.com), or\n"
            "  pass --offline to smoke-test the plumbing with a deterministic stub.",
            file=sys.stderr,
        )
        return 3

    prd_text = None
    if args.prd:
        if not args.prd.is_file():
            print(f"error: PRD not found at {args.prd}", file=sys.stderr)
            return 2
        prd_text = args.prd.read_text(encoding="utf-8", errors="replace")

    run_id = new_run_id()
    credentials = resolve_credentials(args)
    if credentials.present():
        for label, target in (("target", target), ("login_url", credentials.login_url)):
            if target and insecure_for_credentials(target):
                print(
                    f"error: refusing to send credentials to {label} {target!r} over plain "
                    "http://. Use https://, or target a loopback/private host such as "
                    "http://localhost:3000 for local development.",
                    file=sys.stderr,
                )
                return 2
        SECRET_BOX.put(run_id, credentials)

    print(f"\n  Target : {target}")
    print(f"  Run id : {run_id}")
    print(f"  Auth   : {'supplied' if credentials.present() else 'none (public pages only)'}")
    print(f"  Model  : {settings.model_reasoning} / {settings.model_codegen}")
    if settings.llm_offline_mode:
        print("  MODE   : LLM_OFFLINE_MODE - deterministic stub, NOT a real agent run")
    print()

    try:
        report = await execute_run(
            run_id=run_id,
            target_url=target,
            user_intent=args.intent,
            prd_text=prd_text,
            settings=settings,
        )
    finally:
        cleanup_run(run_id)

    totals = report.totals or {}
    print("\n" + "=" * 72)
    print(f"  {report.executive_summary}")
    print("=" * 72)
    print(
        f"  flows={totals.get('flows_planned', 0)}  "
        f"generated={totals.get('tests_generated', 0)}  "
        f"passed={totals.get('passed', 0)}  "
        f"failed={totals.get('failed', 0)}  "
        f"healed={totals.get('healed', 0)}  "
        f"bugs={totals.get('bugs_filed', 0)}  "
        f"review={totals.get('needs_human_review', 0)}"
    )
    for name, path in (report.artifacts or {}).items():
        print(f"  {name:<16} {path}")
    print()
    return 1 if report.status == "failed" else 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
