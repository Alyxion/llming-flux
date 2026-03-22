"""E2E test harness for document-knowledge MCP agents.

Sends questions through ``claude -p`` with an MCP server in parallel,
then programmatically verifies responses against the DB.
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..database import Database

# ANSI
G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; B = "\033[34m"; C = "\033[36m"
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"

# Language detection markers
_DE_MARKERS = re.compile(r'\b(ist|und|für|der|die|das|eine|einer|wird|kann|sind|Produkt|Produkte|haben|nicht|werden)\b')
_EN_MARKERS = re.compile(r'\b(the|and|for|with|this|that|which|product|search|query|are|has|can|will)\b')


def detect_language(text: str) -> str:
    return "de" if len(_DE_MARKERS.findall(text)) > len(_EN_MARKERS.findall(text)) else "en"


@dataclass
class E2ECase:
    """A single end-to-end test case."""
    id: int
    question: str
    lang: str = "en"
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    verify_products_exist: bool = True
    product_regex: re.Pattern | None = None
    search_query: str = ""
    search_must_find: list[str] = field(default_factory=list)
    must_use_tools: bool = True
    must_ask_followup: bool = False
    custom: Callable[[str, Database], list[str]] | None = None


@dataclass
class VerifyResult:
    case_id: int
    question: str
    passed: bool
    errors: list[str]
    warnings: list[str]
    response_len: int
    products_found: list[str]
    response_preview: str
    duration_s: float = 0.0


def extract_codes_with_regex(text: str, pattern: re.Pattern) -> list[str]:
    matches = pattern.findall(text)
    return list({m.strip().upper().replace(" ", "-") for m in matches if len(m) > 2})


def verify_codes_in_db(codes: list[str], db: Database) -> list[str]:
    errors = []
    for code in codes:
        results = db.search(code.replace("-", " "), limit=5)
        if not results:
            results = db.search(code.replace("-", ""), limit=5)
        if not results:
            errors.append(f"Product code '{code}' not found in DB")
    return errors


def verify_response(case: E2ECase, response: str, db: Database) -> VerifyResult:
    errors: list[str] = []
    warnings: list[str] = []
    resp_lower = response.lower()

    if response.startswith("[ERROR]"):
        return VerifyResult(case.id, case.question, False, [response], [], 0, [], response[:200])

    for term in case.must_mention:
        if term.lower() not in resp_lower:
            errors.append(f"Missing: '{term}'")
    for term in case.must_not_mention:
        if term.lower() in resp_lower:
            errors.append(f"Forbidden: '{term}'")

    products: list[str] = []
    if case.product_regex and case.verify_products_exist:
        products = extract_codes_with_regex(response, case.product_regex)
        if products:
            errors.extend(verify_codes_in_db(products, db))

    if case.search_query:
        results = db.search(case.search_query, limit=20)
        if not results:
            warnings.append(f"DB search '{case.search_query}' → 0 results")
        for frag in case.search_must_find:
            if not any(frag in (r.metadata.get("filename", "") or "") for r in results):
                errors.append(f"Search '{case.search_query}' should find '{frag}'")

    if len(response) > 100:
        detected = detect_language(response)
        if detected != case.lang:
            warnings.append(f"Expected {case.lang}, detected {detected}")

    if case.must_ask_followup:
        markers = ["?", "what type", "which crop", "how fast", "speed", "boom width",
                   "application rate", "welche", "wie schnell", "what pressure"]
        if not any(m.lower() in resp_lower for m in markers):
            warnings.append("No follow-up questions detected")

    if case.custom:
        errors.extend(case.custom(response, db))

    if len(response) < 50:
        errors.append(f"Response too short ({len(response)} chars)")

    return VerifyResult(
        case.id, case.question, len(errors) == 0,
        errors, warnings, len(response),
        products, response[:300].replace("\n", " "))


# ── MCP config ────────────────────────────────────────────

def build_mcp_config(server_command: list[str], server_name: str = "kb",
                     cwd: str | None = None) -> dict:
    cfg: dict[str, Any] = {"command": server_command[0]}
    if len(server_command) > 1:
        cfg["args"] = server_command[1:]
    if cwd:
        cfg["cwd"] = cwd
    return {"mcpServers": {server_name: cfg}}


def write_mcp_config(config: dict, directory: Path) -> Path:
    path = directory / "mcp_config.json"
    path.write_text(json.dumps(config))
    return path


# ── Run single question ───────────────────────────────────

def run_question(question: str, mcp_config_path: Path, tool_prefix: str = "kb",
                 system_preamble: str = "", timeout: int = 180) -> tuple[str, float]:
    """Run a question through ``claude -p``. Returns (response, duration_seconds)."""
    prompt = f"{system_preamble}\n\nUser question: {question}" if system_preamble else question
    cmd = [
        "claude", "-p", prompt,
        "--mcp-config", str(mcp_config_path),
        "--strict-mcp-config",
        "--tools", "",
        "--allowedTools", "mcp__*__*",
        "--permission-mode", "bypassPermissions",
        "--output-format", "text",
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                env={**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL": "1"})
        dur = time.time() - t0
        response = result.stdout.strip()
        if not response and result.stderr:
            return f"[ERROR] {result.stderr.strip()[:500]}", dur
        return response, dur
    except subprocess.TimeoutExpired:
        return "[ERROR] Timeout", time.time() - t0
    except FileNotFoundError:
        return "[ERROR] 'claude' CLI not found", 0.0


# ── Parallel test runner ──────────────────────────────────

_print_lock = threading.Lock()


def _run_one(case: E2ECase, mcp_config_path: Path, db: Database,
             tool_prefix: str, system_preamble: str,
             total: int, verbose: bool, timeout: int) -> VerifyResult:
    """Run and verify a single case (called from thread pool)."""
    with _print_lock:
        print(f"  {C}START{RESET}  Q{case.id:2d}: {case.question[:65]}")
        sys.stdout.flush()

    response, dur = run_question(case.question, mcp_config_path, tool_prefix, system_preamble, timeout)
    vr = verify_response(case, response, db)
    vr.duration_s = dur

    with _print_lock:
        status = f"{G}PASS{RESET}" if vr.passed else f"{R}FAIL{RESET}"
        print(f"  {status}   Q{case.id:2d}: {vr.response_len:>5} chars, {dur:5.1f}s  {case.question[:50]}")
        if not vr.passed:
            for err in vr.errors:
                print(f"         {R}x {err}{RESET}")
        for w in vr.warnings:
            print(f"         {Y}! {w}{RESET}")
        if verbose and vr.response_len > 0:
            print(f"         {DIM}{textwrap.shorten(vr.response_preview, 120)}{RESET}")
        sys.stdout.flush()

    return vr


def run_tests(cases: list[E2ECase], mcp_config_path: Path, db: Database,
              tool_prefix: str = "kb", system_preamble: str = "",
              verbose: bool = False, timeout: int = 180,
              parallel: int = 10) -> list[VerifyResult]:
    """Run test cases in parallel and verify responses."""
    print(f"\n{BOLD}Running {len(cases)} tests ({parallel} parallel workers){RESET}\n")
    t0 = time.time()

    results: list[VerifyResult] = []

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _run_one, case, mcp_config_path, db,
                tool_prefix, system_preamble,
                len(cases), verbose, timeout
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by case id for summary
    results.sort(key=lambda r: r.case_id)
    total_time = time.time() - t0
    print(f"\n{DIM}Total wall time: {total_time:.1f}s{RESET}")
    return results


def print_summary(results: list[VerifyResult], output_path: Path | None = None):
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"{BOLD}Results: {passed}/{total} passed{RESET}", end="")
    if passed < total:
        print(f"  {R}{total - passed} failed{RESET}")
    else:
        print(f"  {G}All passed!{RESET}")

    if passed < total:
        print(f"\n{R}Failed:{RESET}")
        for r in results:
            if not r.passed:
                print(f"  Q{r.case_id}: {r.question[:60]}")
                for e in r.errors:
                    print(f"    - {e}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([{
                "id": r.case_id, "question": r.question, "passed": r.passed,
                "errors": r.errors, "warnings": r.warnings,
                "response_len": r.response_len, "products": r.products_found,
                "preview": r.response_preview, "duration_s": r.duration_s,
            } for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results: {output_path}")
