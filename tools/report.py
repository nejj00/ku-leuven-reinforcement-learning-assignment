"""
Generates a PDF submission report for the RL Assignment 2025-2026.

The report contains:
  - Student ID and generation timestamp
  - Integrity check: protected files (tests, infrastructure) vs expected hashes
  - Trained weights existence check
  - Syntax validity of all task files
  - Solution blocks: the code the student wrote inside every TODO section
  - Full pytest output with pass/fail summary

Usage (from repo root, inside Docker):
    docker compose run --rm py python tools/report.py --student-id 12345678

Output:
    report/report_<student_id>.pdf
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
EXPECTED_HASHES_PATH = REPO_ROOT / "tools" / "expected_hashes.json"

SOLUTION_START = "# ### YOUR SOLUTION STARTS HERE"
SOLUTION_END = "# ### END OF YOUR SOLUTION"

# Files containing student TODO implementations
TASK_FILES = [
    "task1/dummy.py",
    "task2/tabular_agent.py",
    "task2/mc-agents.py",
    "task2/td-agents.py",
    "task3/deep_agent.py",
    "task3/dqn.py",
    "task3/ppo.py",
]

# Trained weight files that must exist after training
TRAINED_WEIGHTS = [
    "trained_weights/tabular/monte_carlo/empty-room.pkl",
    "trained_weights/tabular/monte_carlo/cliff-minihack.pkl",
    "trained_weights/tabular/monte_carlo/room-with-monster.pkl",
    "trained_weights/tabular/sarsa/empty-room.pkl",
    "trained_weights/tabular/sarsa/cliff-minihack.pkl",
    "trained_weights/tabular/sarsa/room-with-monster.pkl",
    "trained_weights/tabular/q_learning/empty-room.pkl",
    "trained_weights/tabular/q_learning/cliff-minihack.pkl",
    "trained_weights/tabular/q_learning/room-with-monster.pkl",
    "trained_weights/dqn/empty-room.pt",
    "trained_weights/dqn/cliff-minihack.pt",
    "trained_weights/dqn/room-with-monster.pt",
    "trained_weights/ppo/empty-room.pt",
    "trained_weights/ppo/cliff-minihack.pt",
    "trained_weights/ppo/room-with-monster.pt",
]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_file_integrity() -> list[dict]:
    """Compare protected files against expected hashes."""
    if not EXPECTED_HASHES_PATH.exists():
        return [{"file": str(EXPECTED_HASHES_PATH), "status": "MISSING_HASH_FILE",
                 "ok": False, "note": "tools/expected_hashes.json not found"}]

    expected = json.loads(EXPECTED_HASHES_PATH.read_text())
    results = []
    for rel, expected_digest in expected.items():
        p = REPO_ROOT / rel
        if not p.exists():
            results.append({"file": rel, "status": "MISSING", "ok": False,
                            "note": "file not found"})
            continue
        actual = sha256(p)
        ok = actual == expected_digest
        results.append({
            "file": rel,
            "status": "OK" if ok else "MODIFIED",
            "ok": ok,
            "note": "" if ok else "hash mismatch - this file must not be modified",
        })
    return results


def check_trained_weights() -> list[dict]:
    """Check that all expected trained weight files exist."""
    results = []
    for rel in TRAINED_WEIGHTS:
        p = REPO_ROOT / rel
        results.append({
            "file": rel,
            "status": "OK" if p.exists() else "MISSING",
            "ok": p.exists(),
        })
    return results


def check_syntax(task_files: list[str]) -> list[dict]:
    """Verify that every task file parses as valid Python."""
    results = []
    for rel in task_files:
        p = REPO_ROOT / rel
        if not p.exists():
            results.append({"file": rel, "status": "MISSING", "ok": False})
            continue
        try:
            ast.parse(p.read_text(encoding="utf-8"))
            results.append({"file": rel, "status": "OK", "ok": True})
        except SyntaxError as exc:
            results.append({"file": rel, "status": f"SYNTAX ERROR line {exc.lineno}",
                            "ok": False})
    return results


# ---------------------------------------------------------------------------
# Solution block extraction
# ---------------------------------------------------------------------------

def extract_solution_blocks(rel_path: str) -> list[dict]:
    """
    Return a list of solution blocks found in the file.
    Each block is {"file": rel_path, "index": n, "context": "...", "code": "..."}.
    context is the TODO comment immediately above the START marker.
    code is the text between START and END (stripped of leading/trailing blank lines).
    """
    p = REPO_ROOT / rel_path
    if not p.exists():
        return []

    lines = p.read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    block_index = 0

    while i < len(lines):
        if SOLUTION_START in lines[i]:
            # Collect context: scan backwards for the TODO comment
            context_lines = []
            j = i - 1
            while j >= 0 and lines[j].strip().startswith("#"):
                context_lines.insert(0, lines[j].rstrip())
                j -= 1
            context = "\n".join(context_lines).strip()

            # Collect code until END marker
            code_lines = []
            i += 1
            while i < len(lines) and SOLUTION_END not in lines[i]:
                code_lines.append(lines[i].rstrip())
                i += 1

            # Strip leading/trailing blank lines from the code block
            while code_lines and not code_lines[0].strip():
                code_lines.pop(0)
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()

            blocks.append({
                "file": rel_path,
                "index": block_index,
                "context": context,
                "code": "\n".join(code_lines),
                "empty": len(code_lines) == 0,
            })
            block_index += 1
        i += 1

    return blocks


# ---------------------------------------------------------------------------
# Run pytest
# ---------------------------------------------------------------------------

def run_pytest() -> dict:
    """Run the full test suite and return captured output + summary counts."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-vv",
         "tests/test1.py", "tests/test2.py", "tests/test3.py",
         "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    output = result.stdout + result.stderr

    # Parse counts from the final pytest summary line, e.g.
    # "3 passed, 2 failed, 1 error in 1.23s".
    summary_line = ""
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.startswith("=") and " in " in stripped:
            summary_line = stripped.strip("= ").strip()
            break

    def _extract_count(label: str) -> int:
        match = re.search(rf"(\d+)\s+{label}\b", summary_line)
        return int(match.group(1)) if match else 0

    passed = _extract_count("passed")
    failed = _extract_count("failed")
    errors = _extract_count("error") + _extract_count("errors")

    return {
        "output": output,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": passed + failed + errors,
        "returncode": result.returncode,
    }


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

LINE_HEIGHT = 5
CODE_LINE_HEIGHT = 4
MARGIN = 15
PAGE_WIDTH = 210  # A4
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


class ReportPDF(FPDF):
    def __init__(self, student_id: str, timestamp: str):
        super().__init__()
        self.student_id = student_id
        self.timestamp = timestamp
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=MARGIN)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"RL Assignment 2025-2026  |  Student: {self.student_id}  |  {self.timestamp}", align="R")
        self.ln(8)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    # -- helpers -------------------------------------------------------------

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(30, 30, 80)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {text}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def subsection_title(self, text: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 80)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def status_row(self, label: str, ok: bool, note: str = ""):
        tag = "OK" if ok else "FAIL"
        tag_color = (0, 140, 0) if ok else (200, 0, 0)
        self.set_font("Helvetica", "", 9)
        # status badge
        self.set_fill_color(*tag_color)
        self.set_text_color(255, 255, 255)
        self.cell(14, LINE_HEIGHT, tag, fill=True, align="C")
        self.set_text_color(0, 0, 0)
        # label
        label_w = CONTENT_WIDTH - 14 - (60 if note else 0)
        self.cell(label_w, LINE_HEIGHT, f"  {label}")
        if note:
            self.set_text_color(150, 0, 0)
            self.set_font("Helvetica", "I", 8)
            self.cell(60, LINE_HEIGHT, note, align="R")
            self.set_text_color(0, 0, 0)
        self.ln()

    def code_block(self, code: str, max_lines: int = 80):
        """Render a monospace code block with a light grey background."""
        self.set_font("Courier", "", 7)
        self.set_fill_color(245, 245, 245)
        lines = code.splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines truncated)"]
        for line in lines:
            # Truncate very long lines to avoid overflow
            display = line[:120] + ("..." if len(line) > 120 else "")
            self.cell(0, CODE_LINE_HEIGHT, display, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_fill_color(255, 255, 255)
        self.ln(1)


# ---------------------------------------------------------------------------
# Assemble the report
# ---------------------------------------------------------------------------

def build_report(student_id: str, timestamp: str) -> "ReportPDF":
    pdf = ReportPDF(student_id=student_id, timestamp=timestamp)

    # -----------------------------------------------------------------------
    # Page 1 – Cover / Summary
    # -----------------------------------------------------------------------
    pdf.add_page()

    # Title block
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_fill_color(20, 20, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "RL Assignment 2025-2026", fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, "Submission Report", fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Student info
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(35, 7, "Student ID:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, student_id, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(35, 7, "Generated at:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, timestamp, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ---- File integrity ----------------------------------------------------
    pdf.section_title("1. File Integrity Check")
    integrity = check_file_integrity()
    for item in integrity:
        pdf.status_row(item["file"], item["ok"], item.get("note", ""))
    pdf.ln(3)

    # ---- Trained weights ---------------------------------------------------
    pdf.section_title("2. Trained Weights")
    weights = check_trained_weights()
    weights_ok = sum(1 for w in weights if w["ok"])
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, LINE_HEIGHT,
             f"{weights_ok}/{len(weights)} weight files found. "
             "Missing files will cause performance tests to fail.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    for item in weights:
        pdf.status_row(item["file"], item["ok"])
    pdf.ln(3)

    # ---- Syntax check ------------------------------------------------------
    pdf.section_title("3. Task File Syntax")
    syntax = check_syntax(TASK_FILES)
    for item in syntax:
        pdf.status_row(item["file"], item["ok"], item.get("status", "") if not item["ok"] else "")
    pdf.ln(3)

    # ---- Test summary ------------------------------------------------------
    pdf.section_title("4. Test Results Summary")
    pytest_data = run_pytest()
    total = pytest_data["total"]
    passed = pytest_data["passed"]
    pct = int(100 * passed / total) if total > 0 else 0

    pdf.set_font("Helvetica", "B", 22)
    color = (0, 140, 0) if pct == 100 else (200, 120, 0) if pct >= 50 else (200, 0, 0)
    pdf.set_text_color(*color)
    pdf.cell(0, 14, f"{passed}/{total} tests passed  ({pct}%)", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    if pytest_data["errors"] > 0:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, LINE_HEIGHT,
                 f"  {pytest_data['errors']} test(s) raised errors (likely import or setup failures).",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    # -----------------------------------------------------------------------
    # Page 2 – Full pytest output
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("5. Full Test Output")
    pdf.code_block(pytest_data["output"], max_lines=300)

    # -----------------------------------------------------------------------
    # Pages 3+ – Solution blocks
    # -----------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("6. Student Solution Blocks")

    all_blocks = []
    for rel in TASK_FILES:
        all_blocks.extend(extract_solution_blocks(rel))

    if not all_blocks:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 8, "No solution blocks found. Have you implemented the TODOs?", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        empty_count = sum(1 for b in all_blocks if b["empty"])
        if empty_count:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(200, 0, 0)
            pdf.cell(0, LINE_HEIGHT,
                     f"WARNING: {empty_count} solution block(s) appear to be empty.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

        for block in all_blocks:
            # Block header
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(220, 225, 240)
            label = f"  [{block['file']}]  block {block['index'] + 1}"
            if block["empty"]:
                label += "  << EMPTY >>"
            pdf.cell(0, 6, label, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            # Context (the TODO comment above the block)
            if block["context"]:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(0, 4, block["context"])
                pdf.set_text_color(0, 0, 0)
                pdf.ln(1)

            # Code
            if block["empty"]:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, LINE_HEIGHT, "  (empty - no code was written here)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_text_color(0, 0, 0)
            else:
                pdf.code_block(block["code"])

            pdf.ln(2)

    return pdf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PDF submission report for the RL Assignment."
    )
    parser.add_argument("--student-id", required=True,
                        help="Your student number (e.g. 12345678)")
    args = parser.parse_args()

    student_id = args.student_id.strip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Generating report for student {student_id}...")
    print("Running tests (this may take a few minutes)...")

    pdf = build_report(student_id=student_id, timestamp=timestamp)

    out_dir = REPO_ROOT / "report"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"report_{student_id}.pdf"
    pdf.output(str(out_path))

    print(f"\nReport saved to: {out_path}")
    print("Submit this PDF together with your code.")


if __name__ == "__main__":
    main()
