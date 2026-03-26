#!/usr/bin/env python3
"""
Engineering Docs Viewer
=======================
Browse and search the docs/ directory from the command line.

Usage:
    python docs_viewer.py                  # show the index
    python docs_viewer.py list             # list all available docs
    python docs_viewer.py show <name>      # display a doc (fuzzy name match)
    python docs_viewer.py search <term>    # full-text search across all docs
    python docs_viewer.py help             # show this help
"""

import os
import sys
import textwrap

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

# ANSI colours (disabled automatically on non-TTY)
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


BOLD  = lambda t: _c("1", t)
CYAN  = lambda t: _c("96", t)
GREEN = lambda t: _c("92", t)
YELLOW = lambda t: _c("93", t)
DIM   = lambda t: _c("2", t)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_docs() -> list[str]:
    """Return sorted list of .md filenames in the docs directory."""
    if not os.path.isdir(DOCS_DIR):
        return []
    return sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".md"))


def _read_doc(filename: str) -> str:
    path = os.path.join(DOCS_DIR, filename)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _find_doc(name: str) -> str | None:
    """
    Fuzzy-find a doc by name fragment.
    Exact filename match wins; otherwise first doc whose stem contains `name`.
    """
    docs = _all_docs()
    name_lower = name.lower().removesuffix(".md")

    # exact stem match
    for doc in docs:
        if doc.removesuffix(".md").lower() == name_lower:
            return doc

    # substring match
    for doc in docs:
        if name_lower in doc.lower():
            return doc

    return None


def _render_markdown(text: str) -> str:
    """Minimal markdown render: headers, code fences, bold."""
    lines = text.splitlines()
    out = []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            out.append(DIM("  " + line) if in_code else DIM("  " + line))
            continue
        if in_code:
            out.append(DIM("  " + line))
            continue
        if line.startswith("# "):
            out.append(BOLD(CYAN(line)))
        elif line.startswith("## "):
            out.append(BOLD(GREEN(line)))
        elif line.startswith("### "):
            out.append(BOLD(YELLOW(line)))
        elif line.startswith("| "):
            out.append(DIM(line))
        else:
            out.append(line)
    return "\n".join(out)


def _separator(char: str = "─", width: int = 72) -> str:
    return DIM(char * width)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list():
    docs = _all_docs()
    if not docs:
        print("No docs found in", DOCS_DIR)
        return
    print(BOLD(f"Engineering Docs  ({DOCS_DIR})"))
    print(_separator())
    for i, doc in enumerate(docs, 1):
        stem = doc.removesuffix(".md")
        print(f"  {CYAN(str(i).rjust(2))}  {BOLD(stem):<30}  {DIM(doc)}")
    print(_separator())
    print(DIM(f"  {len(docs)} document(s). Use `show <name>` to read one."))


def cmd_show(name: str):
    doc = _find_doc(name)
    if doc is None:
        print(f"No doc matching '{name}'. Run `list` to see available docs.")
        sys.exit(1)
    content = _read_doc(doc)
    path = os.path.join(DOCS_DIR, doc)
    print(_separator())
    print(BOLD(f"  {doc}") + DIM(f"  ({path})"))
    print(_separator())
    print(_render_markdown(content))
    print(_separator())


def cmd_search(term: str):
    docs = _all_docs()
    term_lower = term.lower()
    results: list[tuple[str, list[tuple[int, str]]]] = []

    for doc in docs:
        content = _read_doc(doc)
        matches = [
            (i + 1, line)
            for i, line in enumerate(content.splitlines())
            if term_lower in line.lower()
        ]
        if matches:
            results.append((doc, matches))

    if not results:
        print(f"No results for '{term}'.")
        return

    total_hits = sum(len(m) for _, m in results)
    print(BOLD(f"Search results for '{term}' — {total_hits} match(es) in {len(results)} doc(s)"))
    print(_separator())

    for doc, matches in results:
        print(BOLD(CYAN(f"  {doc}")))
        for lineno, line in matches:
            # Highlight the matched term
            highlighted = line.replace(term, YELLOW(BOLD(term)))
            if not _USE_COLOR:
                highlighted = line
            print(f"    {DIM(str(lineno).rjust(4))}  {highlighted}")
        print()

    print(_separator())
    print(DIM(f"  Use `show <docname>` to read the full document."))


def cmd_index():
    index = _find_doc("index")
    if index:
        cmd_show("index")
    else:
        cmd_list()


def cmd_help():
    print(textwrap.dedent(f"""\
        {BOLD('Engineering Docs Viewer')}

        {BOLD('Commands:')}
          {CYAN('python docs_viewer.py')}                  Show the index
          {CYAN('python docs_viewer.py list')}             List all docs
          {CYAN('python docs_viewer.py show <name>')}      Display a doc (fuzzy match)
          {CYAN('python docs_viewer.py search <term>')}    Search all docs
          {CYAN('python docs_viewer.py help')}             Show this help

        {BOLD('Examples:')}
          python docs_viewer.py show physics
          python docs_viewer.py show self_correction
          python docs_viewer.py search "energy conservation"
          python docs_viewer.py search checkpoint

        {BOLD('Docs location:')}  {DIM(DOCS_DIR)}
    """))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if not args:
        cmd_index()
    elif args[0] == "list":
        cmd_list()
    elif args[0] == "show" and len(args) >= 2:
        cmd_show(" ".join(args[1:]))
    elif args[0] == "search" and len(args) >= 2:
        cmd_search(" ".join(args[1:]))
    elif args[0] == "help":
        cmd_help()
    else:
        print(f"Unknown command: {' '.join(args)}")
        print("Run `python docs_viewer.py help` for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
