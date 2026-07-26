#!/usr/bin/env bash
# Pre-publish audit. Run from the repo root BEFORE making the repo public.
#
#     bash scripts/pre_publish_audit.sh
#
# Exits non-zero if anything unsafe would be published. Checks the working
# tree, the staging index, AND full git history — a secret deleted in a later
# commit still lives in history and is still scrapable.

set -uo pipefail
FAIL=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=1; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

echo
echo "=============================================="
echo " AgentCare pre-publish audit"
echo "=============================================="

if [ ! -d .git ]; then
  echo "Not a git repository. Run 'git init' first."
  exit 1
fi

# ---------------------------------------------------------------- 1. .env ---
echo
echo "1. Secret files"

if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  fail ".env is TRACKED by git — it would be published"
else
  pass ".env is not tracked"
fi

if git log --all --pretty=format: --name-only 2>/dev/null | sort -u | grep -qx ".env"; then
  fail ".env appears in git HISTORY — rotate keys and rewrite history"
else
  pass ".env has never been committed"
fi

if [ -f .env ]; then
  if git check-ignore -q .env; then
    pass ".env exists locally and is gitignored"
  else
    fail ".env exists but is NOT gitignored"
  fi
else
  warn ".env not found locally (fine if you haven't created it yet)"
fi

# ------------------------------------------------------- 2. secret values ---
echo
echo "2. Secret values in tracked content"

PATTERNS='(gsk_[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'

HITS=$(git grep -InE "$PATTERNS" -- . 2>/dev/null | grep -v "pre_publish_audit" || true)
if [ -n "$HITS" ]; then
  fail "API-key-shaped strings found in tracked files:"
  echo "$HITS" | head -10 | sed 's/^/        /'
else
  pass "no API-key patterns in tracked files"
fi

HIST=$(git log --all -p 2>/dev/null | grep -aoE "$PATTERNS" | sort -u || true)
if [ -n "$HIST" ]; then
  fail "API-key-shaped strings found in git HISTORY — rotate those keys now"
  echo "$HIST" | head -5 | sed 's/^/        /'
else
  pass "no API-key patterns in git history"
fi

# --------------------------------------------------- 3. .env.example clean ---
echo
echo "3. .env.example is a template, not a config"

if [ -f .env.example ]; then
  FILLED=$(grep -E "^(GROQ_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|OPENROUTER_API_KEY|LLM_API_KEY|JWT_SECRET)=.+" .env.example || true)
  if [ -n "$FILLED" ]; then
    fail ".env.example has FILLED-IN values — blank them:"
    echo "$FILLED" | sed 's/=.*/=<redacted>/' | sed 's/^/        /'
  else
    pass ".env.example has no filled values"
  fi
else
  warn ".env.example missing (advisory check in the challenge CI)"
fi

# ------------------------------------------------------- 4. repo hygiene ---
echo
echo "4. Repository hygiene"

for pattern in ".venv" "__pycache__" "agentcare.db" "checkpoints.sqlite"; do
  if git ls-files | grep -q "$pattern"; then
    fail "$pattern is tracked — remove with: git rm -r --cached <path>"
  else
    pass "$pattern not tracked"
  fi
done

DOCS=$(git ls-files | grep "^storage/documents/" | grep -v ".gitkeep" || true)
if [ -n "$DOCS" ]; then
  fail "uploaded documents are tracked — these may contain test data"
else
  pass "no uploaded documents tracked"
fi

COUNT=$(git ls-files | wc -l | tr -d ' ')
if [ "$COUNT" -gt 200 ]; then
  fail "$COUNT files tracked — that's too many, something unwanted is staged"
else
  pass "$COUNT files tracked (expected roughly 50)"
fi

# --------------------------------------------- 5. challenge critical checks ---
echo
echo "5. Challenge eligibility checks"

if python -m compileall -q app tests scripts >/dev/null 2>&1; then
  pass "all Python compiles (critical check)"
else
  fail "Python syntax errors — CRITICAL check would fail"
fi

if grep -Eiq '^(groq|openai|anthropic|langchain|langgraph|crewai|autogen)' requirements.txt 2>/dev/null; then
  pass "LLM client declared in requirements.txt (critical check)"
else
  fail "no LLM client in requirements.txt — CRITICAL check would fail"
fi

[ -f README.md ] && pass "README.md present" || warn "README.md missing"
grep -qx "\.env" .gitignore 2>/dev/null && pass ".gitignore ignores .env" || fail ".gitignore does not ignore .env"

# ------------------------------------------------------------------ verdict ---
echo
echo "=============================================="
if [ "$FAIL" -eq 0 ]; then
  printf ' \033[32mSAFE TO PUBLISH\033[0m — no secrets would be exposed.\n'
else
  printf ' \033[31mDO NOT PUBLISH YET\033[0m — resolve the FAIL items above.\n'
fi
echo "=============================================="
echo
echo "What the world would actually see:"
git ls-files | head -20 | sed 's/^/  /'
[ "$COUNT" -gt 20 ] && echo "  ... and $((COUNT - 20)) more"
echo
exit "$FAIL"
