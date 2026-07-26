#!/usr/bin/env bash
# Compact pre-publish audit.
FAIL=0
ok(){ printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
no(){ printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=1; }

echo; echo "1. Secrets"
git ls-files --error-unmatch .env >/dev/null 2>&1 && no ".env is TRACKED" || ok ".env not tracked"
git log --all --pretty=format: --name-only 2>/dev/null | sort -u | grep -qx ".env" \
  && no ".env in git HISTORY — rotate keys" || ok ".env never committed"
KEYS='(gsk_[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,}|ghp_[A-Za-z0-9]{30,})'
H=$(git grep -InE "$KEYS" -- . 2>/dev/null || true)
[ -n "$H" ] && { no "keys in tracked files:"; echo "$H" | head -5 | sed 's/^/       /'; } || ok "no keys in tracked files"
[ -n "$(git log --all -p 2>/dev/null | grep -aoE "$KEYS" | sort -u || true)" ] \
  && no "keys in git HISTORY — rotate now" || ok "no keys in history"
grep -qE "^(GROQ|OPENAI|ANTHROPIC|GOOGLE|OPENROUTER|LLM)_API_KEY=.+|^JWT_SECRET=.+" .env.example 2>/dev/null \
  && no ".env.example has filled values" || ok ".env.example is blank"

echo; echo "2. Hygiene"
git ls-files | grep -qE "\.venv|__pycache__|\.db$|\.sqlite" && no "venv/db/cache tracked" || ok "no venv, db or cache tracked"
N=$(git ls-files | wc -l | tr -d ' ')
[ "$N" -gt 200 ] && no "$N files staged — too many" || ok "$N files staged"

echo; echo "3. Eligibility"
python -m compileall -q app tests scripts >/dev/null 2>&1 && ok "Python compiles (CRITICAL)" || no "syntax errors (CRITICAL)"
grep -Eiq '^(groq|openai|anthropic|langchain|langgraph|crewai|autogen)' requirements.txt \
  && ok "LLM client declared (CRITICAL)" || no "no LLM client in requirements.txt (CRITICAL)"
[ -f README.md ] && ok "README present" || no "README missing"
grep -qx "\.env" .gitignore && ok ".gitignore ignores .env" || no ".gitignore missing .env"

echo
[ "$FAIL" -eq 0 ] && printf ' \033[32mSAFE TO PUBLISH\033[0m\n' || printf ' \033[31mDO NOT PUBLISH\033[0m\n'
echo
exit $FAIL
