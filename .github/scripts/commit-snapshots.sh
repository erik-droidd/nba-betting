#!/usr/bin/env bash
# Commit + push new snapshot files to main. Called by `snapshot-loop`
# (every ~30 min and when the run ends). A no-op when there is nothing new;
# a failed push keeps the commit and is retried on the next call.
set -uo pipefail

git config user.name >/dev/null || git config user.name "github-actions[bot]"
git config user.email >/dev/null || git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

paths=(data/odds_snapshots data/injury_snapshots)
if [ -n "$(git status --porcelain -- "${paths[@]}")" ]; then
  git add -- "${paths[@]}"
  # [skip ci] keeps the push from triggering other workflows.
  git commit -q -m "chore(snapshots): $(date -u +'%Y-%m-%d %H:%MZ') [skip ci]"
fi
if [ -z "$(git log --oneline origin/main..HEAD 2>/dev/null)" ]; then
  echo "commit-snapshots: nothing to push"
  exit 0
fi

for attempt in 1 2 3; do
  # Odds files merge as a union (.gitattributes). For an injury day file
  # the commit being replayed (our newer capture) wins: -X theirs.
  if git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main; then
    echo "commit-snapshots: pushed (attempt ${attempt})"
    exit 0
  fi
  # A failed rebase leaves the repo mid-rebase; without this every retry
  # fails with "you have unmerged files" (the 2026-09-13 failure).
  git rebase --abort 2>/dev/null || true
  echo "commit-snapshots: push attempt ${attempt} failed; retrying"
  sleep $((attempt * 5))
done
echo "commit-snapshots: push failed; commit kept for the next call"
exit 1
