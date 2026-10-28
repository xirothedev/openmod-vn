#!/usr/bin/env bash
# Publish docs/wiki/*.md to the GitHub Wiki.
# Prerequisite: create the first wiki page once via the web UI
#   https://github.com/xirothedev/openmod-vn/wiki/_new
# After that, this script clones, syncs, and pushes.
set -euo pipefail
WIKI_URL="https://github.com/xirothedev/openmod-vn.wiki.git"
WORK="$(mktemp -d)"
git clone "$WIKI_URL" "$WORK/wiki"
cp -f docs/wiki/*.md "$WORK/wiki/"
cd "$WORK/wiki"
git add -A
if git diff --cached --quiet; then echo "No changes to publish."; exit 0; fi
git commit -m "sync wiki from docs/wiki"
git push origin HEAD
echo "Published. Wiki: https://github.com/xirothedev/openmod-vn/wiki"
rm -rf "$WORK"
