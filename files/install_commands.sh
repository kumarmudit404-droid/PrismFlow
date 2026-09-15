#!/usr/bin/env bash
# Installs each part prompt as a Claude Code slash command.
# After running this, inside Claude Code you type:  /part01  ... /part16
set -euo pipefail
SRC="${1:-parts}"
DEST=".claude/commands"
mkdir -p "$DEST"
n=0
for f in "$SRC"/PART_*.txt; do
  base=$(basename "$f")
  num=$(echo "$base" | sed -E 's/PART_([0-9]+)_.*/\1/')
  out="$DEST/part${num}.md"
  {
    echo "---"
    echo "description: PrismFlow Part ${num}"
    echo "---"
    echo
    cat "$f"
  } > "$out"
  n=$((n+1))
  echo "  /part${num}  <-  $base"
done
echo
echo "Installed $n slash commands into $DEST"
echo "Restart Claude Code, then type /part01 to begin."
