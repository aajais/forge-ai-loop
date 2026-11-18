"""Produce scoped diffs for this non-Git workspace's implementation reviews."""
import difflib
import sys
from pathlib import Path

task = sys.argv[1]
paths = sys.argv[2:]
before = Path(f'docs/forge-loop-work/task{task}-before')
output = Path(f'docs/forge-loop-work/task{task}-diff.txt')
chunks = []
for name in paths:
    old = before / name
    current = Path(name)
    chunks.append(''.join(difflib.unified_diff(
        old.read_text().splitlines(keepends=True) if old.exists() else [],
        current.read_text().splitlines(keepends=True) if current.exists() else [],
        fromfile=f'before/{name}', tofile=f'after/{name}', n=5,
    )))
output.write_text('\n'.join(chunks))
print(output)
