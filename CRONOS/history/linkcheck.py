import re
from pathlib import Path
root = Path(__file__).resolve().parents[2]
files = [root / "README.md"] + sorted((root / "CRONOS" / "doc").rglob("*.md")) + sorted((root / "CRONOS" / "plotting").rglob("*.md"))
bad = 0
for f in files:
    fence = False
    for line in f.read_text().split("\n"):
        if line.lstrip().startswith("```"): fence = not fence
        if fence: continue
        for m in re.finditer(r"\]\(([^)\s]+)\)", line):
            t = m.group(1).split("#")[0]
            if t and not t.startswith(("http", "mailto")) and not (f.parent / t).exists():
                bad += 1; print("BROKEN", f.relative_to(root), "->", m.group(1))
print(f"checked {len(files)} files, {bad} broken")
