import importlib, subprocess, sys

required = [
    # Core training deps from step 1 of setup.sh
    ("torch",        "2.2"),
    ("numpy",        None),
    ("gymnasium",    None),
    ("tyro",         None),
    ("wandb",        None),
    ("tqdm",         None),
    ("transforms3d", None),
    ("einops",       None),   # ActionDecoderHead / MAPBlock
    ("peft",         None),   # LoRA
]

ok = True
for name, min_ver in required:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        if min_ver and not ver.startswith(min_ver):
            print(f"[WARN] {name}: expected >={min_ver}, got {ver}")
        else:
            print(f"[OK]   {name} {ver}")
    except ImportError as e:
        print(f"[FAIL] {name}: {e}")
        ok = False

# Editable installs from step 2 of setup.sh
editable = ["mani_skill", "simpler_env", "openvla", "univla"]
for pkg in editable:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", pkg],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        loc = [l for l in result.stdout.splitlines() if l.startswith("Location:")]
        print(f"[OK]   {pkg} installed — {loc[0] if loc else ''}")
    else:
        print(f"[FAIL] {pkg} not found by pip show")
        ok = False

if not ok:
    raise SystemExit("One or more packages failed — see above.")

print()
print("=== setup.sh integrity PASSED — all packages installed correctly ===")
