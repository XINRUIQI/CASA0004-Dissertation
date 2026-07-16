"""
One-shot migration: rename model tokens inside existing baseline CSVs
(values unchanged) and relocate folders to Flat/ / Deep/.

Does NOT retrain. Safe to re-run (idempotent for already-migrated tokens).

  python3 04_code/scripts/tools/migrate_model_names.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from model_naming import CSV_TOKEN_RENAMES  # noqa: E402

ROOT = SRC_DIR.parent.parent
BASE = ROOT / "05_outputs" / "baselines"


def _rename_token(s: str) -> str:
    if not isinstance(s, str):
        return s
    out = s
    # longest keys first
    for old, new in sorted(CSV_TOKEN_RENAMES.items(), key=lambda kv: -len(kv[0])):
        if old in out:
            out = out.replace(old, new)
    return out


def migrate_csv(path: Path) -> bool:
    """Rename columns, index names, and object cells. Returns True if changed."""
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        df = pd.read_csv(path)
        has_index = False
    else:
        has_index = True

    changed = False
    new_cols = [_rename_token(str(c)) for c in df.columns]
    if new_cols != list(map(str, df.columns)):
        df.columns = new_cols
        changed = True

    if has_index:
        new_idx = [_rename_token(str(i)) for i in df.index]
        if new_idx != list(map(str, df.index)):
            df.index = new_idx
            changed = True
        # also rename index name if needed
        if df.index.name:
            nn = _rename_token(str(df.index.name))
            if nn != df.index.name:
                df.index.name = nn
                changed = True

    for col in df.columns:
        if df[col].dtype == object:
            new_vals = df[col].map(lambda x: _rename_token(x) if isinstance(x, str) else x)
            if not new_vals.equals(df[col]):
                df[col] = new_vals
                changed = True

    if changed:
        df.to_csv(path)
    return changed


def relocate_dirs() -> None:
    """Move Flat Model/* and Deep Model -> Flat/M*_Flat and Deep/."""
    flat_src = BASE / "Flat Model"
    deep_src = BASE / "Deep Model"
    flat_dst = BASE / "Flat"
    deep_dst = BASE / "Deep"

    mapping = {
        "m1 flat": "M1_Flat",
        "m1 Flat": "M1_Flat",
        "M1 Flat": "M1_Flat",
        "m2 flat": "M2_Flat",
        "m2 Flat": "M2_Flat",
        "M2 Flat": "M2_Flat",
        "m3": "M3_Flat",
        "m3 flat": "M3_Flat",
        "m4": "M4_Flat",
        "m4 flat": "M4_Flat",
    }

    flat_dst.mkdir(parents=True, exist_ok=True)
    if flat_src.exists():
        for child in list(flat_src.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            target_name = mapping.get(child.name, child.name.replace(" ", "_"))
            # normalize known patterns
            low = child.name.lower().replace(" ", "")
            if low in ("m1flat", "m1"):
                target_name = "M1_Flat"
            elif low in ("m2flat", "m2"):
                target_name = "M2_Flat"
            elif low in ("m3flat", "m3"):
                target_name = "M3_Flat"
            elif low in ("m4flat", "m4"):
                target_name = "M4_Flat"
            dest = flat_dst / target_name
            if dest.exists():
                # merge: move files that aren't already there
                for f in child.iterdir():
                    if f.name.startswith("."):
                        continue
                    t = dest / f.name
                    if not t.exists():
                        shutil.move(str(f), str(t))
                shutil.rmtree(child, ignore_errors=True)
            else:
                shutil.move(str(child), str(dest))
            print(f"  Flat: {child.name} -> {dest.relative_to(BASE)}")
        # remove empty Flat Model
        try:
            if flat_src.exists() and not any(flat_src.iterdir()):
                flat_src.rmdir()
            elif flat_src.exists():
                # leftover .DS_Store etc.
                for f in flat_src.iterdir():
                    if f.name.startswith("."):
                        f.unlink(missing_ok=True)
                if not any(flat_src.iterdir()):
                    flat_src.rmdir()
        except OSError:
            pass

    if deep_src.exists():
        if deep_dst.exists():
            for f in deep_src.iterdir():
                if f.name.startswith(".") or f.is_dir() and f.name in ("M1 Deep", "M2 Deep"):
                    if f.is_dir():
                        shutil.rmtree(f, ignore_errors=True)
                    continue
                t = deep_dst / f.name
                if not t.exists():
                    shutil.move(str(f), str(t))
            shutil.rmtree(deep_src, ignore_errors=True)
        else:
            shutil.move(str(deep_src), str(deep_dst))
        print(f"  Deep: -> {deep_dst.relative_to(BASE)}")

    # On case-insensitive FS (macOS), Deep and deep are the same path —
    # never rmtree a "legacy" folder that resolves to the destination.
    legacy = BASE / "deep"
    if legacy.exists():
        try:
            same = legacy.resolve() == deep_dst.resolve()
        except OSError:
            same = str(legacy).lower() == str(deep_dst).lower()
        if not same:
            deep_dst.mkdir(parents=True, exist_ok=True)
            for f in legacy.iterdir():
                if f.name.startswith("."):
                    continue
                t = deep_dst / f.name
                if not t.exists():
                    shutil.move(str(f), str(t))
            shutil.rmtree(legacy, ignore_errors=True)
            print("  merged legacy baselines/deep/")
        else:
            print("  Deep/deep are the same path (case-insensitive FS); skip merge")


def main() -> None:
    print(f"Relocating under {BASE} …")
    relocate_dirs()

    print("Migrating CSV tokens …")
    n = 0
    for csv in sorted(BASE.rglob("*.csv")):
        if migrate_csv(csv):
            print(f"  updated {csv.relative_to(BASE)}")
            n += 1
    print(f"Done. {n} CSV files updated.")


if __name__ == "__main__":
    main()
