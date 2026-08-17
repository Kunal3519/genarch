"""
Convert ONLY the new raw towers (50000-150000) -> Converted format, WITHOUT
touching the existing 50k train/val/test splits.

For each tower dir in --src that is NOT already converted:
  - cell_centers (float) * 4 -> new_joint_vecs/<name>.npy   (float32)
  - round(...)              -> new_joint_vecs_int/<name>.npy (int32)
  - prompt                  -> texts/<name>.txt  (full#simplified#0.0#0.0)
Then write --split_out (default all.txt) = list of all newly converted names.

Resumable: skips towers whose int .npy already exists.
Schema-defensive: prints the first tower's params.json keys and aborts clearly
if cell_centers / tower_info.prompt are absent.
"""
import argparse, json, os, re, sys
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True, help='Raw dir with tower_<id>/params.json')
ap.add_argument('--dst', required=True, help='Converted dir (new_joint_vecs* / texts live here)')
ap.add_argument('--split_out', default='all.txt', help='Split filename written into --dst')
ap.add_argument('--existing_split_dirs', nargs='*', default=[],
                help='Optional: existing Converted dirs whose towers to EXCLUDE (already-converted 50k)')
args = ap.parse_args()

SRC, DST = Path(args.src), Path(args.dst)
VECS, VECS_INT, TEXTS = DST/'new_joint_vecs', DST/'new_joint_vecs_int', DST/'texts'
for d in (VECS, VECS_INT, TEXTS):
    d.mkdir(parents=True, exist_ok=True)

def simplify(t):
    t = t.lower(); t = re.sub(r"[^a-z0-9 ]", " ", t); return re.sub(r" +", " ", t).strip()

tower_dirs = sorted([d for d in SRC.iterdir() if d.is_dir() and d.name.startswith('tower_')])
print(f"Found {len(tower_dirs)} tower dirs in {SRC}")

done, skipped, errors, new_names = 0, 0, [], []
for i, td in enumerate(tower_dirs):
    name = td.name
    int_path = VECS_INT/f"{name}.npy"
    if int_path.exists():
        skipped += 1
        new_names.append(name)   # already converted in a prior run of THIS script
        continue
    p = td/'params.json'
    if not p.exists():
        errors.append(f"MISSING params.json: {name}"); continue
    try:
        data = json.load(open(p))
        if i == 0 or done == 0:
            print(f"  schema sample [{name}]: top keys = {list(data.keys())[:8]}")
        centers = np.array(data['cell_centers'], dtype=np.float32) * 4
        centers_int = np.round(centers).astype(np.int32)
        prompt = data['tower_info']['prompt'].strip()
        np.save(VECS/f"{name}.npy", centers)
        np.save(int_path, centers_int)
        (TEXTS/f"{name}.txt").write_text(f"{prompt}#{simplify(prompt)}#0.0#0.0")
        new_names.append(name); done += 1
    except KeyError as e:
        errors.append(f"SCHEMA {name}: missing key {e} (keys={list(data.keys())})")
        if len(errors) <= 3:
            print(f"  !! {errors[-1]}")
    except Exception as e:
        errors.append(f"ERROR {name}: {e}")
    if (done+skipped) % 5000 == 0:
        print(f"  ...{done+skipped}/{len(tower_dirs)} (converted {done}, skipped {skipped})")

print(f"\nConverted {done}, skipped(existing) {skipped}, errors {len(errors)}.")
for e in errors[:10]:
    print("  ", e)

# Exclude already-converted 50k towers (by their int npy presence in given dirs) from the split
exclude = set()
for ed in args.existing_split_dirs:
    ip = Path(ed)/'new_joint_vecs_int'
    if ip.exists():
        exclude |= {f.stem for f in ip.glob('*.npy')}
split_names = [n for n in sorted(set(new_names)) if n not in exclude]
(DST/args.split_out).write_text("\n".join(split_names) + "\n")
print(f"Wrote {DST/args.split_out}: {len(split_names)} towers (excluded {len(exclude)} pre-existing).")
if errors:
    sys.exit(1)
