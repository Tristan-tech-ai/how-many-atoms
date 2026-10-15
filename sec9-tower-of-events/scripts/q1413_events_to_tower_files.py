"""q1413 (9 Oct 2026, registered in amendment 2152): convert the event files event_Kb<K>.json of the arb walker q1411 (small_gap_q206/fw_prod/) for K_b >= 83 into the
files that q1183.measured reads: scratch/tower_K<K_b+1>_birth_out.json (key A_cross, K_b even, a birth K -> K+1) and tower_K<K_b+1>_As.json (key A_s,
K_b odd, a split). An existing file is never overwritten; an event whose kind does not match the parity of K_b, a K_b below 83 or an unreadable file stops with exit 2.
Usage: py q1413_events_to_tower_files.py [--registration-run]  (from temporal_within/centre_scripts/). --registration-run converts a synthetic event into a temporary directory
and checks the file names, keys, the refusal to overwrite and the refusal of a wrong kind."""

import glob, json, os, re, shutil, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert(src_dir, dst_dir):
    done = []
    for f in sorted(
        glob.glob(os.path.join(src_dir, "event_Kb*.json")),
        key=lambda p: int(re.search(r"Kb(\d+)", p).group(1)),
    ):
        d = json.load(open(f))
        K = int(d["K_b"])
        if K < 83:
            continue
        kind = d["kind"]
        if (K % 2 == 0) != (kind == "birth"):
            raise ValueError(f"K_b {K}: kind {kind} does not match the parity")
        name, key = (
            (f"tower_K{K + 1}_birth_out.json", "A_cross")
            if kind == "birth"
            else (f"tower_K{K + 1}_As.json", "A_s")
        )
        out = os.path.join(dst_dir, name)
        if os.path.exists(out):
            done.append((K, name, "exists, kept"))
            continue
        json.dump(
            {
                key: float(d["A_event"]),
                "A_event_20_digits": d["A_event"],
                "instrument": "q1411 arb walker (amendment 2152)",
            },
            open(out, "w"),
        )
        done.append((K, name, "written"))
    return done


def registration_run():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "src")
    dst = os.path.join(tmp, "dst")
    os.makedirs(src)
    os.makedirs(dst)
    json.dump(
        {"K_b": 83, "kind": "split", "A_event": "52.96769437300000000"},
        open(os.path.join(src, "event_Kb83.json"), "w"),
    )
    json.dump(
        {"K_b": 84, "kind": "birth", "A_event": "53.45918779170000000"},
        open(os.path.join(src, "event_Kb84.json"), "w"),
    )
    json.dump(
        {"K_b": 82, "kind": "birth", "A_event": "52.47472053143398"},
        open(os.path.join(src, "event_Kb82.json"), "w"),
    )
    r1 = convert(src, dst)
    ok = sorted(os.listdir(dst)) == ["tower_K84_As.json", "tower_K85_birth_out.json"]
    ok = (
        ok
        and json.load(open(os.path.join(dst, "tower_K84_As.json")))["A_s"] == 52.967694373
        and "A_cross" in json.load(open(os.path.join(dst, "tower_K85_birth_out.json")))
    )
    r2 = convert(src, dst)
    ok = ok and all(t[2] == "exists, kept" for t in r2)
    json.dump({"K_b": 85, "kind": "birth", "A_event": "1.0"}, open(os.path.join(src, "event_Kb85.json"), "w"))
    wrong = False
    try:
        convert(src, dst)
    except ValueError:
        wrong = True
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"controls: names and keys {ok}, wrong kind refused {wrong}")
    return ok and wrong


if __name__ == "__main__":
    if not registration_run():
        print("VOID: control failed")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("controls only")
        sys.exit(0)
    try:
        res = convert(os.path.join(ROOT, "small_gap_q206", "fw_prod"), os.path.join(ROOT, "scratch"))
    except Exception as e:
        print("VOID:", repr(e))
        sys.exit(2)
    for K, name, st in res:
        print(f"K_b {K}: {name} {st}")
    print(f"converted {len(res)} events")
