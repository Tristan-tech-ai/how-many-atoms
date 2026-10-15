"""q1260 (8 Oct 2026, a reading, no item): the three sha256 values printed in Section 6.4, recomputed. Draft reading 4 finding 7.
Rule used by the paper: sha256 of json.dumps(object, sort_keys=True) of the named object. Objects: q1201_table.json without the key sha256 (printed fcf88352...), q1209_predictions.json without the key sha256 (printed f48c3ea9...), the object under the key pred of q1207_predictions.json (printed 64759517...), and also the whole q1207_predictions.json without the key sha256.
Control (first): the same procedure on q1220_table.json reproduces the sha256 stored in it (c5102c2a... in Section 6.1 files; the file carries its own key sha256).
Usage: py q1260_hash_check.py  (from temporal_within/centre_scripts/)"""

import hashlib, json, sys

H = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()
j = json.load(open("q1220_table.json"))
stored = j.pop("sha256")
ok = H(j) == stored
print(f"control: q1220_table.json recomputed {H(j)[:16]} stored {stored[:16]}: {ok}")
if not ok:
    print("READING q1260: VOID (control)")
    sys.exit(2)
for f, key, want in (
    ("q1201_table.json", None, "fcf88352"),
    ("q1209_predictions.json", None, "f48c3ea9"),
    ("q1207_predictions.json", "pred", "64759517"),
    ("q1207_predictions.json", "whole", "64759517"),
):
    j = json.load(open(f))
    j.pop("sha256", None)
    o = j["pred"] if key == "pred" else j
    h = H(o)
    print(f"{f} object {key or 'file without sha256'}: {h} starts with {want}: {h.startswith(want)}")
print("READING q1260 (no verdict)")
