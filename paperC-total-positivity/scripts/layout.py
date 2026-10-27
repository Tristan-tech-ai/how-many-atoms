"""Where the scripts of this folder find their input files. In the repository they read the shared folders next to their own
paper folder (checks/, sec10-births-law/, sec12-total-positivity/, sec13-monotonicity/) and the data/ folders of the paper folders;
the first candidate of each list that exists is used, so the same scripts also run in the working folder where they were written.
find(key) returns one input file, state(name) one stored optimal state named as in count_law_data.json, and state_names() the
names of the 94 states of the computer-assisted proof of paper C."""
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))   # working folder
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))                  # repository root
PLACES = {
    "count_law_data.json": ["gate_truth/count_law_data.json", "@checks/outputs/count_law_data.json"],
    "ring32_polished.json": ["@checks/inputs/F_work__ring32_R2fix_polished.json",
                             "../how-many-atoms-unified/checks/inputs/F_work__ring32_R2fix_polished.json"],
    "q1382_states_to15p8.json": ["centre_scripts/q1382_states_to15p8.json",
                                 "@paperC-total-positivity/data/q1382_states_to15p8.json"],
    "q1389_out_all.txt": ["small_gap_q206/q1389_out_all.txt", "@sec13-monotonicity/logs/q1389_out_all.txt"],
    "q1389_out_gap.txt": ["small_gap_q206/q1389_out_gap.txt", "@sec13-monotonicity/logs/q1389_out_gap.txt"],
    "q1389_out_big.txt": ["small_gap_q206/q1389_out_big.txt", "@sec13-monotonicity/logs/q1389_out_big.txt"],
    "q1416_rows.json": ["small_gap_q206/q1416_rows.json", "@sec10-births-law/logs/q1416_rows.json"],
    "q1496_out.txt": ["small_gap_q206/q1496_out.txt", "@sec13-monotonicity/logs/q1496_out.txt"],
}
FIGSTYLE_DIRS = [os.path.join(PROJECT, "paper_unified", "figs"), HERE]


def _cand(c):
    return os.path.join(REPO, c[1:]) if c.startswith("@") else os.path.join(PROJECT, c)


def find(key):
    for c in PLACES[key]:
        p = _cand(c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{key}: none of {[_cand(c) for c in PLACES[key]]}")


def state(name):
    for p in (os.path.join(PROJECT, name), os.path.join(PROJECT, "small_gap_q206", name),
              os.path.join(REPO, "paperA-lower-bound", "data", "stored-optima", os.path.basename(name))):
        if os.path.exists(p):
            return p
    return None


def state_names():
    d = os.path.join(PROJECT, "small_gap_q206", "q1882_states")
    if os.path.isdir(d):
        return sorted(os.listdir(d))
    logs = glob.glob(os.path.join(REPO, "sec12-total-positivity", "logs", "q1882_states__*.txt"))
    return sorted(os.path.basename(p)[len("q1882_states__"):] for p in logs)
