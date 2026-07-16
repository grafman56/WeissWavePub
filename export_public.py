#!/usr/bin/env python3
"""Export a sanitized snapshot of this repo to the PUBLIC one.

    python export_public.py            # build + commit the snapshot, DO NOT push
    python export_public.py --push     # ...and push it

WHY THIS EXISTS. On 2026-07-16 the private repo started tracking Paul's
proprietary suite (combined.py, strategies.json, bot_strategies.json) so PC2
could pull it. WeissWavePub is public and forked at 4f6706d, before that. A push
mirror copies a branch VERBATIM -- it cannot filter -- so the public repo cannot
mirror from the private one. Something has to produce a sanitized tree. This is
that something, and it is the ongoing cost of the fork. Paul chose it knowingly.

THE DESIGN RULE THIS PROTECTS: the private working copy has exactly ONE remote
(origin -> GitLab). It must never gain a public one -- that was the whole leak
vector, and removing it is why `git push public` now fails with "does not appear
to be a git repository" instead of needing a hook to catch it. So this script
NEVER adds a remote here. It operates on a SEPARATE checkout (PUB_DIR), which
has its own remote and never contains the forbidden files.

IT DOES NOT PUSH BY DEFAULT. Publishing is irreversible: a blob in a public
history does not come back out. So the default is build-and-show; --push is a
second, deliberate act.

FAIL-CLOSED. Every check aborts rather than warns. A sanitizer that warns is a
sanitizer that gets ignored at 1am.
"""

import os
import shutil
import subprocess
import sys

# The suite is Paul's: hand-written in Pine over years and ported by him. Not
# distributed. CLAUDE.md/BACKLOG.md are his working notes -- not secret, but not
# a portfolio artifact either.
FORBIDDEN = [
    "weisswave/combined.py",
    "strategies.json",
    "bot_strategies.json",
    "CLAUDE.md",
    "BACKLOG.md",
]

PUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "WeissWavePub")
PUB_REMOTE = "https://github.com/grafman56/WeissWavePub"

# What the PUBLIC repo's .gitignore must contain. Belt and braces: even if a
# forbidden file is copied into that tree by hand, git will not stage it.
PUB_IGNORE_HEAD = """# This is the PUBLIC mirror of WeissWave. It is generated -- do not edit here.
# Source of truth is the private repo; regenerate with export_public.py.
#
# Paul's proprietary signal suite is NOT part of this repository and never has
# been: it was never tracked in any commit, so there is nothing to find in the
# history. These entries exist so a stray copy cannot be staged by accident.
weisswave/combined.py
strategies.json
bot_strategies.json
CLAUDE.md
BACKLOG.md

"""


def run(cmd, cwd=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        die(f"{' '.join(cmd)}\n{p.stderr.strip() or p.stdout.strip()}")
    return p.stdout.strip()


def die(msg):
    print(f"\nABORT: {msg}\n")
    sys.exit(1)


def main():
    push = "--push" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))

    # ── the private tree must be clean, or "snapshot of <sha>" is a lie ──────
    if run(["git", "status", "--porcelain"], here):
        die("the private tree has uncommitted changes. A snapshot names a "
            "commit; export what is actually committed or the label is wrong.")

    sha = run(["git", "rev-parse", "--short", "HEAD"], here)
    subj = run(["git", "log", "-1", "--format=%s"], here)

    # ── the private tree must not have a public remote. If it does, someone
    #    undid the design and this script is the least of the problems. ───────
    remotes = run(["git", "remote"], here).split()
    for r in remotes:
        url = run(["git", "remote", "get-url", r], here)
        if "WeissWavePub" in url or "github.com" in url:
            die(f"the PRIVATE tree has a public remote ({r} -> {url}).\n"
                f"That is the leak vector the whole design removes. "
                f"`git remote remove {r}` before exporting.")

    # ── the public checkout, separate from this one ─────────────────────────
    pub = os.path.abspath(PUB_DIR)
    if not os.path.isdir(os.path.join(pub, ".git")):
        print(f"cloning {PUB_REMOTE}\n   -> {pub}")
        run(["git", "clone", PUB_REMOTE, pub])
    else:
        run(["git", "fetch", "origin"], pub)
        run(["git", "checkout", "main"], pub)
        run(["git", "reset", "--hard", "origin/main"], pub)

    # Carry the identity across. Paul's is set LOCALLY in the private repo, not
    # globally, so a fresh clone has none and the commit dies with "Author
    # identity unknown" -- which is how this was found. Copying it also means
    # the public history is authored by him rather than by whatever a machine
    # happens to have configured.
    for k in ("user.name", "user.email"):
        v = run(["git", "config", k], here, check=False)
        if v:
            run(["git", "config", k, v], pub)

    # ── copy the tracked files, minus the forbidden ─────────────────────────
    tracked = run(["git", "ls-tree", "-r", "HEAD", "--name-only"], here).splitlines()
    keep = [f for f in tracked if f not in FORBIDDEN]
    dropped = [f for f in tracked if f in FORBIDDEN]

    for entry in os.listdir(pub):            # wipe, so a deletion propagates
        if entry == ".git":
            continue
        p = os.path.join(pub, entry)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

    for f in keep:
        dst = os.path.join(pub, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(here, f), dst)

    # the public .gitignore is OURS, not a copy of the private one
    with open(os.path.join(pub, ".gitignore"), encoding="utf-8") as fh:
        body = fh.read()
    body = "\n".join(l for l in body.splitlines()
                     if l.strip() and not l.lstrip().startswith("#")
                     and l.strip() not in FORBIDDEN)
    with open(os.path.join(pub, ".gitignore"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(PUB_IGNORE_HEAD + body + "\n")

    # ── VERIFY. Fail closed: nothing forbidden may exist in that tree at all,
    #    tracked or not, and nothing forbidden may be in its history. ─────────
    for f in FORBIDDEN:
        if os.path.exists(os.path.join(pub, f)):
            die(f"{f} is present in the public tree after sanitising. "
                f"The copy step is wrong. Nothing was pushed.")
    hist = run(["git", "log", "--all", "--oneline", "--"] + FORBIDDEN, pub,
               check=False)
    if hist:
        die(f"a forbidden path is ALREADY IN THE PUBLIC HISTORY:\n{hist}\n"
            f"A blob in a public history does not come back out. This needs a "
            f"human, not a script.")

    run(["git", "add", "-A"], pub)
    staged = run(["git", "diff", "--cached", "--name-only"], pub)
    if not staged:
        print(f"\npublic repo already matches {sha}. Nothing to export.\n")
        return 0

    for f in run(["git", "diff", "--cached", "--name-only"], pub).splitlines():
        if f in FORBIDDEN:
            die(f"{f} is STAGED in the public repo. Nothing was pushed.")

    run(["git", "commit", "-m",
         f"snapshot of the private repo at {sha}\n\n"
         f"Private HEAD: {sha} {subj}\n\n"
         f"Generated by export_public.py. Paul's proprietary signal suite is "
         f"not part of this repository and never has been -- it was never "
         f"tracked in any commit here, so there is nothing to find in the "
         f"history.\n\n"
         f"Excluded from this export: {', '.join(FORBIDDEN)}"], pub)

    print(f"\nsanitised snapshot of {sha} committed in {pub}")
    print(f"  kept    {len(keep)} files")
    print(f"  dropped {len(dropped)}: {', '.join(dropped)}")
    print("\nchanged:")
    for f in staged.splitlines()[:20]:
        print(f"    {f}")

    if not push:
        print(f"\nNOT pushed (default). Review, then:")
        print(f"    cd {pub} && git push origin main")
        print(f"  or re-run: python export_public.py --push\n")
        return 0

    run(["git", "push", "origin", "main"], pub)
    print(f"\npushed to {PUB_REMOTE}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
