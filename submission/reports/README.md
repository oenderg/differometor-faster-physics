# Draft before/after reports (not final)

Made Sat 21 Nov ~21:04 on the team judge (i7-13700K, WSL2, one logical CPU via `taskset` + `--cpus 1`,
judge lock held) to test the artifact handoff for area E. **Regenerate after the code freeze** for the
real submission.

| | commit | complete first run | warm median | result |
|---|---|---:|---:|---|
| before | `8169dff` (original dense Differometor in this repo) | 27.91 s | 11.09 s | PASS |
| after | `a8b6c67` (main at that time, after #9) | 0.508 s | 0.089 s | PASS |

Files: `before.json` / `after.json` (raw benchmark reports), `before.png` / `after.png` (phase plots),
`before.log` / `after.log` (console output incl. PASS line). Command per side:
`taskset -c 2 .venv/bin/python benchmark.py --cpus 1 --repeats 3 --output <side>.json --plot <side>.png`
