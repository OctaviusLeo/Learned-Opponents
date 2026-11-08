# Contributing

## Checks to run before a commit

GitHub Actions (`.github/workflows/ci.yml`) runs these on every push and pull request, after
`pip install -r requirements-ci.txt && pip install --no-deps -e .` on Python 3.12:

```bash
pipx run --spec ruff==0.17.3 ruff check .                  # rules in pyproject.toml
pytest -m "not gpu"                                        # also checks the generated files below
python -m learned_opponents.export.parity --fixtures-only  # all six ONNX models vs saved actions
```

The GPU tests (`~/.venvs/learned-opponents/bin/python -m pytest` in WSL) and the Unreal
automation tests (`scripts\run_ue_tests.ps1`) run only on the local GPU desktop; see "Tests" in
the README.

## Results are generated

Never edit a file under `results/` or a README result block by hand. `results/summary.md` is
written by `python -m learned_opponents.eval.summary` from `results/eval/*.json`, and the README
result blocks by `python -m learned_opponents.eval.readme` from that summary. The CI suite fails
when either one no longer matches its source. A changed result comes from re-running the script
that produced it, with its command, seed and commit recorded.

## Dependencies

Python packages are exact pins shared by `requirements-ci.txt`, `requirements-gpu.lock` and the
recorded results, so they change only together with a re-run, and `THIRD_PARTY.md` and
`docs/ENVIRONMENT.md` must list the same versions. Check a new package's license before adding it
and record it in `THIRD_PARTY.md`. Dependabot (`.github/dependabot.yml`) only opens monthly bumps
for the GitHub Actions in the workflow.

## Branches

`master` is the default branch and its CI stays green.
