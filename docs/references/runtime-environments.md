# Runtime Environments

The foundational sweep intentionally spans multiple pinned environments. This is architecture, not a temporary workaround.

## Environment Families

- `.venv`: base benchmark environment
- `.venv-tabular`: tabular adapters
- `.venv-timemoe`: Time-MoE, EIDOS, Timer, Sundial
- `.venv-tempopfn`: TempoPFN
- `.venv-moirai`: explicit Moirai family
- `.venv-mantis2`: Mantis family
- `.venv-tivit`: image-model adapters
- `.venv-toto`: Toto 1.0 (`toto-ts`) and Toto 2.0 (`toto2` plus `dd_unit_scaling` from the upstream Toto repo)

The authoritative family-to-environment mapping is the model registry plus `scripts/run_foundational_sequential.py`.

## Repository Rule

Do not hide missing packages and do not try to collapse all families into one environment. If an environment is incomplete, fail with a clear error and repair that environment directly.
