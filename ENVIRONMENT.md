# OVLAB host CLI environment

The installer creates the named Conda environment `ovlab-env` when Conda is
available. Otherwise it creates a Python `venv` at `.ovlab-env/` in the
repository. Conda is recommended because it pins the host interpreter to
Python 3.10; `venv` uses the Python supplied to the installer. The installer
installs all
OVLAB Python packages from this checkout in editable mode, including the
regular `wandb` dependency of `ovlab-benchctl`. Re-run the installer to update
the environment after changing package requirements; use `--update` to force
a reinstall without requirement changes. Training and evaluation
still use their dedicated Docker images for model and simulator dependencies.

## Linux and macOS

From the repository root:

```bash
source ./install-ovlab-env.sh
```

The installer creates the environment, installs OVLAB and activates it in the
current Bash session. `ovlab` is then on PATH:

```bash
ovlab --help
```

In a new Bash session, activate the Conda environment with
`conda activate ovlab-env`, or run the same `source` command. With the `venv` fallback, run
`source .ovlab-env/bin/activate`. Once installed, the installer skips package
installation unless requirements changed or `--update` is specified. Running
the script with `bash` installs the environment but cannot activate the calling
shell; use `source` for both steps.

Choose a manager explicitly with `--manager conda` or `--manager venv`.
For `venv`, use Python 3.10 or newer; `--python /path/to/python3.10` selects
the base interpreter explicitly.
The bootstrap script itself can use a different Python through
`OVLAB_BOOTSTRAP_PYTHON`. An explicit `--prefix PATH` installs either manager
at a custom path instead of using the named Conda environment or default venv.
The existing environment is reused on subsequent runs.

## Windows Command Prompt

From the repository root:

```bat
install-ovlab-env.bat
```

The installer leaves the environment active in that Command Prompt window, so
the CLI is available immediately:

```bat
ovlab --help
```

In a new Command Prompt window, activate the Conda environment with
`conda activate ovlab-env`, or run the same `.bat` command. With the `venv` fallback,
run `.ovlab-env\Scripts\activate.bat`. For Conda, use an Anaconda Prompt or a
Command Prompt where `conda` is available.

The `.bat` installer accepts `--manager`, `--prefix PATH`, `--python`, and
`--update`. For full Docker based training and evaluation on Windows, run OVLAB
under WSL2 with Linux containers and use `install-ovlab-env.sh` there. The
Windows CLI environment supports host commands such as `ovlab --help`, config
validation, and run inspection.

## Running without activation

On Unix, `./ovlab` uses the named Conda environment after setup or the local
`venv` fallback. Set `OVLAB_PYTHON` to override that choice. On Windows, invoke
`.ovlab-env\Scripts\ovlab.exe` for `venv` or run
`conda run --name ovlab-env ovlab --help` for Conda.

The setup checks package consistency with `pip check`, imports OVLAB and W&B,
and runs `ovlab --version`. W&B logging starts when `OVLAB_WANDB_PROJECT` or
`WANDB_PROJECT` is set. Authenticate with `wandb login` in the activated
environment or set `WANDB_API_KEY`.

After activation, these commands exercise the installed CLI without starting a
training or evaluation container:

```bash
ovlab policy list --json
ovlab eval run configs/experiments/evaluation/libero10-openvla-vanilla-4bit-rpc-smoke.yaml --dry-run --json
```
