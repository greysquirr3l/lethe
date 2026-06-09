# Ansible Cross-Architecture Runner for Lethe Gates

This folder provisions and launches the cross-architecture run for each lethe
gate (T08 generalisation gate, T21 per-substrate live-DOF pivot, ...).

## What it does

- Installs system dependencies on Debian-family and Arch Linux hosts.
- Clones or updates this repository on each host.
- Installs stable Rust via rustup and builds workspace release binaries.
- Launches the configured gate command inside a tmux session.
- Collects result artifacts back into local results/ with host + arch prefixes.

## 1) Inventory

The inventory is copied from substrate_zero style.

- Template: ansible/inventory.example.ini
- Active: ansible/inventory.ini

Edit hostnames/users in ansible/inventory.ini as needed.

## 2) Configure run variables

Edit ansible/group_vars/all.yml:

- repo_url
- repo_version
- repo_dest
- run_session
- run_name
- gate_command

You must set gate_command before launching.

## Optional: GitHub token for checkout

If a remote host needs authenticated Git checkout, put a token in:

ansible/.github.token

The playbook reads this file on the control machine and injects it into HTTPS clone URLs.

## 3) Launch a run

Edit `ansible/group_vars/all.yml` to point at the gate you want:

- T08 generalisation gate: `run_session: t08_gate`, `run_name: t08_generalisation_gate`,
  `gate_command: ./target/release/lethe-cli gate --output-dir results --samples 160 --burn-in 40`
- T21 per-substrate live-DOF pivot: `run_session: t21_pivot`, `run_name: t21_pivot`,
  `gate_command: ./target/release/lethe-cli pivot --output-dir results/t21_pivot --samples 160 --burn-in 40`

Then run the matching playbook:

```bash
# T08
ansible-playbook -i ansible/inventory.ini ansible/run_t08_gate.yml
ansible-playbook -i ansible/inventory.ini ansible/collect_t08_results.yml

# T21
ansible-playbook -i ansible/inventory.ini ansible/run_t21_pivot.yml
ansible-playbook -i ansible/inventory.ini ansible/collect_t21_results.yml
```

## 4) Collect artifacts

Collected files are written under `results/` using `<host>_<arch>_` prefixes.
The T21 pivot keeps the `t21_pivot/` subdirectory layout intact on the remote
and re-prefixes the per-file name locally.
