# Ansible Cross-Architecture Runner for Lethe T08

This folder provisions and launches the T08 generalisation gate run on multiple hosts.

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

## 3) Launch T08 gate run

ansible-playbook -i ansible/inventory.ini ansible/run_t08_gate.yml

## 4) Collect artifacts

ansible-playbook -i ansible/inventory.ini ansible/collect_t08_results.yml

Collected files are written under results/ using host + arch prefixes.
