# A Large Language Model — from scratch

This repository accompanies my personal walkthrough of the book "Build a Large Language Model from Scratch" (Manning Publications). The implementation and examples here follow the book's material but are my own code and experiments.

Book: https://www.manning.com/books/build-a-large-language-model-from-scratch

This project is a working implementation of concepts and exercises from the book. The goal is practical understanding: I implement, experiment with, and refine components as I progress through the chapters. Changes are committed incrementally — when I feel I've understood and validated a concept — so the repository grows alongside my learning.

## What this repository contains

- A small, educational implementation of transformer building blocks (tokenizer, attention, training loop, etc.). See the top-level Python files such as `tokenizer.py`, `attention.py`, `data_loader.py`, and `main.py`.
- Minimal scripts and helpers to run short experiments and sanity checks rather than a production training pipeline.

Note: this is a study / exploration repository, not a production model. It intentionally favors clarity and pedagogy over performance and scale.

## Status / Progress

- Current approach: I work through the book chapter-by-chapter. For each chapter I:
	- Read and take notes from the source material.
	- Implement the referenced components in code.
	- Run small experiments or unit checks to verify behavior.
	- Commit the results once I'm comfortable with the concept and the implementation.

- Commit policy: commits are frequent and incremental. Expect intermediate experiments, commented-out experiments, and exploratory code — these are part of the learning trail.

If you browse the commit history you'll see the progressive development aligned with chapters and topics.

## Quick start


This project uses plain Python files and the `uv` runner for experiments. Each top-level Python file can be executed directly with `uv run <python-file>` (for example `uv run main.py`). The project declares dependencies in `pyproject.toml` and uses `uv`'s lockfile (`uv.lock`) for reproducible installs; there is no `requirements.txt` file.

Example:

```bash
# install dependencies using uv (if you use uv locally)
# uv install

# run a single example file
uv run main.py

# run another module directly
uv run tokenizer.py
```

Because this repository is focused on learning, many experiments rely only on the Python standard library and NumPy. If a chapter requires additional packages they will be added and documented in `pyproject.toml` and the `uv.lock` file.

## Repository layout

- `tokenizer.py` — small tokenizer implementation and helpers.
- `attention.py` — attention and transformer building blocks.
- `data_loader.py` — dataset loading and preprocessing utilities.
- `main.py` — example runner / experiment entrypoint.
- `data/` — example datasets or small text files used during experiments.

## How to follow along

- Read the book alongside this repository. I try to name commits and branches to indicate the chapter/topic they implement.
- If you want to reproduce an experiment, check the commit message for the date and chapter, then run the related script. If you need help reproducing something, open an issue describing the commit and I'll help when possible.

## Contributing and license

This repository is primarily a personal learning project. If you'd like to suggest improvements or fixes, please open an issue or a pull request. Be aware that some files are intentionally exploratory.

Unless stated otherwise, code in this repository is provided under the MIT license. See `LICENSE` for details (or contact me if you want to reuse code and the license file is missing).

---

I'll continue to work through the book and commit changes as I master each concept. If you'd like to follow progress, watch the repository or check the commit history for chapter-based milestones.

Happy reading and experimenting!

```
