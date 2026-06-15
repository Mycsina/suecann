.PHONY: build test clippy bench rules dataset figures report slides

build:
	cargo build -p sueca_wann --release

test:
	cargo test --all

clippy:
	cargo clippy --all

bench:
	./target/release/sueca_wann benchmark \
		--deals 3000 \
		--genome checkpoints/production/2026-06-14-2/genomes/best_genome_final.json

rules:
	./target/release/sueca_wann compile-rules \
		--genome checkpoints/production/2026-06-14-2/genomes/best_genome_final.json \
		--output-dir checkpoints/production/2026-06-14-2

dataset:
	./target/release/sueca_wann generate-dataset \
		--n-worlds 200 --teacher rollout --target-count 15000 \
		--soft-balance-min-ratio 0.0 \
		--output expert_states_v6.npz

figures:
	uv run python scripts/make_report_figures.py

report:
	cd report && latexmk -pdf main.tex

slides:
	cd "presentation slides" && latexmk -pdf slides.tex
