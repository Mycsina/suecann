# Makefile for Sueca-WANN LNCS paper
# Build:   make          (compiles main.pdf)
# Clean:   make clean    (removes build artifacts)
# Rebuild: make rebuild  (clean + build)
# Arch:    make arch     (builds only the architecture diagram)

MAIN   = main
TEX    = pdflatex
BIB    = bibtex
OPTS   = -interaction=nonstopmode

.PHONY: all clean rebuild arch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib figures/arch.pdf figures/training_curve.pdf figures/tournament.pdf figures/complexity.pdf figures/topology_follow.pdf
	$(TEX) $(OPTS) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(OPTS) $(MAIN)
	$(TEX) $(OPTS) $(MAIN)

figures/arch.pdf: figures/arch.tex
	cd figures && $(TEX) $(OPTS) arch.tex && rm -f arch.aux arch.log

arch: figures/arch.pdf

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).out $(MAIN).toc figures/arch.aux figures/arch.log

rebuild: clean all
