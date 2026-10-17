LATEX ?= pdflatex

.PHONY: all clean
all: architecture.pdf

architecture.pdf: architecture.tex figures/network.tex figures/training.tex figures/unrolled.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error architecture.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error architecture.tex

clean:
	rm -f architecture.aux architecture.log architecture.out architecture.fls architecture.fdb_latexmk
