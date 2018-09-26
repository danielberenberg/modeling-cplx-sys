HOMEWORK_1: $(wildcard *.tex)
	mkdir -p out
	pdflatex --output-directory=out daniel_berenberg_243_HOMEWORK_1.tex
	cp out/daniel_berenberg_243_HOMEWORK_1.pdf daniel_berenberg_243_HOMEWORK_1.pdf
	open daniel_berenberg_243_HOMEWORK_1.pdf

.PHONY: clean
clean:
	rm -f daniel_berenberg_243_HOMEWORK_1.pdf
	rm -rf out/

