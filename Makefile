HOMEWORK_1: $(wildcard *.tex)
	mkdir -p out
	pdflatex --output-directory=out assignment_01_MoCS-2018-09-26.tex
	cp out/assignment_01_MoCS-2018-09-26.pdf assignment_01_MoCS-2018-09-26.pdf
	open assignment_01_MoCS-2018-09-26.pdf

.PHONY: clean
clean:
	rm -f assignment_01_MoCS-2018-09-26.pdf
	rm -rf out/

