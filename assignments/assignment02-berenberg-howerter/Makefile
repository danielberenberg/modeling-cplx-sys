HOMEWORK_1: $(wildcard *.tex)
	mkdir -p out
	pdflatex --output-directory=out assignment_02_MoCS-2018-09-26.tex
	cp out/assignment_02_MoCS-2018-09-26.pdf assignment_02_MoCS-2018-09-26.pdf
	open assignment_02_MoCS-2018-09-26.pdf

.PHONY: clean
clean:
	rm -f assignment_02_MoCS-2018-09-26.pdf
	rm -rf out/

