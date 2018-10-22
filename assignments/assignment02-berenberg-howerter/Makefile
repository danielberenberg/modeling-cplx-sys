HOMEWORK_1: $(wildcard *.tex)
	mkdir -p out
	pdflatex --output-directory=out assignment_02_MoCS-berenberg-howerter.tex
	cp out/assignment_02_MoCS-berenberg-howerter.pdf assignment_02_MoCS-berenberg-howerter.pdf
	open assignment_02_MoCS-berenberg-howerter.pdf

.PHONY: clean
clean:
	rm -f assignment_02_MoCS-berenberg-howerter.pdf
	rm -rf out/

