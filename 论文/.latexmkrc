$pdf_mode = 1;
$out_dir = '.';
$aux_dir = '.';

# Use XeLaTeX for Chinese typesetting and keep editor sync options enabled.
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# This thesis uses biblatex with the biber backend.
$bibtex = 'biber %O %B';

# Prefer XeLaTeX as the default engine when latexmk is invoked by editors.
$pdflatex = $xelatex;

# Avoid unnecessary prompts and keep repeated builds stable.
$max_repeat = 5;
$silent = 0;
