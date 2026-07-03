$ErrorActionPreference = "Stop"

$env:Path = "C:\Users\cecca\AppData\Local\Programs\MiKTeX\miktex\bin\x64;C:\Strawberry\perl\bin;C:\Strawberry\c\bin;$env:Path"

$perl = "C:\Strawberry\perl\bin\perl.exe"
$latexmk = "C:\Users\cecca\AppData\Local\Programs\MiKTeX\scripts\latexmk\latexmk.pl"

& $perl $latexmk `
  "-synctex=1" `
  "-interaction=nonstopmode" `
  "-file-line-error" `
  "-pdflatex=C:/Users/cecca/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe %O %S" `
  "-pdf" `
  "main.tex"
