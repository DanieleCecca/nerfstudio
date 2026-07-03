# Tesi — versione template AI4ST

Versione della tesi in `../main.tex` impaginata con il template
[thesis-templatex](https://github.com/toolleeo/thesis-templatex) (front page AI4ST).

## Contenuto

Il testo dei capitoli è identico a `../main.tex`, suddiviso in file `.inc.tex`.
Immagini, figure e pseudo-code sono collegati alla cartella padre tramite junction Windows.

## Compilazione

```powershell
.\scripts\build_thesis.ps1
```

Oppure, con `make` (Git Bash / WSL):

```bash
make all
```

Output: `thesis.pdf`
