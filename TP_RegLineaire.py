\documentclass{article}      % Specifies the document class

% -------------------- Packages --------------------
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[noend]{algpseudocode}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{float}
\usepackage{fontawesome5}
\usepackage{listings}

\lstset{language=Python,keywordstyle={\bfseries \color{blue}}}
\NewDocumentCommand{\codeword}{v}{%
    \texttt{\textcolor[HTML]{5c5c65}{#1}}%
}


\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    pdftitle={Rapport Probabilités et statistiques},
    pdfpagemode=FullScreen,
    }

\urlstyle{same}

\usepackage{bookmark}
\hypersetup{hidelinks} %enlève les cadres rouges autour des hyperliens


% ---------- PSEUDO CODE : hack to remove indent ----------
% https://tex.stackexchange.com/questions/354564/how-to-remove-leading-indentation-from-algorithm
\usepackage{xpatch}
\makeatletter
\xpatchcmd{\algorithmic}
  {\ALG@tlm\z@}{\leftmargin\z@\ALG@tlm\z@}
  {}{}
\makeatother

\usepackage{xcolor}
\usepackage[framemethod=tikz]{mdframed}
\usepackage{tikzpagenodes}
\usetikzlibrary{calc}

% add foreach
\algnewcommand\algorithmicforeach{\textbf{for each}}
\algdef{S}[FOR]{ForEach}[1]{\algorithmicforeach\ #1\ \algorithmicdo}



% -------------------- Couleurs --------------------
\definecolor{definition}{HTML}{2f80ed}
\definecolor{definition-bg}{HTML}{e0ecfd}

\definecolor{danger}{HTML}{e6505f}
\definecolor{danger-bg}{HTML}{fce5e7}

\definecolor{exogris}{gray}{0.4}



% -------------------- Code --------------------
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{code-style}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}

% -------------------- Styles --------------------
\mdfdefinestyle{definition-style}{%
  innertopmargin=10px,
  innerbottommargin=10px,
  linecolor=definition,
  backgroundcolor=definition-bg,
  roundcorner=4px
}
\newmdenv[style=definition-style]{definition}

\mdfdefinestyle{danger-style}{%
  innertopmargin=10px,
  innerbottommargin=10px,
  linecolor=danger,
  backgroundcolor=danger-bg,
  roundcorner=4px
}
\newmdenv[style=danger-style]{danger}


% -------------------- Document --------------------
\title{Probabilités et Statistiques\\\Large{Projet noté}}
\author{MADANI Abdenour\\TRIOLET Hugo}
\date{Licence 3\\2021 - 2022}
\begin{document}
\normalsize
\maketitle

\renewcommand*\contentsname{Table des matières}
\tableofcontents
\newpage



\section{Introduction}
\subsection{Objectifs}
Les objectifs de ce TPs sont :
\begin{itemize}
  \item implémenter nous-mêmes plusieurs algorithmes de régression linéaire et les comparer à des fonctions issues de librairies scientifiques
  \item manipuler différentes lois vues en cours via leur implémentation issues de librairies scientifiques
  \item déterminer des intervalles de confiance et effectuer des applications sur quelques exemples
\end{itemize}

On utilisera pour ceci \textbf{Python} et les bibliothèques de fonctions : Numpy, Scipy, Matplotlib, et Statsmodels, entre autres.



\subsection{Définitions}
\begin{definition}
{ \scriptsize \textcolor{definition}{\faIcon{graduation-cap} \textbf{DÉFINITION}}}
\vspace{3px}
\\ \underline{\textbf{Mot défini}}
\vspace{2.5px}
\\ Définition ici
\end{definition}


\subsection{Résumé de notre approche}
Nous avons 3 fichiers, 1 pour chaque TP.
\\%
\\Vis-à-vis du code, nous l’avons documenté à l’aide de la docstring de Python, ainsi que des commentaires normaux : les fonctions se comprennent donc naturellement grâce à ceux-ci.

\section{Régression linéaire}
\subsection{Régression Linéaire simple}
La fonction calculant la régression linéaire simple est regression\_lineaire.
\\%
\\Étant donné deux listes $x$ et $y$ de même taille, ellec alcule la régression linéaire $$y = \beta_1 \cdot x + \beta_0$$
%

\subsubsection{Modèle vectoriel}
texte

\subsection{Régression linéaire et descente de gradient}



\section{Étude et manipulation de lois de probabilités}
\subsection{Loi Binomiale}
texte

Si tu veux mettre une image Hugo
\begin{figure}[H]
    \centering
     \scalebox{.35}{  % le chiffre c'est le pourcentage à laquelle l'image est scale
                      % ici c'est 0.35 donc 35%
        %\includegraphics{adresse-image.png}
    }
    \\Un graphe orienté non fortement connexe
\end{figure}
\subsection{Loi Normale univariée}
texte
\subsection{Simulation de données à partir d’une loi}
texte

\subsubsection{Cas de la loi normale}
texte

\subsection{Estimation de densité}
texte

\subsubsection{Cas de la loi normale}
texte
\subsubsection{Cas de la loi exponentielle}
texte


\section{Intervalles de confiance}
\subsection{Problème 1}
texte
\subsection{Problème 2}
texte
\subsection{Problème 3}
texte



\section{Exemples d'utilisation du code}
\subsection{Comment utiliser le code}
texte


\end{document}
