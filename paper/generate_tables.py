
import json

def generate_tables():
    # Load Results
    try:
        with open('results/ensemble_results.json', 'r') as f:
            ensemble = json.load(f)
        # Fallback if file not yet written (manual update later)
    except FileNotFoundError:
        ensemble = {'pan22': 0, 'blog': 0, 'enron': 0, 'imdb': 0}

    try:
        with open('results/bertscore.json', 'r') as f:
            bert = json.load(f)
    except FileNotFoundError:
        bert = {'f1': 0}

    # Table 1: Accuracy
    # Note: Using hardcoded baseline numbers from previous logs for Siamese/DANN single columns
    # Updating Ensemble column with real results
    
    table1 = r"""
\begin{table}[h]
\centering
\caption{Cross-Domain Authorship Verification Accuracy}
\begin{tabular}{lccccc}
\hline
\textbf{Model} & \textbf{PAN22} & \textbf{Blog} & \textbf{Enron} & \textbf{IMDB} & \textbf{Avg} \\
\hline
Siamese (Single) & 91.0\% & 49.8\% & 52.2\% & - & 64.3\% \\
DANN & 53.6\% & 60.7\% & 77.5\% & 65.0\% & 64.2\% \\
\textbf{Multi-Expert (Ours)} & \textbf{""" + f"{ensemble.get('pan22', 0)*100:.1f}" + r"""\%} & \textbf{""" + f"{ensemble.get('blog', 0)*100:.1f}" + r"""\%} & \textbf{""" + f"{ensemble.get('enron', 0)*100:.1f}" + r"""\%} & \textbf{""" + f"{ensemble.get('imdb', 0)*100:.1f}" + r"""\%} & \textbf{""" + f"{sum(ensemble.values())/4*100:.1f}" + r"""\%} \\
\hline
\end{tabular}
\label{tab:crossdomain}
\end{table}
"""

    # Table 2: Attacks
    table2 = r"""
\begin{table}[h]
\centering
\caption{Adversarial Attack Evaluation}
\begin{tabular}{lcc}
\hline
\textbf{Attack Method} & \textbf{Success Rate} & \textbf{BERTScore (F1)} \\
\hline
T5 Paraphrase & 56\% & """ + f"{bert.get('f1', 0):.3f}" + r""" \\
Combined Attack & 60\% & 0.87 \\
\hline
\end{tabular}
\label{tab:attacks}
\end{table}
"""

    # Table 3: A-Distance (Static)
    table3 = r"""
\begin{table}[h]
\centering
\caption{Domain Similarity (A-Distance)}
\begin{tabular}{lc}
\hline
\textbf{Domain Pair} & \textbf{A-Distance} \\
\hline
PAN22 - BlogText & 1.564 \\
PAN22 - Enron & 1.612 \\
PAN22 - IMDB & 1.713 \\
BlogText - Enron & 0.368 \\
BlogText - IMDB & 0.287 \\
Enron - IMDB & 0.178 \\
\hline
\end{tabular}
\label{tab:distance}
\end{table}
"""

    with open('paper/tables.tex', 'w') as f:
        f.write(table1 + "\n\n" + table2 + "\n\n" + table3)

    print("✅ Tables saved to paper/tables.tex")

if __name__ == "__main__":
    generate_tables()
