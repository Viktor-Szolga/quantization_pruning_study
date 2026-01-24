import pandas as pd
import io
import pyperclip

def generate_latex_table(path_to_csv):
    df = pd.read_csv(path_to_csv)
    column_mapping = {
        'run_name': 'Method',
        'hr1': 'HR@1',
        'ndcg1': 'NDCG@1',
        'hr5': 'HR@5',
        'ndcg5': 'NDCG@5',
        'hr10': 'HR@10',
        'ndcg10': 'NDCG@10'
    }
    
    df = df[list(column_mapping.keys())].copy()
    
    cols_to_round = ['hr1', 'ndcg1', 'hr5', 'ndcg5', 'hr10', 'ndcg10']
    df[cols_to_round] = df[cols_to_round].round(4)

    latex = [
        "\\begin{table*}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Method} & \\textbf{HR@1} & \\textbf{NDCG@1} & \\textbf{HR@5} & \\textbf{NDCG@5} & \\textbf{HR@10} & \\textbf{NDCG@10} \\\\",
        "\\hline"
    ]

    for i, row in df.iterrows():
        row_str = f"{row['run_name']} & {row['hr1']:.4f} & {row['ndcg1']:.4f} & {row['hr5']:.4f} & {row['ndcg5']:.4f} & {row['hr10']:.4f} & {row['ndcg10']:.4f} \\\\"
        latex.append(row_str)
        
        if (i + 1) % 5 == 0:
            latex.append("\\hline")

    latex.extend([
        "\\end{tabular}",
        "\\caption{Performance of BERT under quantization and pruning.}",
        "\\label{tab:bert_perf}",
        "\\end{table*}"
    ])

    return "\n".join(latex)


results = generate_latex_table("results/bert.csv")
results = results.replace(r"_", r"\_")
pyperclip.copy(results)
print("Successfully copied to clipboard")