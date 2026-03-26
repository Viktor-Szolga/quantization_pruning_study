import subprocess
import sys
import os

if __name__=="__main__":
    datasets = ["ml-1m", "beauty", "steam", "ml-20m"]
    models = ["bert", "nmf"]
    sparsities = [0.3, 0.5, 0.7]
    seeds = list(range(5))
    for dataset in datasets:
        for model in models:
            for sparsity in sparsities:
                for seed in seeds:
                    subprocess.run([sys.executable, "-u", "quantization.py", "-m", f"{model}", "-d", f"{dataset}", 
                                    "-s", f"{seed}", "-p", "fp32", "fp16", "int8", "fp4", "nf4", "--sparsity", f"{sparsity}"], check=True)
    