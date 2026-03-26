import subprocess
import sys
import os

if __name__=="__main__":
    datasets = ["ml-1m", "beauty", "steam", "ml-20m"]
    models = ["bert", "nmf"]
    seeds = list(range(5))
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                subprocess.run([sys.executable, "-u", "quantization.py", "-m", f"{model}", "-d", f"{dataset}", 
                                "-s", f"{seed}", "-p", "fp32", "fp16", "int8", "fp4", "nf4"], check=True)
    