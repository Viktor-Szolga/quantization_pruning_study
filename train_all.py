import subprocess
import sys
import os

if __name__=="__main__":
    datasets = ["ml-20m"]
    models = ["bert", "nmf"]
    seeds = list(range(5))
    for dataset in datasets:
        subprocess.run([sys.executable, "-u", os.path.join("preprocessing", f"{dataset}.py")], check=True)
        for model in models:
            for seed in seeds:
                subprocess.run([sys.executable, "-u", "train.py", "-m", f"{model}", "-d", f"{dataset}", "-s", f"{seed}"], check=True)
    