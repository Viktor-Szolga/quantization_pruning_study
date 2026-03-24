import subprocess
import sys
import os

if __name__=="__main__":
    datasets = ["ml-1m", "beauty", "steam", "ml-20m"]
    models = ["bert", "nmf"]
    for dataset in datasets:
        subprocess.run([sys.executable, "-u", os.path.join("preprocessing", f"{dataset}.py")], check=True)
        for model in models:
            subprocess.run([sys.executable, "-u", "train.py", "-m", f"{model}", "-d", f"{dataset}"], check=True)
    