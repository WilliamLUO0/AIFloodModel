import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

river_dir = "./data/river_input/"
river_file = "./data/Porangahau/rid_8233959_BG.txt"
tide_path = "./data/porangahau_basin/tide_Porangahau.txt"


def plot_river_inputs(river_dir):
    river_files = [f for f in os.listdir(river_dir) if f.endswith(".txt")]
    plt.figure(figsize=(12, 6))
    for fname in river_files:
        fpath = os.path.join(river_dir, fname)
        try:
            df = pd.read_csv(fpath, sep="\t", header=None, names=["time", "discharge"])
            plt.plot(df["time"], df["discharge"], label=fname)
        except Exception as e:
            print(f"Failed to read {fname}: {e}")
    plt.xlabel("Time (s)")
    plt.ylabel("Discharge (m³/s)")
    plt.title("River Boundary Condition")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_single_river(river_file_path):
    try:
        df = pd.read_csv(river_file_path, sep="\t", header=None, names=["time", "discharge"])
        plt.figure(figsize=(10, 4))
        plt.plot(df["time"], df["discharge"], label=os.path.basename(river_file_path))
        plt.xlabel("Time (s)")
        plt.ylabel("Discharge (m³/s)")
        plt.title(f"River Boundary Discharge: {os.path.basename(river_file_path)}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Failed to read {river_file_path}: {e}")


def plot_tide_input(tide_path):
    try:
        df_tide = pd.read_csv(tide_path, sep="\s+", header=None, names=["time", "sea_level"])
        plt.figure(figsize=(12, 4))
        plt.plot(df_tide["time"], df_tide["sea_level"], color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("Sea Level (m)")
        plt.title("Tide Boundary Condition")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Failed to read tide file: {e}")


plot_tide_input(tide_path)
plot_single_river(river_file)