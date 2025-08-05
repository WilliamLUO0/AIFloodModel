import pandas as pd
import matplotlib.pyplot as plt

ts_path = "./data/priority_site_24301.txt"
df = pd.read_csv(ts_path, sep=r"\s+", comment="#", header=None, names=["time", "zs", "h", "u", "v"])

# zs water surface elevation
plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["zs"], label="Water Surface Elevation (zs) [m]", color="blue")
plt.xlabel("Time [s]")
plt.ylabel("Elevation [m]")
plt.title("Time Series of Water Surface Elevation at Priority Site")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# h water depth
plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["h"], label="Water Depth (h) [m]", color="blue")
plt.xlabel("Time [s]")
plt.ylabel("Depth [m]")
plt.title("Time Series of Water Depth at Priority Site")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# u water depth
plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["u"], label="Velocity X (u) [m/s]", color="blue")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Time Series of Velocity X at Priority Site")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# h water depth
plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["v"], label="Velocity Y (h) [m/s]", color="blue")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Time Series of Velocity Y at Priority Site")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()