import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import argparse
import matplotlib
import os

# Function to compute moment magnitude (Mw) from moment rate data
def computeMw(label, time, moment_rate):
    M0 = np.trapz(moment_rate[:], x=time[:])  # Calculate seismic moment (M0) using the trapezoidal rule
    Mw = 2.0 * np.log10(M0) / 3.0 - 6.07  # Convert M0 to moment magnitude (Mw)
    print(f"{label} moment magnitude: {Mw} (M0 = {M0:.4e})")  # Print the result
    return Mw

# Set matplotlib parameters for consistent plotting
ps = 8
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)
matplotlib.rcParams['lines.linewidth'] = 0.5

# Set up argument parser to handle command line inputs
parser = argparse.ArgumentParser(description="plot moment rate comparison")
parser.add_argument("prefix_paths", nargs="+", help="path to prefix of simulations to plots")
parser.add_argument("--labels", nargs="+", help="labels associated with the prefix")
args = parser.parse_args()

# Check if labels are provided and match the number of prefix paths
if args.labels:
    assert len(args.prefix_paths) == len(args.labels)

# Create output directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")

# Create figure and axis for mainshock
fig, ax = plt.figure(figsize=(5, 2)), plt.subplot(111)

cols_mainshock = ["m", "b", "g", "y"]

# Loop through each prefix path and process the data
for i, prefix_path in enumerate(args.prefix_paths):
    df0 = pd.read_csv(f"{prefix_path}/output-energy.csv")  # Read CSV file containing energy data
    df0 = df0.pivot_table(index="time", columns="variable", values="measurement")  # Pivot table to organize data by time

    df = df0[df0.index < 85]  # Filter data for mainshock (0-85s)
    print("warning: Mw computed over 0-85s (avoiding contribution of small residual moment after rupture)")
    
    df["seismic_moment_rate"] = np.gradient(df["seismic_moment"], df.index[1] - df.index[0])  # Compute seismic moment rate
    label = args.labels[i] if args.labels else os.path.basename(prefix_path)  # Set label
    Mw = computeMw(label, df.index.values, df["seismic_moment_rate"])  # Compute moment magnitude

    ax.plot(df.index.values, df["seismic_moment_rate"] / 1e19, cols_mainshock[i], label=f"{label} (Mw={Mw:.2f})")  # Plot moment rate

# Plot comparison data from Melgar et al., 2023 and USGS
# Melgar = np.loadtxt(f"../../ThirdParty/moment_rate_Melgar_et_al_mainshock.txt")
# ax.plot(Melgar[:, 0], Melgar[:, 1], "k", label="Melgar et al., 2023")

usgs = np.loadtxt(f"/Users/hyin/ags_local/data/haiti_seissol_data/usgs-model/moment_rate.txt")
ax.plot(usgs[:, 0], usgs[:, 1]/1e19, "k:", label="USGS")

ax.legend(frameon=False, loc="upper right")
ax.set_xlim([0, 80])
ax.set_ylim(bottom=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_ylabel("Moment Rate \n"+r"($10^{19}$ Nm/s)")
ax.set_xlabel("time (s)")

# Adjust layout to prevent label cutoff
plt.tight_layout()
fn = f"{prefix_path}/moment_rate_mainshock.png"
fig.savefig(fn, bbox_inches="tight", transparent=True, dpi=300)
print(f"done write {fn}")
