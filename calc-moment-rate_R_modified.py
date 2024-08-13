import argparse
import seissolxdmf
import seissolxdmfwriter as sxw
import numpy as np
import glob
import os
import pandas as pd
import matplotlib
import matplotlib.pylab as plt



def find_xdmffile_wahweap(jobid):
    """Find XDMF file on wahweap. 
    Given a jobid, returns a string which is the full path to the extracted-fault.xdmf file on Wahweap laptop

    Args:
        jobid (string): i.e. '3438438'
    Return: 
        xdmfFilename (string): full path to the extracted-fault.xdmf file matching the jobid
        example output: `/Users/hyin/agsd/projects/insar/2021_haiti/dynamic-rupture/data_tmp/dynamic-rupture/FL33-only/jobid_3440215/output_jobid_3440215_extracted-fault.xdmf`
    """

    datadir = '/Users/hyin/ags_local/data/haiti_seissol_data/'

    try:
    # Attempt to find and assign the first matching file
        xdmfFile = glob.glob(datadir + 'dynamic-rupture/*/jobid_'+jobid+'*/*extracted-fault.xdmf')[0]
    except IndexError:
        # Handle the case where no files are found
        print(f"No matching file found for jobid {jobid}. Skipping...")
        xdmfFile = None  # or handle it in another way if needed
    # xdmfFile = glob.glob(datadir + 'dynamic-rupture/*/jobid_'+jobid+'/*extracted-fault.xdmf')[0]
    print(xdmfFile)
    momentCsvFile = glob.glob(datadir + 'dynamic-rupture/*/jobid_'+jobid+'*/output-energy.csv')[0]
    return xdmfFile, momentCsvFile

def define_mu(jobdir):

    import re

    # Define the file path to the Haiti_fault.yaml file
    file_path =  jobdir + '/logs/inputs/Haiti_fault.yaml'

    # Initialize variables to store mu_s and mu_d
    mu_s = None
    mu_d = None

    # Define regex patterns to match mu_s and mu_d
    mu_s_pattern = re.compile(r'\bmu_s\s*:\s*(\d*\.?\d+)')
    mu_d_pattern = re.compile(r'\bmu_d\s*:\s*(\d*\.?\d+)')

    # Read the file and search for mu_s and mu_d values
    with open(file_path, 'r') as file:
        for line in file:
            mu_s_match = mu_s_pattern.search(line)
            mu_d_match = mu_d_pattern.search(line)
            if mu_s_match:
                mu_s = float(mu_s_match.group(1))
            if mu_d_match:
                mu_d = float(mu_d_match.group(1))

    # Print the extracted values
    print("mu_s:", mu_s)
    print("mu_d:", mu_d)

    return mu_s, mu_d


### Calculate Moment Rate
# Function to compute moment magnitude (Mw) from moment rate data
def computeMw(label, time, moment_rate):
    M0 = np.trapz(moment_rate[:], x=time[:])  # Calculate seismic moment (M0) using the trapezoidal rule
    Mw = 2.0 * np.log10(M0) / 3.0 - 6.07  # Convert M0 to moment magnitude (Mw)
    print(f"{label} moment magnitude: {Mw} (M0 = {M0:.4e})")  # Print the result
    return Mw




def generateRMoment(jobid_list):
    for i in range(len(jobid_list)):
        jobid = jobid_list[i]
        print('Working on JobID: ' + jobid)
        xdmfFilename, momentCsvFile = find_xdmffile_wahweap(jobid)
        jobdir = os.path.dirname(momentCsvFile)
        sx = seissolxdmf.seissolxdmf(xdmfFilename)

        # x, y, z components of each point where the surface displacement is defined. 
        xyz = sx.ReadGeometry() # xyz.shape is (16347, 3) np arrray
        connect = sx.ReadConnect()  # connect.shape is (32526, 3) np array

        # define the barycenter of each element
        barycenter = (
            xyz[connect[:, 0], :]
            + xyz[connect[:, 1], :]
            + xyz[connect[:, 2], :]
        ) / 3.0

        # define cohesion in each element
        cohesion = np.maximum((barycenter[:, 2] + 3000) / 3000, 0) * 1e6 + 0.5e6

        # define time step you wish to extract (@todo: is this right?)
        dt = sx.ReadTimeStep()
        outputTimes = sx.ReadTimes()
        idt = int(outputTimes[0])    # Pull first time step
        idt

        # Pull data from the xdmf file
        # Mud is defined as the 'Current effective friction coefficient'
        # We assume this is the static coefficient of friction because we are extracting at time zero. 
        mu_st = sx.ReadData("Mud", idt) 
        Ts0 = sx.ReadData("Ts0", idt)
        Td0 = sx.ReadData("Td0", idt)
        Pn0 = abs(sx.ReadData("Pn0", idt))

        # define dynamic coefficient of friction
        mu_s, mu_d = define_mu(jobdir)  # get value of mu_d from the fault.yaml file
        mu_dy = mu_d

        # Calculate the value of R. Same as definiton but added cohesion to the strength drop
        myData = (np.sqrt(np.power(Ts0, 2) + np.power(Td0, 2)) - mu_dy * Pn0) / (
            (mu_st - mu_dy)* Pn0 + cohesion)

        ## Rewrite to xdmf file
        # Pull other data fields
        geom = sx.ReadGeometry()
        connect = sx.ReadConnect()

        # Write all selected time steps to files dCFS-fault.xdmf/dCFS-fault.h5
        outpath = os.path.dirname(xdmfFilename)
        outfile = outpath + '/jobid_'+jobid+'_R-fault'
        sxw.write(
            outfile,
            geom,
            connect,
            {"R": myData},
            {idt: 0},
            reduce_precision=True,
            backend="hdf5",
        )

        # Set matplotlib parameters for consistent plotting
        ps = 8
        matplotlib.rcParams.update({"font.size": ps})
        plt.rcParams["font.family"] = "sans"
        matplotlib.rc("xtick", labelsize=ps)
        matplotlib.rc("ytick", labelsize=ps)
        matplotlib.rcParams['lines.linewidth'] = 0.5

        # Create figure and axis for mainshock
        fig, ax = plt.figure(figsize=(5, 2)), plt.subplot(111)

        cols_mainshock = ["m", "b", "g", "y"]

        # Loop through each prefix path and process the data
        df0 = pd.read_csv(momentCsvFile)  # Read CSV file containing energy data
        df0 = df0.pivot_table(index="time", columns="variable", values="measurement")  # Pivot table to organize data by time

        df = df0[df0.index < 85]  # Filter data for mainshock (0-85s)
        print("warning: Mw computed over 0-85s (avoiding contribution of small residual moment after rupture)")

        df["seismic_moment_rate"] = np.gradient(df["seismic_moment"], df.index[1] - df.index[0])  # Compute seismic moment rate
        label = 'jobid_'+jobid  # Set label
        Mw = computeMw(label, df.index.values, df["seismic_moment_rate"])  # Compute moment magnitude

        ax.plot(df.index.values, df["seismic_moment_rate"] / 1e19, cols_mainshock[0], label=f"{label} (Mw={Mw:.2f})")  # Plot moment rate

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
        fn = f"{jobdir}/moment_rate_mainshock.png"
        fig.savefig(fn, bbox_inches="tight", transparent=True, dpi=300)
        print(f"Finished writing {fn}")



# Set up argument parser
parser = argparse.ArgumentParser(description='Process job IDs.')
parser.add_argument('jobid_list', nargs='+', help='List of job IDs')

# Parse arguments
args = parser.parse_args()

# Access jobid_list from command line
jobid_list = args.jobid_list

# Sample usage: 
# python calc-moment-rate_R_modified.py 3491360 3491361


generateRMoment(jobid_list)