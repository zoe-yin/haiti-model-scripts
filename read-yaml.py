
yamlfile = '/Users/hyin/ags_local/data/haiti_seissol_data/dynamic-rupture/FL33-only/jobid_3457692/logs/inputs/Haiti_fault.yaml'

import re

# Define the file path
file_path = '/mnt/data/Haiti_fault.yaml'

# Initialize variables to store mu_s and mu_d
mu_s = None
mu_d = None

# Define regex patterns to match mu_s and mu_d
mu_s_pattern = re.compile(r'\bmu_s\s*:\s*(\d*\.?\d+)')
mu_d_pattern = re.compile(r'\bmu_d\s*:\s*(\d*\.?\d+)')

# Read the file and search for mu_s and mu_d values
with open(yamlfile, 'r') as file:
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



