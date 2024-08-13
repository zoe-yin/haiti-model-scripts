
# Treat .yaml as a text file
def extract_R_values_from_text(file_path):
    R_values = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'R =' in line:
                parts = line.split('R =')
                if len(parts) > 1:
                    try:
                        R_value = float(parts[1].strip().split()[0])
                        R_values.append(R_value)
                    except ValueError:
                        pass  # Handle the case where conversion to float fails
    
    return R_values

# Example usage
file_path = '/Users/hyin/agsd/projects/insar/2021_haiti/dynamic-rupture/data_tmp/scripts/parameter-tracking/inputs/Haiti_initial_stress.yaml'

R_values = extract_R_values_from_text(file_path)

print(R_values)

# # Treat Yaml as a yaml file: 
# # Parse the YAML content
# import yaml
# file_path = '/Users/hyin/agsd/projects/insar/2021_haiti/dynamic-rupture/data_tmp/scripts/parameter-tracking/inputs/Haiti_alpha.yaml'
# import yaml

# with open(file_path, 'r') as file:
#     data = yaml.safe_load(file)
# # # Access the second value of alpha
# # second_alpha = data['components'][1]['components']['map']['alpha']

# # print(f"The second value of alpha is: {second_alpha}")

# print(data)