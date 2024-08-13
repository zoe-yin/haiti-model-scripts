def parse_parameter_file(file_path):
    parameters = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip headers and empty lines
            if line.startswith('&') or not line:
                continue
            
            # Remove comments
            if '!' in line:
                line = line.split('!')[0].strip()
            
            # Split key-value pairs
            if '=' in line:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                
                # Convert value to float if possible, otherwise keep as string
                try:
                    value = float(value)
                except ValueError:
                    pass
                
                parameters[key] = value
    
    return parameters

# Example usage
file_path = '/Users/hyin/agsd/projects/insar/2021_haiti/dynamic-rupture/data_tmp/scripts/parameter-tracking/inputs/parameters.par'
parameters = parse_parameter_file(file_path)

print(parameters)
