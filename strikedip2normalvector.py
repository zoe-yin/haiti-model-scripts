import numpy as np

def fnormal(strike, dip):
    """
    Function to compute the fault normal vector
    given the strike and dip (in degrees).
    
    Parameters:
    strike (float): Strike angle in degrees.
    dip (float): Dip angle in degrees.
    
    Returns:
    np.ndarray: Fault normal vector (north, east, vertical components).
    """
    deg_to_rad = np.pi / 180
    
    # Convert degrees to radians
    strike = strike * deg_to_rad
    dip = dip * deg_to_rad
    
    # Calculate the fault normal vector components
    n = np.zeros(3)
    n[0] = -np.sin(dip) * np.sin(strike)  # north component
    n[1] = np.sin(dip) * np.cos(strike)   # east component
    n[2] = np.cos(dip)                   # vertical component (positive is up)
    
    return n

# Example usage:
strike = 265
dip = 66

normal_vector = fnormal(strike, dip)
print(normal_vector)
