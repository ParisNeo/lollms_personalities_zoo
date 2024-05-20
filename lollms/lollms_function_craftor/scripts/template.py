# Lollms function call definition file
# Here you need to import any necessary imports depending on the function requested by the user
# exemple import math

# Partial is useful if we need to preset some parameters
from functools import partial

# It is advised to import typing elements
# from typing import List

# here is the core of the function to be built
# Change the name of this function depending on the user requets, 
def add_constant(parameter1:str, parameter2:int) -> float: # use typed parameters depending on the requested function, only int, float or text outputs are allowed
    try:
        # handle exceptions

        # Here you perform your computation or you execute the function
        result = 5.5+parameter2 if parameter1=="hi" else 5.5-parameter2
        
        # Finally we return the output
        return result
    except Exception as e:
        return str(e)
    

#Here is the metadata function that shoule has the name in format function_name_function
def return_constantadd_constant_function(processor, client):
    return {
        "function_name": "add_constant", # The function name in string
        "function": add_constant, # The function to be called
        "function_description": "Returns a constabnt value.", # Change this with the description
        "function_parameters": [{"name": "parameter1", "type": "str"}, {"name": "parameter2", "type": "str"}] # Te set of paraeters          
    }