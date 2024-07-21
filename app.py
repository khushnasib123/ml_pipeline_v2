# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:32:26 2024

@author: Ajay Patidar
"""

import streamlit as st

# User input elements
st.title("Your Web Application Title")
number_input = st.number_input("Enter a number:", min_value=0)

# Processing logic (replace with your specific calculations)
squared_number = number_input * number_input

# Output display
st.write("The squared value is:", squared_number)
