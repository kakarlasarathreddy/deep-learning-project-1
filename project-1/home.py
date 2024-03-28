import streamlit as st
from randomdata import page_1
from defineddata import page_2

# Add custom CSS to set the background color


    # Create a multiselect widget to select the page
page_options = ["Home", "randomdata", "defineddata"]
selected_page = st.sidebar.radio('',page_options)

if "Home" in selected_page:
    home_image = open(r'C:\Users\HOME\OneDrive\Desktop\Tensorflow-septmber-update-social (2) (1) (1).png', 'rb').read()
    st.image(home_image, caption='Welcome to TensorFlow Play Ground by sarath reddy', use_column_width=True)

    styled_text = """<div style="font-size: 50px; color: white; text-align: left;">
    <span style="font-weight: bold;">TensorFlow Play Ground</span> 
    <div style="position:absolute; top:10%; left:50%; transform: translate(-50%, -50%); text-align:center;">
        <h1 style="color:blue;">TensorFlow Play Ground</h1>
        <h2 style="color:blue";sarath reddy</h2>
    </div>
    """



if "randomdata" in selected_page:
    page_1()

if "defineddata" in selected_page:
    page_2()