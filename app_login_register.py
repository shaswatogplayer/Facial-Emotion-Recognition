import streamlit as st
import subprocess
import os
import sqlite3
import sys

# Check if 'stage', 'logged_in' keys exist in the session state
if 'stage' not in st.session_state:
    st.session_state['stage'] = 'login'  # Initial stage is 'login'
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False  # Initialize it as False

# Setup SQLite database
conn = sqlite3.connect('userdata.db')
c = conn.cursor()

# Create table if not exists
c.execute('''
    CREATE TABLE IF NOT EXISTS userdata
    (name TEXT, age INTEGER, gender TEXT, graphfile BLOB)
''')

if not st.session_state['logged_in']:
    if st.session_state['stage'] == 'login':
        # Login form
        st.subheader('Login')
        name = st.text_input('Name')
        age = st.number_input('Age', min_value=1, max_value=100)

        if st.button('Login'):
            # Check if user exists in the database
            c.execute('''
                SELECT * FROM userdata WHERE name = ? AND age = ?
            ''', (name, age))
            user = c.fetchone()
            if user is not None:
                # User exists, log them in
                st.session_state['logged_in'] = True
                st.session_state['user_name'] = name
                st.success('Logged in')
            else:
                st.error('No user found. Please register.')

        if st.button('Go to registration'):
            st.session_state['stage'] = 'register'

    elif st.session_state['stage'] == 'register':
        # Registration form
        st.subheader('Register')
        name = st.text_input('Name')
        age = st.number_input('Age', min_value=1, max_value=100)
        gender = st.selectbox('Gender', options=['Male', 'Female', 'Other'])

        if st.button('Register'):
            # Check if user already exists in the database
            c.execute('''
                SELECT * FROM userdata WHERE name = ? AND age = ?
            ''', (name, age))
            user = c.fetchone()
            if user is None:
                # User doesn't exist, register them
                c.execute('''
                    INSERT INTO userdata (name, age, gender) VALUES (?, ?, ?)
                ''', (name, age, gender))
                conn.commit()
                st.success('Registered. Please login.')
                st.session_state['stage'] = 'login'
            else:
                st.error('User already exists. Please login.')

        if st.button('Back to login'):
            st.session_state['stage'] = 'login'

else:
    st.write(f"Welcome, {st.session_state['user_name']}, hope you are well!! You can start the detection process by clicking on the start button")
    process = None  # Initialize the process variable

    if st.button("Start Facial Emotion Recognition"):
        if os.path.exists("stop_sentinel.txt"):
            os.remove("stop_sentinel.txt")  # Remove the sentinel file if it exists
        # Pass the username as a command-line argument

        process = subprocess.Popen([sys.executable,"plzchaljao2.py",st.session_state['user_name']], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

        # process = subprocess.Popen(["python", "C:\\Users\\HP\\PycharmProjects\\frontend_facialEmotionRecognition\\tes_graph_login.py", st.session_state['user_name']], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        st.write("Starting the process...")
        st.write("Thank you for trying our system...")

    if st.button("Stop Facial Emotion Recognition"):
        with open("stop_sentinel.txt", "w") as file:
            pass  # Create a sentinel file
        st.write("Stopping the process...")

    if st.button("Logout"):
        # Logout the user
        st.session_state['logged_in'] = False
        st.write("Logged out")
        # Clear the page
        st.empty()

    if st.button("Back"):
        # Go back to login/register page
        st.session_state['logged_in'] = False
        st.session_state['stage'] = 'login'
        st.write("Going back to login/register page")
        # Clear the page
        st.empty()
