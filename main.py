from threading import Thread
import os

def run_streamlit() :
    os.system('streamlit run app.py')

thread = Thread(target = run_streamlit)
thread.start()