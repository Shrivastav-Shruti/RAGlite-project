"""
RAGLite Streamlit Application Entry Point
"""

import os
import sys
from pathlib import Path
import subprocess

# Increase inotify watch limit
def increase_inotify_limit():
    try:
        # Try to increase the limit using sysctl
        current_limit = int(subprocess.check_output(['cat', '/proc/sys/fs/inotify/max_user_watches']).decode().strip())
        if current_limit < 524288:  # If limit is less than desired
            subprocess.run(['sysctl', '-w', 'fs.inotify.max_user_watches=524288'], check=True)
    except Exception as e:
        print(f"Warning: Could not increase inotify limit: {e}")
        print("You might need to manually increase the limit or use --server.fileWatcherType none")

# Try to increase the limit
increase_inotify_limit()

# Add compatibility fixes for Python 3.13
import typing
if not hasattr(typing, "ParamSpec"):
    from typing_extensions import ParamSpec
    setattr(typing, "ParamSpec", ParamSpec)

# Change to RAG directory
os.chdir(os.path.join(os.path.dirname(__file__), 'RAG'))

# Import and run the main app
try:
    import streamlit as st
    
    # Set Streamlit configuration
    st.set_option('server.fileWatcherType', 'none')  # Disable file watcher if needed
    
    from raglite.api.app import RAGLiteApp
    
    # Initialize and run the app
    if __name__ == "__main__":
        app = RAGLiteApp()
        app.run()
except Exception as e:
    import streamlit as st
    st.error(f"Failed to initialize RAGLite: {str(e)}")
    st.info("Please check the logs for more details.") 