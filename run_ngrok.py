import os
from pyngrok import ngrok, conf
import subprocess
import atexit
import sys

# --- Your Streamlit App Details ---
STREAMLIT_APP_FILE = "salary_app_dual.py"
STREAMLIT_PORT = 8501

def run_with_ngrok():
    """
    Launches the Streamlit app and exposes it to the web using ngrok.
    Handles ngrok authtoken configuration robustly.
    """
    # --- Robust Authtoken Handling ---
    authtoken = os.environ.get("NGROK_AUTHTOKEN")
    if authtoken is None:
        print("--- Ngrok Authtoken Setup ---")
        authtoken = input("Please enter your ngrok authtoken (from https://dashboard.ngrok.com/get-started/your-authtoken): ")

    ngrok.set_auth_token(authtoken)
    
    # Ensure ngrok processes are killed upon exit
    atexit.register(ngrok.kill)

    # Open a tunnel to the Streamlit port
    print(f"\nStarting ngrok tunnel for Streamlit app on port {STREAMLIT_PORT}...")
    public_url = ngrok.connect(STREAMLIT_PORT)
    print("=" * 50)
    print(f"🚀 Your Streamlit app is live at: {public_url}")
    print("=" * 50)

    # Start the Streamlit app using subprocess for better control
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", STREAMLIT_APP_FILE, "--server.port", str(STREAMLIT_PORT)], check=True)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped. Shutting down ngrok tunnel.")
    except Exception as e:
        print(f"An error occurred while running Streamlit: {e}")
    finally:
        ngrok.kill()

def run_local_only():
    """
    Launches the Streamlit app for local access only.
    """
    print(f"Starting Streamlit app locally on port {STREAMLIT_PORT}...")
    print("You can view your app in your browser.")
    print(f"Local URL: http://localhost:{STREAMLIT_PORT}")
    print("Press Ctrl+C to stop the app.")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", STREAMLIT_APP_FILE, "--server.port", str(STREAMLIT_PORT)], check=True)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")
    except Exception as e:
        print(f"An error occurred while running Streamlit: {e}")

if __name__ == "__main__":
    print("Choose how to run the application:")
    print("1: Local Only (access via http://localhost:8501)")
    print("2: Public via Ngrok (get a shareable public URL)")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        run_local_only()
    elif choice == '2':
        run_with_ngrok()
    else:
        print("Invalid choice. Please run the script again and enter 1 or 2.")
