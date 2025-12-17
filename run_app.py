"""
Run Streamlit app
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit application"""
    app_path = Path(__file__).parent / "app" / "main.py"
    
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting Anime Recommendation System...")
    print(f"📂 App path: {app_path}")
    print("\n" + "=" * 70)
    print("The app will open in your browser at: http://localhost:8501")
    print("=" * 70 + "\n")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port=8501",
        "--server.headless=false"
    ])

if __name__ == "__main__":
    main()