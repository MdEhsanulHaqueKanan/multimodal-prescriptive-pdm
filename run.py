"""
Main entry point to run the Flask web application.
"""
from app import app

if __name__ == '__main__':
    # Using port 5000 for local development. Render will use the port from the CMD.
    app.run(host='0.0.0.0', port=5000, debug=False)