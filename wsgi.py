# WSGI entrypoint for production (gunicorn / Clever Cloud)
# This file imports the Flask app and starts the background tasks.
# Gunicorn will look for the "application" WSGI variable below.

from Hidden_bot import app, start_background_tasks

# Start background threads once when the WSGI module is imported by gunicorn
start_background_tasks()

# Expose the WSGI application callable
application = app