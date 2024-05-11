import os
from dotenv import load_dotenv
from server.constant.env_constants import check_env_variables

# Load environment variables from .env file
load_dotenv(override=True)
check_env_variables()

from flask import Flask, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import safe_join
from server.app import account, auth, bot_config, common, files, intervention, queries, sitemaps, urls
from server.constant.constants import STATIC_DIR, MEDIA_DIR
from server.logger.logger_config import my_logger as logger


app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)


"""
Background:
In scenarios where using a dedicated static file server (like Nginx) is not feasible or desired, Flask can be configured to serve static files directly. This setup is particularly useful during development or in lightweight production environments where simplicity is preferred over the scalability provided by dedicated static file servers.

This Flask application demonstrates how to serve:
- Static media files from a dynamic path (`MEDIA_DIR`)
- The homepages and assets for two single-page applications (SPAs): 'open-kf-chatbot' and 'open-kf-admin'.

Note:
While Flask is capable of serving static files, it's not optimized for the high performance and efficiency of a dedicated static file server like Nginx, especially under heavy load. For large-scale production use cases, deploying a dedicated static file server is recommended.

The provided routes include a dynamic route for serving files from a specified media directory and specific routes for SPA entry points and assets. This configuration ensures that SPA routing works correctly without a separate web server.
"""
# Dynamically serve files from the MEDIA_DIR
@app.route(f'/{MEDIA_DIR}/<path:filename>')
def serve_media_file(filename):
    # Use safe_join to securely combine the static folder path and filename
    file_path = safe_join(app.static_folder, MEDIA_DIR, filename)

    # Check if the file exists and serve it if so
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(os.path.join(app.static_folder, MEDIA_DIR), filename)
    else:
        # Return a 404 error if the file does not exist
        return abort(404)


# Route for the homepage of the 'open-kf-chatbot' site
@app.route('/open-kf-chatbot', strict_slashes=False)
def index_chatbot():
    return send_from_directory(f'{app.static_folder}/open-kf-chatbot', 'index.html')


# Route for serving other static files of the 'open-kf-chatbot' application
@app.route('/open-kf-chatbot/<path:path>')
def serve_static_chatbot(path):
    return send_from_directory(f'{app.static_folder}/open-kf-chatbot', path)


# Route for the homepage of the 'open-kf-admin' site
@app.route('/open-kf-admin', strict_slashes=False)
def index_admin():
    return send_from_directory(f'{app.static_folder}/open-kf-admin', 'index.html')


# Route for serving other static files of the 'open-kf-admin' application
@app.route('/open-kf-admin/<path:path>')
def serve_static_admin(path):
    return send_from_directory(f'{app.static_folder}/open-kf-admin', path)


app.register_blueprint(account.account_bp)
app.register_blueprint(auth.auth_bp)
app.register_blueprint(bot_config.bot_config_bp)
app.register_blueprint(common.common_bp)
app.register_blueprint(files.files_bp)
app.register_blueprint(intervention.intervention_bp)
app.register_blueprint(queries.queries_bp)
app.register_blueprint(sitemaps.sitemaps_bp)
app.register_blueprint(urls.urls_bp)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7000)
