from datetime import datetime
import os
import uuid
from flask import Blueprint, Flask, request
from werkzeug.utils import secure_filename
from server.constant.constants import STATIC_DIR, MEDIA_DIR
from server.app.utils.decorators import token_required
from server.logger.logger_config import my_logger as logger


URL_PREFIX = os.getenv('URL_PREFIX')

common_bp = Blueprint('common', __name__, url_prefix='/open_kf_api/common')


@common_bp.route('/upload_picture', methods=['POST'])
@token_required
def upload_picture():
    picture_file = request.files.get('picture_file')
    if not picture_file:
        logger.error("Missing required parameters picture_file")
        return {'retcode': -20000, 'message': 'Missing required parameters picture_file', data:{}}

    try:
        original_filename = secure_filename(picture_file.filename)

        day_folder = datetime.now().strftime("%Y_%m_%d")
        unique_folder = str(uuid.uuid4())
        save_directory = os.path.join(STATIC_DIR, MEDIA_DIR, day_folder, unique_folder)
        os.makedirs(save_directory, exist_ok=True)

        image_path = os.path.join(save_directory, original_filename)
        picture_file.save(image_path)
        picture_url  = f"{URL_PREFIX}{MEDIA_DIR}/{day_folder}/{unique_folder}/{original_filename}"
        return {'retcode': 0, 'message': 'upload picture success', 'data': {'picture_url': picture_url}}
    except Exception as e:
        logger.error(f"An error occureed: {str(e)}")
        return {'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}}
