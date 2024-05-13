from flask import Blueprint, request
from server.logger.logger_config import my_logger as logger
from server.app.utils.token_helper import TokenHelper


auth_bp = Blueprint('auth', __name__, url_prefix='/open_kf_api/auth')


@auth_bp.route('/get_token', methods=['POST'])
def get_token():
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return {'retcode': -20000, 'message': 'user_id is required', 'data': {}}

    try:
        # generate token
        token = TokenHelper.generate_token(user_id)
        logger.success(f"Generate token: '{token}' with user_id: '{user_id}'")
        return {"retcode": 0, "message": "success", "data": {"token": token}}
    except Exception as e:
        logger.error(f"Generate token with user_id: '{user_id}' is failed, the exception is {e}")
        return {'retcode': -20001, 'message': str(e), 'data': {}}
