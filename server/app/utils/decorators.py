from functools import wraps
from typing import Callable, Any, Dict, Tuple, Union
from flask import request, Response
from server.app.utils.token_helper import TokenHelper
from server.logger.logger_config import my_logger as logger


# Define the type for the decorator's inner function
DecoratorFunction = Callable[..., Union[Dict[str, Any], Tuple[Dict[str, Any], int]]]


def token_required(f: DecoratorFunction) -> DecoratorFunction:
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int], Response]:
        token: Union[str, None] = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]

        if not token:
            logger.error("Token is missing!")
            return {'retcode': -10000, 'message': 'Token is missing!', 'data': {}}, 401

        try:
            user_payload = TokenHelper.verify_token(token)
            if user_payload == 'Token expired':
                logger.error(f"Token: '{token}' is expired!")
                return {'retcode': -10001, 'message': 'Token is expired!', 'data': {}}, 401
            elif user_payload == 'Invalid token':
                logger.error(f"Token: '{token}' is invalid")
                return {'retcode': -10002, 'message': 'Token is invalid!', 'data': {}}, 401
            request.user_payload = user_payload  # Store payload in request for further use
        except Exception as e:
            logger.error(f"Token: '{token}' is invalid, the exception is {e}")
            return {'retcode': -10003, 'message': 'Token is invalid!', 'data': {}}, 401

        return f(*args, **kwargs)
    return decorated_function
