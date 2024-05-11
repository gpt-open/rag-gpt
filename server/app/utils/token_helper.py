import jwt
import datetime
from typing import Union, Dict, Any
from jwt import ExpiredSignatureError, InvalidTokenError


class TokenHelper:
    JWT_SECRET = 'open_kf_2024'  # Should be replaced with a secure key
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_DELTA = datetime.timedelta(days=7)  # Token expiration time

    @staticmethod
    def generate_token(user_id: str) -> str:
        """
        Generate JWT token.

        Args:
            user_id (str): The unique identifier for the user.

        Returns:
            str: Encoded JWT token as a string.
        """
        payload = {
            'user_id': user_id,
            'exp': datetime.datetime.utcnow() + TokenHelper.JWT_EXPIRATION_DELTA
        }
        # jwt.encode returns a byte string, decode it to convert to a normal string
        return jwt.encode(payload, TokenHelper.JWT_SECRET, algorithm=TokenHelper.JWT_ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Union[Dict[str, Any], str]:
        """
        Verify JWT token.

        Args:
            token (str): The JWT token to verify.

        Returns:
            Union[Dict[str, Any], str]: The payload as a dictionary if valid, otherwise an error message string.
        """
        try:
            payload = jwt.decode(token, TokenHelper.JWT_SECRET, algorithms=[TokenHelper.JWT_ALGORITHM])
            return payload
        except ExpiredSignatureError:
            return 'Token expired'
        except InvalidTokenError:
            return 'Invalid token'
