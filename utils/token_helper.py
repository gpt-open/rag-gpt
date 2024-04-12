# coding=utf-8
import jwt
import datetime

class TokenHelper:
    JWT_SECRET = 'open_kf_2024'  # Should be replaced with a secure key
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_DELTA = datetime.timedelta(days=7)  # Token expiration time

    @staticmethod
    def generate_token(user_id):
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.datetime.utcnow() + TokenHelper.JWT_EXPIRATION_DELTA
        }
        return jwt.encode(payload, TokenHelper.JWT_SECRET, algorithm=TokenHelper.JWT_ALGORITHM)

    @staticmethod
    def verify_token(token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, TokenHelper.JWT_SECRET, algorithms=[TokenHelper.JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return 'Token expired'
        except jwt.InvalidTokenError:
            return 'Invalid token'
