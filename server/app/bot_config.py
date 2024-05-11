import json
import time
from flask import Blueprint, Flask, request
from server.app.utils.decorators import token_required
from server.app.utils.sqlite_client import get_db_connection
from server.app.utils.diskcache_client import diskcache_client
from server.app.utils.diskcache_lock import diskcache_lock
from server.logger.logger_config import my_logger as logger


bot_config_bp = Blueprint('bot_config', __name__, url_prefix='/open_kf_api/bot_config')


@bot_config_bp.route('/get_bot_setting', methods=['POST'])
def get_bot_setting():
    """Retrieve bot setting, first trying Cache and falling back to DB if not found."""
    try:
        # Attempt to retrieve the setting from Cache
        key = "open_kf:bot_setting"
        setting_cache = diskcache_client.get(key)
        if setting_cache:
            setting_data = json.loads(setting_cache)
            return {'retcode': 0, 'message': 'Success', 'data': {'config': setting_data}}
        else:
            logger.warning(f"could not find '{key}' in Cache!")
    except Exception as e:
        logger.error(f"Error retrieving setting from Cache, excpetion:{e}")
        # Just ignore Cache error
        #return {'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}}

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('SELECT * FROM t_bot_setting_tab LIMIT 1')
        setting = cur.fetchone()
        if setting:
            setting = dict(setting)
            # Process and return the setting details
            setting_data = {k: json.loads(v) if k in ['initial_messages', 'suggested_messages'] else v for k, v in setting.items()}

            # Add bot setting into Cache
            try:
                key = "open_kf:bot_setting"
                diskcache_client.set(key, json.dumps(setting_data))
            except Exception as e:
                logger.error(f"Add bot setting into Cache is failed, the exception is {e}")
                # Just ignore Reids error

            return {'retcode': 0, 'message': 'Success', 'data': {'config': setting_data}}
        else:
            logger.warning(f"No setting found")
            return {'retcode': -20001, 'message': 'No setting found', 'data': {}}
    except Exception as e:
        logger.error(f"Error retrieving setting: {e}")
        return {'retcode': -30000, 'message': f'An error occurred: {e}', 'data': {}}
    finally:
        if conn:
            conn.close()


@bot_config_bp.route('/update_bot_setting', methods=['POST'])
@token_required
def update_bot_setting():
    data = request.json
    # Extract and validate all required fields
    setting_id = data.get('id')
    initial_messages = data.get('initial_messages')
    suggested_messages = data.get('suggested_messages')
    bot_name = data.get('bot_name')
    bot_avatar = data.get('bot_avatar')
    chat_icon = data.get('chat_icon')
    placeholder = data.get('placeholder')
    model = data.get('model')

    # Check for the presence of all required fields
    if None in (setting_id, initial_messages, suggested_messages, bot_name, bot_avatar, chat_icon, placeholder, model):
        return {'retcode': -20000, 'message': 'All fields are required', 'data': {}}

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if the setting with provided ID exists
        cur.execute('SELECT id FROM t_bot_setting_tab WHERE id = ?', (setting_id,))
        if not cur.fetchone():
            logger.error(f"No setting found")
            return {'retcode': -20001, 'message': 'Setting not found', 'data': {}}

        # Convert lists to JSON strings for storage
        initial_messages_json = json.dumps(initial_messages)
        suggested_messages_json = json.dumps(suggested_messages)

        # Update bot setting in DB
        timestamp = int(time.time())
        try:
            with diskcache_lock.lock():
                cur.execute('''
                    UPDATE t_bot_setting_tab
                    SET initial_messages = ?, suggested_messages = ?, bot_name = ?, bot_avatar = ?, chat_icon = ?, placeholder = ?, model = ?, mtime = ?
                    WHERE id = ?
                ''', (initial_messages_json, suggested_messages_json, bot_name, bot_avatar, chat_icon, placeholder, model, timestamp, setting_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Process discache_lock exception: {e}")
            return {'retcode': -30000, 'message': f'An error occurred: {e}', 'data': {}}

        # Update bot setting in Cache
        try:
            key = "open_kf:bot_setting"
            bot_setting = {
                'id': setting_id,
                'initial_messages': initial_messages,
                'suggested_messages': suggested_messages,
                'bot_name': bot_name,
                'bot_avatar': bot_avatar,
                'chat_icon': chat_icon,
                'placeholder': placeholder,
                'model': model,
                'ctime': timestamp,
                'mtime': timestamp
            }
            diskcache_client.set(key, json.dumps(bot_setting))
        except Exception as e:
            logger.error(f"Ppdate bot seeting in Cache is failed, the exception is {e}")
            return {'retcode': -20001, 'message': f'An error occurred: {e}', 'data': {}}

        return {'retcode': 0, 'message': 'Settings updated successfully', 'data': {}}
    except Exception as e:
        logger.error(f"Error updating setting in DB: {e}")
        return {'retcode': -30000, 'message': f'An error occurred: {e}', 'data': {}}
    finally:
        if conn:
            conn.close()
