import re
import py3langid as langid
from server.logger.logger_config import my_logger as logger


def detect_query_lang(query: str) -> str:
    # Dictionary to map language codes to full language names
    lang_map = {
        'en': 'English',
        'zh': 'Chinese',
        'fr': 'French',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'de': 'German',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'hi': 'Hindi',
        'ar': 'Arabic'
    }

    # Detect the language of the query
    lang, _ = langid.classify(query)

    # Get the full language name
    full_language = lang_map.get(lang, 'English')
    return full_language


def query_rewrite(query: str, bot_topic: str) -> str:
    # Detect the language of the query
    lang, _ = langid.classify(query)
    
    # Convert to lowercase for case-insensitive comparison
    query_lower = query.lower()
    bot_topic_lower = bot_topic.lower()
    
    # Check if the bot_topic is already included in the query
    if bot_topic_lower not in query_lower:
        # Using regular expression to remove trailing punctuation if present
        query_trimmed = re.sub(r'[?!.，。？！,]*$', '', query)

        # Language-specific adjustments for more colloquial expressions
        if lang == 'en':
            adjust_query = f"{query_trimmed}, about '{bot_topic}'"
        elif lang == 'zh':
            adjust_query = f"{query_trimmed}，关于'{bot_topic}'的信息"
        elif lang == 'fr':
            adjust_query = f"{query_trimmed}, à propos de '{bot_topic}'"
        elif lang == 'es':
            adjust_query = f"{query_trimmed}, sobre '{bot_topic}'"
        elif lang == 'pt':
            adjust_query = f"{query_trimmed}, sobre '{bot_topic}'"
        elif lang == 'de':
            adjust_query = f"{query_trimmed}, über '{bot_topic}'"
        elif lang == 'ru':
            adjust_query = f"{query_trimmed}, о '{bot_topic}'"
        elif lang == 'ja':
            adjust_query = f"{query_trimmed}、'{bot_topic}'について"
        elif lang == 'ko':
            adjust_query = f"{query_trimmed}，'{bot_topic}'에 대해"
        elif lang == 'hi':
            adjust_query = f"{query_trimmed}, '{bot_topic}' के बारे में"
        elif lang == 'ar':
            adjust_query = f"{query_trimmed}، حول '{bot_topic}'"
        else:
            adjust_query = f"{query_trimmed}, '{bot_topic}'"
        
        # Record the adjusted query
        logger.warning(f"Detected language: {lang}, Original query: '{query}', Adjusted query: '{adjust_query}'")
        return adjust_query
    else:
        # If the bot_topic is already in the query, log this info and return the original query
        logger.info(f"No adjustment needed for query: '{query}'")
        return query
