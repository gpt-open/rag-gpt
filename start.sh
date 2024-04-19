#!/bin/bash

# init SQLite DB
python create_sqlite_db.py

gunicorn -c gunicorn_config.py rag_gpt_app:app 
