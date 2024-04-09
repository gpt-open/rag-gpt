#!/bin/bash

# init SQLite DB
python create_sqlite_db.py

nohup gunicorn -c gunicorn_config.py smart_qa_app:app > /dev/null 2>&1 &
