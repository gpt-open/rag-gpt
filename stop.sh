#!/bin/bash

ps -ef | grep "smart_qa_app:app" | grep -v grep | awk '{print $2}' | xargs kill -9
