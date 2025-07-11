#!/bin/bash
python -m venv antenv
source antenv/bin/activate
pip install -r requirements.txt
exec gunicorn app:app
