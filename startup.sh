#!/bin/bash
set -e
echo "creating venv###############################################################"
python -m venv antenv
echo "activating venv #################################################################"
source antenv/bin/activate
echo "Installing requirements#################################################################"
pip install -r requirements.txt
echo "executing Gunicorn #########################################################################"
exec gunicorn app:app
