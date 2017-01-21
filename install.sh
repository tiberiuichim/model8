#!/bin/sh

virtualenv .
bin/pip install -r requirements.txt
bin/pip install -e .
