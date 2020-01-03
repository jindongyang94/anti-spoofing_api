#!/usr/bin/env bash

# # Install custom python package if requirements.txt is present
# if [ -e "/requirements.txt" ]; then
#     $(command -v pip) install --user -r /requirements.txt
# fi

case "$1" in
    gunicorn)
        exec gunicorn -c config/gunicorn.conf.py src:app
        ;;
    *)
        # The command is something like bash, not an airflow subcommand. Just run it in the right environment.
        exec "$@"
        ;;
esac
