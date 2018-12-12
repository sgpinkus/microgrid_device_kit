export PYTHONPATH=$PYTHONPATH${PYTHONPATH:+:}$(dirname $(dirname $(readlink -f "$0")))
