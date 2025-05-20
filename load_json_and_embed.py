import sys
import json

path = sys.argv[1]

with open(path) as f:
    d = json.load(f)

from IPython import embed; embed();