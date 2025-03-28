import sys
import json
from collections import Counter

path = sys.argv[1]

with open(path) as f:
    j = json.load(f)

from IPython import embed; embed();