import sys
import datasets
import numpy as np
from collections import Counter
path = sys.argv[1]
d = datasets.load_from_disk(path)
print(d)
from IPython import embed;
embed();
