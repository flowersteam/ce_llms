import sys
import datasets
from collections import Counter
path = sys.argv[1]
d = datasets.load_from_disk(path)
from IPython import embed;
embed();
