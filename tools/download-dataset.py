from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tarfile
import tempfile
import shutil

try:
    # noinspection PyCompatibility
    from urllib.request import urlopen  # python 3
except ImportError:
    from urllib2 import urlopen  # python 2

dataset = sys.argv[1]
url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz" % dataset
with tempfile.TemporaryFile() as tmp:
    print("downloading", url)
    shutil.copyfileobj(urlopen(url), tmp)
    print("extracting")
    tmp.seek(0)
    tar = tarfile.open(fileobj=tmp)
    tar.extractall()
    tar.close()
    print("done")
