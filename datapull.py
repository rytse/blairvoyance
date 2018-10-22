import os
import errno
import urllib.request

FILES = ['poll_input.csv', 'demographics.csv', 'district_input.csv']
START = 'https://raw.githubusercontent.com/polistat/polistat-data/master/'

try:
    os.makedirs(os.path.dirname('./data/'))
except OSError as exc:  # Guard against race condition
    if exc.errno != errno.EEXIST:
        raise

for f in FILES:
    urllib.request.urlretrieve(START + f, './data/' + f)