import hdfs
import os
import random

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse
import gzip


def decision(probability):
    return random.random() < probability


def sampel(list, count):
    indices = random.sample(range(len(list)), count)
    return [list[i] for i in sorted(indices)]


def read(path):
    if path.startswith('http://'):
        client = hdfs.client.Client("http://" + urlparse(path).netloc)
        reader = client.read(urlparse(path).path, encoding='utf-8')
        return reader
    else:
        return open(path, 'r')


def list(path):
    if path.startswith('http://'):
        client = hdfs.client.Client("http://" + urlparse(path).netloc)
        reader = client.list(urlparse(path).path)
        return reader
    else:
        return os.listdir(path)


def loadGZ(path):
    client = hdfs.client.Client("http://" + urlparse(path).netloc)
    with client.read(urlparse(path).path) as r:
        buf = StringIO.StringIO(r.read())
        gzip_f = gzip.GzipFile(fileobj=buf)
        content = gzip_f.read()
        return content
