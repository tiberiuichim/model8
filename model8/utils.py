""" Generic utils
"""

import fcntl
from contextlib import contextmanager
import os.path
import time


@contextmanager
def folder_lock(path):
    """ A generic context manager that locks the destination folder for
    exclusive writing in it.

    Because we write multiple files in that location, but the writing/checks
    can happen out of order, instead of locking single files, we lock the whole
    destination folder.
    """

    lpath = os.path.join(path, '.lock')
    if os.path.exists(lpath):   # TODO: use real locking
        raise ValueError("Path is already locked")
    lockfile = open(lpath, 'w+')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    yield
    time.sleep(500)
    fcntl.flock(lockfile, fcntl.LOCK_UN)
    os.remove(lpath)
