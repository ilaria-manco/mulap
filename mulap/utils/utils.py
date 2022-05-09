import os
import sys
import json
import shutil
import logging
import requests
import tempfile
import numpy as np
from io import open
from tqdm import tqdm
from pathlib import Path
from hashlib import sha256
from omegaconf import OmegaConf
from urllib.parse import urlparse

import torch.functional as F


PYTORCH_PRETRAINED_BERT_CACHE = Path(
    os.getenv("PYTORCH_PRETRAINED_BERT_CACHE",
              Path.home() / ".pytorch_pretrained_bert")
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def load_conf(path_to_yaml):
    """ Wrapper for configuration file loading through OmegaConf. """
    conf = OmegaConf.load(path_to_yaml)
    # this should be training.base_dir
    if "env" in conf.keys() and conf.env.base_dir is None:
        OmegaConf.update(conf, "env.base_dir", get_root_dir())
    return conf


def merge_conf(base_conf_path, dataset_conf_path, model_conf_path):
    """ Wrapper for to merge multiple config files through OmegaConf. """
    base_conf = load_conf(base_conf_path)
    dataset_conf = load_conf(dataset_conf_path)
    model_conf = load_conf(model_conf_path)

    conf = OmegaConf.merge(
        base_conf, dataset_conf, model_conf)
    return conf


def save_json(output_path, content):
    with open(output_path, 'w') as outfile:
        json.dump(content, outfile)


def get_root_dir():
    root = os.path.dirname(
        os.path.abspath(__file__))
    root = os.path.abspath(
        os.path.join(root, "../.."))
    return root


def normalize(x, p, dim):
    norm = F.norm(x, p=p, dim=dim)
    return x / norm


def scale(x, axis=0):
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    scale = std[std == 0.0] = 1.0
    x -= mean
    x /= scale
    return np.array(x, dtype=np.float32)


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError(
                "HEAD request failed for url {} with status code {}".format(
                    url, response.status_code
                )
            )
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s",
                        url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s",
                        temp_file.name, cache_path)
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def cached_path(url_or_filename, cache_dir=None):
    """
    Code from https://github.com/facebookresearch/vilbert-multi-task/blob/master/vilbert/utils.py.

    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https", "s3"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(
                url_or_filename)
        )
