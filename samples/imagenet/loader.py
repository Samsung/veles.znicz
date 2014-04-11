"""
Created on Apr 10, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import cv2
import jpeg4py
import json
import leveldb
import numpy
from progressbar import ProgressBar
import struct
import os
import xmltodict

import veles.config as config
import veles.opencl_types as opencl_types
import veles.znicz.loader as loader


class Loader(loader.Loader):
    """
    Imagenet images and metadata loader.
    """

    MAPPING = {
        "train": {
            "2013": {
                "img": ("ILSVRC2012_img_train", "ILSVRC2012_bbox_train_v2"),
                "DET": ("ILSVRC2013_DET_train", "ILSVRC2013_DET_bbox_train"),
            },
        },
        "validation": {
            "2013": {
                "img": ("ILSVRC2012_img_val", "ILSVRC2012_bbox_val_v3"),
                "DET": ("ILSVRC2013_DET_val", "ILSVRC2013_DET_bbox_val"),
            },
        },
        "test": {
            "2013": {
                "img": ("ILSVRC2012_img_test", ""),
                "DET": ("ILSVRC2013_DET_test", ""),
            },
        }
    }

    def __init__(self, workflow, **kwargs):
        self._dbpath = kwargs.get("dbpath", config.root.imagenet.dbpath)
        super(Loader, self).__init__(workflow, **kwargs)
        self._ipath = kwargs.get("ipath", config.root.imagenet.ipath)
        self._year = kwargs.get("year", config.root.imagenet.year)
        self._series = kwargs.get("series", config.root.imagenet.series)
        self._data_shape = kwargs.get("data_shape",
                                      config.root.imagenet.shape or (256, 256))
        self._dtype = opencl_types.dtypes[config.root.common.precision_type]
        self._colorspace = kwargs.get("colorspace",
                                      config.root.imagenet.colorspace or "RGB")
        self._include_derivative = kwargs.get(
            "derivative", config.root.imagenet.derivative or False)
        self._sobel_kernel_size = kwargs.get(
            "sobel_kernel_size", config.root.imagenet.sobel_ksize or 5)

    def init_unpickled(self):
        super(Loader, self).init_unpickled()
        self._db_ = leveldb.LevelDB(self._dbpath)

    @property
    def images_path(self):
        return self._ipath

    @property
    def db_path(self):
        return self._dbpath

    @property
    def year(self):
        return self._year

    @property
    def series(self):
        return self._series

    def load_data(self):
        self._init_files()
        self._init_metadata()
        self._fill_class_samples()

    def create_minibatches(self):
        count = self.minibatch_maxsize
        minibatch_shape = [count, 3] + list(self._data_shape)
        self.minibatch_data << numpy.zeros(shape=minibatch_shape,
                                           dtype=self._dtype)
        self.minibatch_labels << numpy.zeros(count, dtype=numpy.int32)
        self.minibatch_indexes << numpy.zeros(count, dtype=numpy.int32)

    def fill_minibatch(self):
        pass

    def _get_file_name(self, index):
        for i in range(len(self._files_locator) - 1):
            left_index, files, set_name = self._files_locator[i]
            right_index = self._files_locator[i + 1][0]
            if left_index < index < right_index:
                mapping = Loader.MAPPING[set_name][self.year][self.series]
                return os.path.join(self._ipath, mapping[0],
                                    files[index - left_index]) + ".JPEG"

    def _decode_image(self, index):
        file_name = self._get_file_name(index)
        try:
            data = jpeg4py.JPEG(file_name).decode()
        except:
            self.exception("Failed to decode %s", file_name)
        if self._include_derivative:
            deriv = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            deriv = cv2.Sobel(
                deriv,
                cv2.CV_32F if self._dtype == numpy.float32 else cv2.CV_64F, 1,
                1, ksize=self._sobel_kernel_size)
        if self._colorspace == "HSV":
            cv2.cvtColor(data, cv2.COLOR_RGB2HSV, data)
        if self._include_derivative:
            shape = list(data.shape)
            shape[2] += 1
            res = numpy.empty(shape, dtype=self._dtype)
            res[:, :, :-1] = data[:, :, :]
            res.ravel()[3::4] = deriv.ravel()
        else:
            res = data.astype(self._dtype)
        return res

    def _img_file_name(self, base, full):
        res = full[len(os.path.commonprefix([base, full])):]
        res = os.path.splitext(res)[0]
        while (res[0] == os.sep):
            res = res[1:]
        parts = res.split(os.sep)
        if len(parts) >= 2 and parts[0] == parts[1]:
            res = os.sep.join(parts[1:])
        return res

    def _init_files(self):
        self.debug("Initializing files table...")
        files_key = ("files_%s_%s" % (self.year, self.series)).encode()
        try:
            files = self._db_.Get(files_key)
            self.info("Loaded files table from DB")
            self._files = json.loads(files.decode())
        except KeyError:
            self.debug("Will look for images in %s", self._ipath)
            self._files = {}
            index = 0
            for set_name, years in Loader.MAPPING.items():
                imgs = []
                subdir = years[self.year][self.series][0]
                path = os.path.join(self._ipath, subdir)
                self.debug("Scanning %s...", path)
                for root, _, files in os.walk(path, followlinks=True):
                    imgs.extend([self._img_file_name(path,
                                                     os.path.join(root, f))
                                 for f in files
                                 if os.path.splitext(f)[1] == ".JPEG" and
                                 f.find("-256") < 0])
                self._files[set_name] = (imgs, index)
                index += len(imgs)
            self.debug("Saving files table to DB...")
            self._db_.Put(files_key, json.dumps(self._files).encode())
            self.info("Initialized files table")
        self._files_locator = sorted([(files[1], files[0], set_name)
                                      for set_name, files
                                      in self._files.items()])
        self._files_locator.append((self._files_locator[-1][0] +
                                    len(self._files_locator[-1][1]),
                                    None, None))

    def _gen_img_key(self, index):
        return struct.pack("I", index) + self.year.encode() + \
            self.series.encode()

    def _get_meta(self, index):
        return json.loads(self._db_.Get(self._gen_img_key(index)).decode())

    def _set_meta(self, index, value):
        self._db_.Put(self._gen_img_key(index), json.dumps(value).encode())

    def _init_metadata(self):
        self.debug("Initializing metadata...")
        metadata_key = ("metadata_%s_%s" % (self.year, self.series)).encode()
        try:
            self._db_.Get(metadata_key)
            self.info("Found metadata in DB")
            return
        except KeyError:
            pass
        self.debug("Will look for metadata in %s", self._ipath)
        all_xmls = {}
        for set_name, years in Loader.MAPPING.items():
            all_xmls[set_name] = xmls = []
            subdir = years[self.year][self.series][1]
            if not subdir:
                continue
            path = os.path.join(self._ipath, subdir)
            self.debug("Scanning %s...", path)
            for root, _, files in os.walk(path, followlinks=True):
                xmls.extend([os.path.join(root, f)
                             for f in files
                             if os.path.splitext(f)[1] == ".xml"])
        self.debug("Building image indices mapping")
        ifntbl = {}
        for set_name, files in self._files.items():
            flist = files[0]
            base = files[1]
            table = {}
            for i in range(len(flist)):
                table[flist[i]] = i + base
            if len(table) < len(flist):
                self.error("Duplicate file names detected in %s (%s, %s)",
                           set_name, self.year, self.series)
            ifntbl[set_name] = table
        self.debug("Parsing XML files...")
        progress = ProgressBar(maxval=sum(
            [len(xmls) for xmls in all_xmls.values()]))
        progress.start()
        for set_name, xmls in all_xmls.items():
            for xml in xmls:
                progress.inc()
                with open(xml, "r") as fr:
                    tree = xmltodict.parse(fr.read())
                del tree["annotation"]["folder"]
                del tree["annotation"]["filename"]
                file_key = self._img_file_name(os.path.join(
                    self._ipath,
                    Loader.MAPPING[set_name][self.year][self.series][1]), xml)
                try:
                    index = ifntbl[set_name][file_key]
                except KeyError:
                    self.error(
                        "%s references unexistent file %s", xml, os.path.join(
                            self._ipath,
                            Loader.MAPPING[set_name][self.year]
                            [self.series][0], file_key))
                    continue
                self._set_meta(index, tree["annotation"])
        progress.finish()
        self._db_.Put(metadata_key, b"")
        self.info("Initialized metadata")

    def _fill_class_samples(self):
        triage = {"train": 2, "validation": 1, "test": 0}
        for key, val in triage.items():
            self.class_samples[val] = len(self._files[key][0])
