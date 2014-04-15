"""
Created on Apr 10, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


from concurrent.futures import ThreadPoolExecutor
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
from veles.external.prettytable import PrettyTable
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
        aperture = kwargs.get("aperture",
                              config.get(config.root.imagenet.aperture) or 256)
        self._data_shape = (aperture, aperture)
        self._dtype = opencl_types.dtypes[config.root.common.precision_type]
        self._crop_color = kwargs.get(
            "crop_color",
            config.get(config.root.imagenet.crop_color) or (127, 127, 127))
        self._colorspace = kwargs.get(
            "colorspace", config.get(config.root.imagenet.colorspace) or "RGB")
        self._include_derivative = kwargs.get(
            "derivative", config.get(config.root.imagenet.derivative) or False)
        self._sobel_kernel_size = kwargs.get(
            "sobel_kernel_size",
            config.get(config.root.imagenet.sobel_ksize) or 5)
        self._force_reinit = kwargs.get(
            "force_reinit",
            config.get(config.root.imagenet.force_reinit) or False)

    def init_unpickled(self):
        super(Loader, self).init_unpickled()
        self._db_ = leveldb.LevelDB(self._dbpath)
        self._executor_ = ThreadPoolExecutor(
            config.get(config.root.imagenet.thread_pool_size) or 4)

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
        sample_sets, cats = self._init_samples_categories(None, None)
        if sample_sets is not None and cats is not None:
            self._label_map = self._init_labels(cats)
            self._fill_class_samples(sample_sets)
            return
        file_sets = self._init_files()
        metadata = self._init_metadata(file_sets)
        sample_sets, cats = self._init_samples_categories(file_sets, metadata)
        self._label_map = self._init_labels(cats)
        self._fill_class_samples(sample_sets)

    def create_minibatches(self):
        count = self.minibatch_maxsize
        minibatch_shape = [count] + list(self._data_shape) + \
            [3 + (1 if self._include_derivative else 0)]
        self.minibatch_data << numpy.zeros(shape=minibatch_shape,
                                           dtype=self._dtype)
        self.minibatch_labels << numpy.zeros(count, dtype=numpy.int32)
        self.minibatch_indexes << numpy.zeros(count, dtype=numpy.int32)

    def fill_minibatch(self):
        images = self._executor_.map(
            lambda i: (i, self._get_sample(self.shuffled_indexes[i])),
            range(self.minibatch_size))
        for i, data in images:
            self.minibatch_data[i] = data
        for i in range(self.minibatch_size):
            meta = self._get_meta(self.shuffled_indexes[i])[1]
            if meta[2] is not None:
                name = meta["name"]
            else:
                fn = self._get_file_name(self.shuffled_indexes[i])
                name = os.path.basename(os.path.dirname(fn))
            self.minibatch_labels[i] = self._label_map[name]

    def _get_file_name(self, index):
        meta = self._get_meta(index)
        set_name = loader.CLASS_NAME[meta[1]]
        mapping = Loader.MAPPING[set_name][self.year][self.series]
        return os.path.join(self._ipath, mapping[0], meta[0]) + ".JPEG"

    def _get_bbox(self, meta):
        bbox = meta[2]["bndbox"]
        return (int(bbox["xmin"]), int(bbox["ymin"]),
                int(bbox["xmax"]), int(bbox["ymax"]))

    def _decode_image(self, index):
        file_name = self._get_file_name(index)
        try:
            data = jpeg4py.JPEG(file_name).decode()
        except:
            self.exception("Failed to decode %s", file_name)
            raise
        return data

    def _crop_and_scale(self, img, index):
        width = img.shape[1]
        height = img.shape[0]
        meta = self._get_meta(index)
        if meta[2] is not None:
            bbox = list(self._get_bbox(meta))
        else:
            # No bbox found: crop the squared area and resize it
            offset = (width - height) / 2
            if offset > 0:
                img = img[:, offset:(width - offset), :]
            else:
                img = img[offset:(height - offset), :, :]
            img = cv2.resize(img, self._data_shape,
                             interpolation=cv2.INTER_AREA)
            return img
        # Check if the specified bbox is a square
        offset = (bbox[2] - bbox[0] - (bbox[3] - bbox[1])) / 2
        if offset > 0:
            # Width is bigger than height
            bbox[1] -= numpy.floor(offset)
            bbox[3] += numpy.ceil(offset)
            bottom_height = -bbox[1]
            if bottom_height > 0:
                bbox[1] = 0
            else:
                bottom_height = 0
            top_height = bbox[3] - height
            if top_height > 0:
                bbox[3] = height
            else:
                top_height = 0
            img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            if bottom_height > 0:
                fixup = numpy.empty((bottom_height, bbox[2] - bbox[0], 3),
                                    dtype=img.dtype)
                fixup[:, :, :] = self._crop_color
                img = numpy.concatenate((fixup, img), axis=0)
            if top_height > 0:
                fixup = numpy.empty((top_height, bbox[2] - bbox[0], 3),
                                    dtype=img.dtype)
                fixup[:, :, :] = self._crop_color
                img = numpy.concatenate((img, fixup), axis=0)
        elif offset < 0:
            # Height is bigger than width
            bbox[0] -= numpy.floor(offset)
            bbox[2] += numpy.ceil(offset)
            left_width = -bbox[0]
            if left_width > 0:
                bbox[0] = 0
            else:
                left_width = 0
            right_width = bbox[2] - width
            if right_width > 0:
                bbox[2] = width
            else:
                right_width = 0
            img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            if left_width > 0:
                fixup = numpy.empty((bbox[3] - bbox[1], left_width, 3),
                                    dtype=img.dtype)
                fixup[:, :, :] = self._crop_color
                img = numpy.concatenate((fixup, img), axis=1)
            if right_width > 0:
                fixup = numpy.empty((bbox[3] - bbox[1], right_width, 3),
                                    dtype=img.dtype)
                fixup[:, :, :] = self._crop_color
                img = numpy.concatenate((img, fixup), axis=1)
        else:
            img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        assert img.shape[0] == img.shape[1]
        if img.shape[0] != self._data_shape[0]:
            img = cv2.resize(img, self._data_shape,
                             interpolation=cv2.INTER_AREA)
        return img

    def _preprocess_sample(self, data):
        if self._include_derivative:
            deriv = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            deriv = cv2.Sobel(deriv,
                              cv2.CV_32F if self._dtype == numpy.float32
                              else cv2.CV_64F,
                              1, 1, ksize=self._sobel_kernel_size)
        if self._colorspace != "RGB":
            cv2.cvtColor(data, getattr(cv2, "COLOR_RGB2" + self._colorspace),
                         data)
        if self._include_derivative:
            shape = list(data.shape)
            shape[-1] += 1
            res = numpy.empty(shape, dtype=self._dtype)
            res[:, :, :-1] = data[:, :, :]
            begindex = len(shape)
            res.ravel()[begindex::(begindex + 1)] = deriv.ravel()
        else:
            res = data.astype(self._dtype)
        return res

    def _get_sample(self, index):
        data = self._decode_image(index)
        data = self._crop_and_scale(data, index)
        data = self._preprocess_sample(data)
        return data

    def _img_file_name(self, base, full):
        res = full[len(os.path.commonprefix([base, full])):]
        res = os.path.splitext(res)[0]
        while (res[0] == os.sep):
            res = res[1:]
        return res

    def _fixup_duplicate_dirs(self, path):
        parts = path.split(os.sep)
        if len(parts) >= 2 and parts[0] == parts[1]:
            res = os.sep.join(parts[1:])
            return res
        return path

    def _gen_sample_key(self, index):
        return struct.pack("I", index) + self.year.encode() + \
            self.series.encode()

    def _get_meta(self, index):
        return json.loads(self._db_.Get(self._gen_sample_key(index)).decode())

    def _set_meta(self, index, value):
        self._db_.Put(self._gen_sample_key(index), json.dumps(value).encode())

    def _init_files(self, force=False):
        self.debug("Initializing files table...")
        files_key = ("files_%s_%s" % (self.year, self.series)).encode()
        if not force:
            try:
                files = self._db_.Get(files_key)
                self.debug("Loaded files table from DB")
                return json.loads(files.decode())
            except KeyError:
                pass
        self.info("Looking for images in %s:", self._ipath)
        file_sets = {}
        for set_name, years in Loader.MAPPING.items():
            file_sets[set_name] = imgs = []
            subdir = years[self.year][self.series][0]
            path = os.path.join(self._ipath, subdir)
            self.info("Scanning %s...", path)
            for root, _, files in os.walk(path, followlinks=True):
                imgs.extend([self._img_file_name(path, os.path.join(root, f))
                             for f in files
                             if os.path.splitext(f)[1] == ".JPEG" and
                             f.find("-256") < 0])
        self.info("Saving files table to DB...")
        self._db_.Put(files_key, json.dumps(file_sets).encode())
        self.info("Initialized files table")
        return file_sets

    def _init_metadata(self, file_sets, force=False):
        self.debug("Initializing metadata...")
        metadata_key = ("metadata_%s_%s" % (self.year, self.series)).encode()
        if not force:
            try:
                metadata = json.loads(self._db_.Get(metadata_key).decode())
                self.debug("Loaded metadata from DB")
                return metadata
            except KeyError:
                pass
        self.info("Looking for metadata in %s:", self._ipath)
        all_xmls = {}
        for set_name, years in Loader.MAPPING.items():
            all_xmls[set_name] = xmls = []
            subdir = years[self.year][self.series][1]
            if not subdir:
                continue
            path = os.path.join(self._ipath, subdir)
            self.info("Scanning %s...", path)
            for root, _, files in os.walk(path, followlinks=True):
                xmls.extend([os.path.join(root, f)
                             for f in files
                             if os.path.splitext(f)[1] == ".xml"])
        self.info("Building image indices mapping")
        ifntbl = {}
        for set_name, files in file_sets.items():
            ifntbl[set_name] = {self._fixup_duplicate_dirs(name): index
                                for index, name in enumerate(files)}
            assert len(ifntbl[set_name]) == len(files), \
                "Duplicate file names detected in %s (%s, %s)" % \
                    (set_name, self.year, self.series)
        self.info("Parsing XML files...")
        metadata = {}
        progress = ProgressBar(maxval=sum(
            [len(xmls) for xmls in all_xmls.values()]))
        progress.start()
        for set_name, xmls in all_xmls.items():
            metadata[set_name] = mdsn = {}
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
                mdsn[index] = tree["annotation"]
        progress.finish()
        self.info("Saving metadata to DB...")
        self._db_.Put(metadata_key, json.dumps(metadata).encode())
        self.info("Initialized metadata")
        return metadata

    def _init_samples_categories(self, file_sets, metadata, force=False):
        self.debug("Initializing samples and categories...")
        samples_key = ("samples_%s_%s" % (self.year, self.series)).encode()
        cats_key = ("categories_%s_%s" % (self.year, self.series)).encode()
        if not force:
            try:
                sample_sets = json.loads(self._db_.Get(samples_key).decode())
                categories = json.loads(self._db_.Get(cats_key).decode())
                self.debug("Loaded %d, %d, %d samples in %d categories",
                           len(sample_sets["test"]),
                           len(sample_sets["validation"]),
                           len(sample_sets["train"]),
                           len(categories))
                return sample_sets, categories
            except KeyError:
                pass
        if file_sets is None or metadata is None:
            return None, None
        self.info("Building samples and categories...")
        samples = []
        sample_sets = {}
        categories = {}
        sindex = 0
        fmeta_misses = {}
        progress = ProgressBar(maxval=sum(
            [len(files) for files in file_sets.values()]))
        progress.start()
        for set_name, files in file_sets.items():
            sample_sets[set_name] = ss = []
            fmeta_misses[set_name] = 0
            set_index = loader.TRIAGE[set_name]
            for i in range(len(files)):
                progress.inc()
                try:
                    objs = metadata[set_name][str(i)]["object"]
                except KeyError:
                    fmeta_misses[set_name] += 1
                    samples.append((files[i], set_index, None))
                    self._set_meta(sindex, samples[sindex])
                    ss.append(sindex)
                    ckey = os.path.basename(os.path.dirname(files[i]))
                    try:
                        categories[ckey]
                    except KeyError:
                        categories[ckey] = []
                    categories[ckey].append(sindex)
                    sindex += 1
                    continue
                if isinstance(objs, dict):
                    objs = [objs]
                for obj in objs:
                    samples.append((files[i], set_index, obj))
                    self._set_meta(sindex, samples[sindex])
                    ss.append(sindex)
                    ckey = obj["name"]
                    try:
                        categories[ckey]
                    except KeyError:
                        categories[ckey] = []
                    categories[ckey].append(sindex)
                    sindex += 1
        progress.finish()
        assert sum(len(files) for files in sample_sets.values()) == \
               sum(len(files) for files in categories.values())
        table = PrettyTable("set", "files", "objects", "bbox", "bbox objs",
                            "bbox/files,%", "bbox objs/objects,%")
        table.align["set"] = "l"
        table.align["files"] = "l"
        table.align["objects"] = "l"
        table.align["bbox"] = "l"
        table.align["bbox objs"] = "l"
        for set_name, files in file_sets.items():
            bbox = len(files) - fmeta_misses[set_name]
            set_index = loader.TRIAGE[set_name]
            bbox_objs = len([t for t in samples if t[2] and t[1] == set_index])
            table.add_row(set_name, len(files), len(sample_sets[set_name]),
                          bbox, bbox_objs, int(bbox * 100 / len(files)),
                          int(bbox_objs * 100 / len(sample_sets[set_name])))
        self.info("Stats:\n%s", str(table))
        self.info("Saving samples and categories to DB...")
        self._db_.Put(samples_key, json.dumps(sample_sets).encode())
        self._db_.Put(cats_key, json.dumps(categories).encode())
        return sample_sets, categories

    def _init_labels(self, categories, force=False):
        self.debug("Initializing labels...")
        labels_key = ("labels_%s_%s" % (self.year, self.series)).encode()
        if not force:
            try:
                label_map = json.loads(self._db_.Get(labels_key).decode())
                self.debug("Found %d labels in DB", len(label_map))
                return label_map
            except KeyError:
                pass
        self.info("Building labels...")
        label_map = {v: i for i, v in enumerate(sorted(categories.keys()))
                     if v}
        self._db_.Put(labels_key, json.dumps(label_map).encode())
        self.info("Initialized %d labels", len(label_map))

    def _fill_class_samples(self, sample_sets):
        for set_name, files in sample_sets.items():
            index = loader.TRIAGE[set_name]
            self.class_samples[index] = len(files)
