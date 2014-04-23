"""
Created on Apr 10, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


from concurrent.futures import ThreadPoolExecutor
import cv2
import jpeg4py
import json
import leveldb
import multiprocessing as mp
import numpy
from PIL import Image
import shutil
import struct
import tempfile
import threading
import time
import os
from six.moves import cPickle as pickle

import veles.config as config
import veles.formats as formats
import veles.opencl_types as opencl_types
from veles.external.prettytable import PrettyTable
from veles.external.progressbar import ProgressBar
import veles.znicz.loader as loader
from veles.znicz.external import xmltodict


class LoaderBase(loader.Loader):
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

    def __init__(self, workflow, aperture, **kwargs):
        self._dbpath = kwargs.get("dbpath", config.root.imagenet.dbpath)
        super(LoaderBase, self).__init__(workflow, **kwargs)
        self._ipath = kwargs.get("ipath", config.root.imagenet.ipath)
        self._year = kwargs.get("year", config.root.imagenet.year)
        self._series = kwargs.get("series", config.root.imagenet.series)
        self.data_shape = (aperture, aperture)
        self._dtype = opencl_types.dtypes[config.root.common.precision_type]
        self._crop_color = kwargs.get(
            "crop_color",
            config.get(config.root.imagenet.crop_color) or (64, 64, 64))
        self._colorspace = kwargs.get(
            "colorspace", config.get(config.root.imagenet.colorspace) or "RGB")
        if self.colorspace == "GRAY":
            self._crop_color = self._crop_color[0]
        self._include_derivative = kwargs.get(
            "derivative", config.get(config.root.imagenet.derivative) or False)
        self._sobel_kernel_size = kwargs.get(
            "sobel_kernel_size",
            config.get(config.root.imagenet.sobel_ksize) or 5)
        self._force_reinit = kwargs.get(
            "force_reinit",
            config.get(config.root.imagenet.force_reinit) or False)

    def init_unpickled(self):
        super(LoaderBase, self).init_unpickled()
        if self._dbpath:
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

    @property
    def colorspace(self):
        return self._colorspace

    @property
    def include_derivative(self):
        return self._include_derivative

    @include_derivative.setter
    def include_derivative(self, value):
        self._include_derivative = value
        self._mean = self._init_mean(None, db_only=True)

    @property
    def channels(self):
        channels = 1 if self._colorspace == "GRAY" else 3
        if self._include_derivative:
            channels += 1
        return channels

    def load_data(self):
        self.info("Loading from %s...", self.db_path)
        progress = ProgressBar(maxval=4, term_width=11)
        progress.start()
        objects, categories = self._init_objects(None, None, db_only=True)
        progress.inc()
        images = self._init_images(None, None, db_only=True)
        progress.inc()
        if objects is not None and images is not None:
            self._label_map = self._init_labels(categories)
            self.fill_class_samples(objects, images)
            progress.inc()
            self._mean = self._init_mean(objects, db_only=True)
            if (self._mean is not None):
                progress.finish()
                return
        progress.finish()
        self.info("DB does not have all the necessary data, recovering...")
        file_sets = self._init_files()
        metadata = self._init_metadata(file_sets)
        objects, categories = self._init_objects(file_sets, metadata)
        images = self._init_images(file_sets, metadata)
        self._label_map = self._init_labels(categories)
        self.fill_class_samples(objects, images)
        self._mean = self._init_mean(objects)

    def create_minibatches(self):
        count = self.minibatch_maxsize
        minibatch_shape = [count] + list(self._data_shape) + [self.channels]
        self.minibatch_data << numpy.zeros(shape=minibatch_shape,
                                           dtype=self._dtype)
        self.minibatch_indexes << numpy.zeros(count, dtype=numpy.int32)

    def fill_minibatch(self):
        images = self._executor_.map(
            lambda i: (i, self._get_sample(self.shuffled_indexes[i])),
            range(self.minibatch_size))
        for i, data in images:
            self.minibatch_data[i] = data

    def get_object_file_name(self, index):
        meta = self.get_object_meta(index)
        return self._get_file_name_common(meta)

    def get_image_file_name(self, index):
        meta = self.get_image_meta(index)
        return self._get_file_name_common(meta)

    def get_category_by_file_name(self, file_name):
        return os.path.basename(os.path.dirname(file_name))

    def get_object_meta(self, index):
        return json.loads(self._db_.Get(self._gen_object_key(index)).decode())

    def get_image_meta(self, index):
        return json.loads(self._db_.Get(self._gen_image_key(index)).decode())

    def _set_object_meta(self, index, value):
        self._db_.Put(self._gen_object_key(index), json.dumps(value).encode())

    def _set_image_meta(self, index, value):
        self._db_.Put(self._gen_image_key(index), json.dumps(value).encode())

    def _get_file_name_common(self, meta):
        set_name = loader.CLASS_NAME[meta[1]]
        mapping = LoaderBase.MAPPING[set_name][self.year][self.series]
        return os.path.join(self._ipath, mapping[0], meta[0]) + ".JPEG"

    def _get_bbox(self, meta):
        bbox = meta["bndbox"]
        return (int(bbox["xmin"]), int(bbox["ymin"]),
                int(bbox["xmax"]), int(bbox["ymax"]))

    def _gen_object_key(self, index):
        return b"o" + struct.pack("I", index) + self.year.encode() + \
            self.series.encode()

    def _gen_image_key(self, index):
        return b"i" + struct.pack("I", index) + self.year.encode() + \
            self.series.encode()

    def _decode_image(self, index):
        file_name = self.get_sample_file_name(index)
        try:
            data = jpeg4py.JPEG(file_name).decode()
        except jpeg4py.JPEGRuntimeError as e:
            try:
                data = numpy.array(Image.open(file_name).convert("RGB"))
                self.warning("Falling back to PIL with file %s: %s",
                             file_name, repr(e))
            except:
                self.exception("Failed to decode %s", file_name)
                raise
        if len(data.shape) == 2 and self.colorspace != "GRAY":
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        if len(data.shape) == 2:
            data = data[:, :, numpy.newaxis]
        return data

    def _preprocess_sample(self, data):
        if self._include_derivative:
            deriv = cv2.Sobel(
                cv2.cvtColor(data, cv2.COLOR_RGB2GRAY) if data.shape[-1] > 1
                else data, cv2.CV_32F if self._dtype == numpy.float32
                else cv2.CV_64F, 1, 1, ksize=self._sobel_kernel_size)
        if self.colorspace != "RGB" and not (data.shape[-1] == 1 and
                                             self.colorspace == "GRAY"):
            cv2.cvtColor(data, getattr(cv2, "COLOR_RGB2" + self._colorspace),
                         data)
        if self._include_derivative:
            shape = list(data.shape)
            shape[-1] += 1
            res = numpy.empty(shape, dtype=self._dtype)
            res[:, :, :-1] = data
            begindex = len(shape)
            res.ravel()[begindex::(begindex + 1)] = deriv.ravel()
        else:
            res = data.astype(self._dtype)
        return res

    def _get_sample_raw(self, index):
        data = self._decode_image(index)
        data = self.crop_and_scale(data, index)
        data = self._preprocess_sample(data)
        return data

    def _get_sample(self, index):
        data = self._get_sample_raw(index)
        data -= self._mean
        formats.normalize(data)
        return data

    def _key_file_name(self, base, full):
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

    def _init_files(self, force=False, db_only=False):
        self.debug("Initializing files table...")
        files_key = ("files_%s_%s" % (self.year, self.series)).encode()
        if not force:
            try:
                file_sets = json.loads(self._db_.Get(files_key).decode())
                self.debug("Loaded files table from DB")
                return file_sets
            except KeyError:
                pass
        if db_only:
            return None
        self.info("Looking for images in %s:", self._ipath)
        file_sets = {}
        for set_name, years in LoaderBase.MAPPING.items():
            file_sets[set_name] = imgs = []
            subdir = years[self.year][self.series][0]
            path = os.path.join(self._ipath, subdir)
            self.info("Scanning %s...", path)
            for root, _, files in os.walk(path, followlinks=True):
                imgs.extend([self._key_file_name(path, os.path.join(root, f))
                             for f in files
                             if os.path.splitext(f)[1] == ".JPEG" and
                             f.find("-256") < 0])
        self.info("Saving files table to DB...")
        self._db_.Put(files_key, json.dumps(file_sets).encode())
        self.info("Initialized files table")
        return file_sets

    def _init_metadata(self, file_sets, force=False, db_only=False):
        self.debug("Initializing metadata...")
        metadata_key = ("metadata_%s_%s" % (self.year, self.series)).encode()
        if not force:
            try:
                metadata = json.loads(self._db_.Get(metadata_key).decode())
                self.debug("Loaded metadata from DB")
                return metadata
            except KeyError:
                pass
        if db_only:
            return None
        self.info("Looking for metadata in %s:", self._ipath)
        all_xmls = {}
        for set_name, years in LoaderBase.MAPPING.items():
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
                ("Duplicate file names detected in %s (%s, %s)" %
                 (set_name, self.year, self.series))
        self.info("Parsing XML files...")
        metadata = {}
        progress = ProgressBar(term_width=80, maxval=sum(
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
                file_key = self._key_file_name(
                    os.path.join(
                        self._ipath,
                        LoaderBase.MAPPING[set_name][self.year][
                            self.series][1]), xml)
                try:
                    index = ifntbl[set_name][file_key]
                except KeyError:
                    self.error(
                        "%s references unexistent file %s", xml, os.path.join(
                            self._ipath,
                            LoaderBase.MAPPING[set_name][self.year]
                            [self.series][0], file_key))
                    continue
                meta = tree["annotation"]
                mdsn[str(index)] = meta
        progress.finish()
        self.info("Saving metadata to DB...")
        self._db_.Put(metadata_key, json.dumps(metadata).encode())
        self.info("Initialized metadata")
        return metadata

    def _init_objects(self, file_sets, metadata, force=False, db_only=False):
        self.debug("Initializing objects and categories...")
        objects_key = ("objects_%s_%s" % (self.year, self.series)).encode()
        cats_key = ("categories_%s_%s" % (self.year, self.series)).encode()
        if not force:
            try:
                objects = json.loads(self._db_.Get(objects_key).decode())
                categories = json.loads(self._db_.Get(cats_key).decode())
                self.debug("Loaded %d, %d, %d objects and %d categories",
                           len(objects["test"]),
                           len(objects["validation"]),
                           len(objects["train"]),
                           len(categories))
                return objects, categories
            except KeyError:
                pass
        if db_only:
            return None, None
        self.info("Building objects and categories...")
        samples = []
        objects = {}
        categories = {}
        sindex = 0
        fmeta_misses = {}
        no_cat_recovery = {}
        bad_ckeys = {""}
        for year in LoaderBase.MAPPING.values():
            for sets in year.values():
                bad_ckeys.update([pair[0] for pair in sets.values()])
        progress = ProgressBar(term_width=80, maxval=sum(
            [len(files) for files in file_sets.values()]))
        progress.start()
        for set_name, files in file_sets.items():
            objects[set_name] = ss = []
            fmeta_misses[set_name] = 0
            no_cat_recovery[set_name] = 0
            set_index = loader.TRIAGE[set_name]
            for i in range(len(files)):
                progress.inc()
                try:
                    objs = metadata[set_name][str(i)]["object"]
                except KeyError:
                    fmeta_misses[set_name] += 1
                    ckey = self.get_category_by_file_name(files[i])
                    if ckey in bad_ckeys:
                        no_cat_recovery[set_name] += 1
                        continue
                    try:
                        categories[ckey]
                    except KeyError:
                        categories[ckey] = []
                    samples.append((files[i], set_index, None))
                    self._set_object_meta(sindex, samples[-1])
                    ss.append(sindex)
                    categories[ckey].append(sindex)
                    sindex += 1
                    continue
                if isinstance(objs, dict):
                    objs = [objs]
                for obj in objs:
                    samples.append((files[i], set_index, obj))
                    self._set_object_meta(sindex, samples[-1])
                    ss.append(sindex)
                    ckey = obj["name"]
                    try:
                        categories[ckey]
                    except KeyError:
                        categories[ckey] = []
                    categories[ckey].append(sindex)
                    sindex += 1
        progress.finish()
        sum_objects = sum(len(files) for files in objects.values())
        sum_categories = sum(len(files) for files in categories.values())
        assert sum_objects == sum_categories
        table = PrettyTable("set", "files", "objects", "bbox", "bbox objs",
                            "bad", "bbox/files,%", "bbox objs/objects,%")
        table.align["set"] = "l"
        table.align["files"] = "l"
        table.align["objects"] = "l"
        table.align["bbox"] = "l"
        table.align["bbox objs"] = "l"
        for set_name, files in file_sets.items():
            bbox = len(files) - fmeta_misses[set_name]
            set_index = loader.TRIAGE[set_name]
            bbox_objs = len([t for t in samples if t[2] and t[1] == set_index])
            table.add_row(set_name, len(files), len(objects[set_name]),
                          bbox, bbox_objs, no_cat_recovery[set_name],
                           int(bbox * 100 / len(files)),
                          int(bbox_objs * 100 / (len(objects[set_name])
                          if len(objects[set_name]) > 0 else 1)))
        self.info("Stats:\n%s", str(table))
        self.info("Saving objects and categories to DB...")
        self._db_.Put(objects_key, json.dumps(objects).encode())
        self._db_.Put(cats_key, json.dumps(categories).encode())
        return objects, categories

    def _init_images(self, file_sets, metadata, force=False, db_only=False):
        self.debug("Initializing images...")
        images_key = ("images_%s_%s" % (self.year, self.series)).encode()
        if not force:
            try:
                images = json.loads(self._db_.Get(images_key).decode())
                self.debug("Loaded %d, %d, %d images",
                           images[0][1] - images[0][0],
                           images[1][1] - images[1][0],
                           images[2][1] - images[2][0])
                return images
            except KeyError:
                pass
        if db_only:
            return None
        self.info("Building images...")
        images = [0] * len(loader.CLASS_NAME)
        for set_name, files in file_sets.items():
            images[loader.TRIAGE[set_name]] = len(files)
        index = 0
        for i in range(len(images)):
            num = images[i]
            images[i] = [index, index + num]
            index += num

        self.info("Saving images to DB...")
        bad_ckeys = {""}
        for year in LoaderBase.MAPPING.values():
            for sets in year.values():
                bad_ckeys.update([pair[0] for pair in sets.values()])
        progress = ProgressBar(term_width=80,
                               maxval=sum([(i[1] - i[0]) for i in images]))
        progress.start()
        for set_name, files in file_sets.items():
            set_index = loader.TRIAGE[set_name]
            iindex = 0
            for i in range(len(files)):
                progress.inc()
                meta = metadata[set_name].get(i)
                try:
                    metadata[set_name][str(i)]["object"]
                except KeyError:
                    ckey = self.get_category_by_file_name(files[i])
                    if ckey in bad_ckeys:
                        images[set_index][1] -= 1
                        for j in range(set_index + 1, len(images)):
                            images[j][0] -= 1
                            images[j][1] -= 1
                        continue
                self._set_image_meta(images[set_index][0] + iindex,
                                     (files[i], set_index, meta))
                iindex += 1
        progress.finish()
        self._db_.Put(images_key, json.dumps(images).encode())
        return images

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

    def _init_mean(self, objects, force=False, db_only=False):
        self.debug("Initializing mean sample...")
        mean_key = ("mean_%s_%s_%s_%d" % (self.year, self.series,
                                          self.colorspace,
                                          self.include_derivative)).encode()
        if not force:
            try:
                mean = pickle.loads(self._db_.Get(mean_key))
                self.debug("Found mean sample in DB")
                return mean
            except KeyError:
                pass
        if db_only:
            return None
        self.info("Preparing to calculate the mean...")
        actors = mp.cpu_count()
        indices = objects["train"]
        indices = indices[:(len(indices) // actors) * actors]
        mean_done = mp.Value('i', 0)
        shared_mean = mp.Array(
            'd', self._data_shape[0] * self._data_shape[1] * self.channels)
        mean = numpy.frombuffer(shared_mean.get_obj())
        mean = mean.reshape(list(self._data_shape) + [self.channels])
        mean.fill(0)
        progress = ProgressBar(maxval=len(indices), term_width=80)
        progress_thread = threading.Thread(target=self._show_mean_progress,
                                           args=(progress, mean_done))
        dirs = [tempfile.mkdtemp("-imagenet-db-%d" % i, "veles-")
                for i in range(actors)]
        self.info("Replicating the database...")
        dir_progress = ProgressBar(maxval=len(dirs),
                                   term_width=(7 + len(dirs)))
        dir_progress.start()
        for d in dirs:
            os.rmdir(d)
            shutil.copytree(self.db_path, d)
            dir_progress.inc()
        tasks = [(dirs[i], indices[i::actors], len(indices))
                 for i in range(actors)]
        self.info("Calculating the mean sample...")
        progress_thread.start()
        dbpath = self._dbpath
        self._dbpath = None
        with mp.Pool(actors, self._init_mean_process,
                     (shared_mean, mean_done)) as pool:
            pool.starmap(self._calculate_mean_process, tasks, 1)
        progress_thread.join()
        self._dbpath = dbpath
        self._db_.Put(mean_key, pickle.dumps(mean))
        self.info("Calculated mean sample")

    def _init_mean_process(self, shared_mean, mean_done):
        global _shared_mean
        _shared_mean = shared_mean
        global _mean_done
        _mean_done = mean_done

    def _calculate_mean_process(self, dbpath, indices, N):
        self._db_ = leveldb.LevelDB(dbpath)
        mean = numpy.frombuffer(_shared_mean.get_obj())
        mean = mean.reshape(list(self._data_shape) + [self.channels])
        for index in indices:
            sample = self._get_sample_raw(index)
            sample /= N
            with _shared_mean.get_lock():
                mean += sample
            with _mean_done.get_lock():
                _mean_done.value += 1

    def _show_mean_progress(self, progress, mean_done):
        value = 0
        progress.start()
        while (value < progress.maxval):
            time.sleep(0.5)
            with mean_done.get_lock():
                value = mean_done.value
            progress.update(value)
        progress.finish()


class LoaderDetection(LoaderBase):
    def __init__(self, workflow, **kwargs):
        aperture = kwargs.get("aperture",
                              config.get(config.root.imagenet.aperture) or 256)
        super(LoaderDetection, self).__init__(workflow, aperture, **kwargs)
        self._use_bboxes = kwargs.get(
            "bboxes", config.get(config.root.imagenet.force_reinit) or True)

    def create_minibatches(self):
        super(LoaderDetection, self).create_minibatches()
        self.minibatch_labels << numpy.zeros(self.minibatch_maxsize,
                                             dtype=numpy.int32)

    def fill_minibatch(self):
        super(LoaderBase, self).fill_minibatch()
        for i in range(self.minibatch_size):
            meta = self.get_object_meta(self.shuffled_indexes[i])
            if meta[2] is not None:
                name = meta["name"]
            else:
                fn = self.get_object_file_name(self.shuffled_indexes[i])
                name = self.get_category_by_file_name(fn)
            self.minibatch_labels[i] = self._label_map[name]

    def fill_class_samples(self, objects, images):
        for set_name, files in objects.items():
            index = loader.TRIAGE[set_name]
            self.class_samples[index] = len(files)

    def get_sample_file_name(self, index):
        return self.get_object_file_name(index)

    def crop_and_scale(self, img, index):
        width = img.shape[1]
        height = img.shape[0]
        meta = self.get_object_meta(index)
        if meta[2] is not None and self._use_bboxes:
            bbox = list(self._get_bbox(meta[2]))
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
            bbox[1] -= int(numpy.floor(offset))
            bbox[3] += int(numpy.ceil(offset))
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
            bbox[0] += int(numpy.ceil(offset))
            bbox[2] -= int(numpy.floor(offset))
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


class LoaderBoundingBox(LoaderBase):
    def __init__(self, workflow, **kwargs):
        aperture = kwargs.get("aperture",
                              config.get(config.root.imagenet.aperture)
                              or 4000)
        super(LoaderDetection, self).__init__(workflow, aperture, **kwargs)
        self.max_detected_bboxes = 10

    def create_minibatches(self):
        super(LoaderDetection, self).create_minibatches()
        # label = category index + xmin + ymin + xmax + ymax
        label_size = self.minibatch_maxsize * 5 * self._max_detected_bboxes
        self.minibatch_labels << numpy.zeros(label_size, dtype=numpy.int32)

    def fill_minibatch(self):
        super(LoaderBase, self).fill_minibatch()
        stride = 5 * self._max_detected_bboxes
        for i in range(self.minibatch_size):
            index = self.shuffled_indexes[i]
            meta = self.get_image_meta(index)[1]
            if meta[2] is not None:
                objects = meta[2]["object"]
            else:
                objects = {
                    "name":
                    self.get_category_by_file_name(
                        self.get_image_file_name(index)),
                    "bndbox": {"xmin": (-1), "ymin": (-1), "xmax": (-1),
                               "ymax": (-1)}}
            if isinstance(objects, dict):
                objects = [objects]
            for j, obj in enumerate(objects):
                name = obj["name"]
                bbox = self._get_bbox(obj)
                self.minibatch_labels[i * stride + j * 5] = \
                    self._label_map[name]
                for k in range(4):
                    self.minibatch_labels[i * stride + j * 5 + 1 + k] = bbox[k]

    def fill_class_samples(self, objects, images):
        for index, files in enumerate(images):
            self.class_samples[index] = files[1] - files[0]

    def get_sample_file_name(self, index):
        return self.get_image_file_name(index)

    def crop_and_scale(self, img, index):
        assert img.shape[:2] < self._data_shape, "Too small source shape set"
        res = numpy.empty((self._data_shape[0], self._data_shape[1],
                           img.shape[2]))
        res[:, :, :] = self._crop_color
        res[:img.shape[0], :img.shape[1], :] = img
        return res
