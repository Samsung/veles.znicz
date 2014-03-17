#!/usr/bin/python3.3 -O
"""
Created on June 29, 2013

File for generation of samples for kanji recognition.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import sys
import os
import logging


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


this_dir = os.path.dirname(__file__)
if not this_dir:
    this_dir = "."
add_path("%s/../../../src" % (this_dir))
add_path("%s/../.." % (this_dir))


import six
from six.moves import xrange as range
from six.moves import cPickle as pickle
from freetype import *
import numpy
import sqlite3
import xml.etree.ElementTree as et
import glob
import config
import scipy.misc
import formats
import time


SX = 32
SY = 32
TARGET_SX = 24
TARGET_SY = 24
N_TRANSFORMS = 200
ANGLE = 17.5
SCALE = 0.61
KANJI_COUNT = 100


def do_plot(fontPath, text, size, angle, sx, sy,
            randomizePosition, SX, SY):
    face = Face(fontPath.encode("utf-8"))
    #face.set_char_size(48 * 64)
    face.set_pixel_sizes(0, size)

    c = text[0]

    angle = (angle / 180.0) * numpy.pi

    mx_r = numpy.array([[numpy.cos(angle), -numpy.sin(angle)],
                        [numpy.sin(angle), numpy.cos(angle)]],
                       dtype=numpy.double)
    mx_s = numpy.array([[sx, 0.0],
                        [0.0, sy]], dtype=numpy.double)

    mx = numpy.dot(mx_s, mx_r)

    matrix = FT_Matrix((int)(mx[0, 0] * 0x10000),
                       (int)(mx[0, 1] * 0x10000),
                       (int)(mx[1, 0] * 0x10000),
                       (int)(mx[1, 1] * 0x10000))
    flags = FT_LOAD_RENDER
    pen = FT_Vector(0, 0)
    FT_Set_Transform(face._FT_Face, byref(matrix), byref(pen))

    j = 0
    while True:
        slot = face.glyph
        if not face.get_char_index(c):
            return None
        face.load_char(c, flags)
        bitmap = slot.bitmap
        width = bitmap.width
        height = bitmap.rows
        if width < 1 or height < 1:
            logging.warning("strange (width, height) = (%d, %d), skipping" % (
                                                                width, height))
            return None
        if width > SX or height > SY:
            j = j + 1
            face.set_pixel_sizes(0, size - j)
            #logging.info("Set pixel size for font %s to %d" % (
            #    fontPath, size - j))
            continue
        break

    if randomizePosition:
        x = int(numpy.floor(numpy.random.rand() * (SX - width)))
        y = int(numpy.floor(numpy.random.rand() * (SY - height)))
    else:
        x = int(numpy.floor((SX - width) * 0.5))
        y = int(numpy.floor((SY - height) * 0.5))

    img = numpy.zeros([SY, SX], dtype=numpy.uint8)
    try:
        img[y:y + height, x: x + width] = numpy.array(bitmap.buffer,
            dtype=numpy.uint8).reshape(height, width)
    except ValueError:
        logging.warning("Strange bitmap was generated: width=%d, height=%d, "
            "len=%d but should be %d, skipping" % (bitmap.width, bitmap.rows,
            len(bitmap.buffer), bitmap.width * bitmap.rows))
        return None
    if img.max() == img.min():
        logging.info("Font %s returned empty glyph" % (fontPath))
        return None
    return img


def create_tables(db):
    logging.info("Will create tables...")
    db.execute(
        "create table kanji (\n"
        "idx             integer    not null primary key autoincrement,\n"
        "literal         text       not null unique,\n"
        "grade           integer    null,\n"
        "stroke_count    integer    null,\n"
        "freq            integer    null,\n"
        "jlpt            integer    null,\n"
        "pinyin          text       null,\n"
        "korean_r        text       null,\n"
        "korean_h        text       null,\n"
        "ja_on           text       null,\n"
        "ja_kun          text       null,\n"
        "meaning         text       null,\n"
        "nanori          text       null)")
    logging.info("done")


def fill_tables(db):
    logging.info("Will fill tables...")
    tree = et.parse("kanjidic2.xml")
    root = tree.getroot()
    def_kanji = {
        "literal": "",
        "grade": 0,
        "stroke_count": 0,
        "freq": 0,
        "jlpt": 0,
        "pinyin": "",
        "korean_r": "",
        "korean_h": "",
        "ja_on": "",
        "ja_kun": "",
        "meaning": "",
        "nanori": ""}
    kanji = def_kanji.copy()
    for char in root.iter("character"):
        kanji.update(def_kanji)
        for sub in char.iter():
            if sub.tag == "reading":
                tag = sub.attrib["r_type"]
            else:
                tag = sub.tag
                if len(sub.attrib):
                    continue
            if tag not in kanji.keys():
                continue
            if type(kanji[tag]) == str:
                if len(kanji[tag]):
                    kanji[tag] += "\n"
                kanji[tag] += sub.text
            elif type(kanji[tag]) == int:
                kanji[tag] += int(sub.text)
            else:
                raise Exception("Unknown type for kanji attribute found")
        query = "insert into kanji ("
        q = ""
        params = []
        first = True
        for key, value in kanji.items():
            if not first:
                query += ", "
                q += ", "
            else:
                first = False
            query += key
            q += "?"
            params.append(value)
        query += ") values (" + q + ")"
        db.execute(query, params)
    db.commit()
    logging.info("done")


if __name__ == '__main__':
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    numpy.random.seed(numpy.fromfile("seed", dtype=numpy.int32, count=1024))

    db = sqlite3.connect("%s/kanji/kanji.db" % (config.test_dataset_root))

    try:
        rs = db.execute("select count(*) from kanji")
    except sqlite3.OperationalError:
        create_tables(db)
        rs = db.execute("select count(*) from kanji")
    n_kanji = rs.fetchone()[0]
    if not n_kanji:
        fill_tables(db)
    #query = ("select idx, literal from kanji where grade <> 0 "
    #         "order by grade asc, freq desc, idx asc limit %d" % (
    #                                                    KANJI_COUNT))
    query = ("select idx, literal from kanji where jlpt >= 2 "
             "order by grade asc, freq desc, jlpt desc, idx asc limit %d" % (
                                                                KANJI_COUNT))
    rs = db.execute("select count(*) from (%s)" % (query))
    n_kanji = rs.fetchone()[0]
    logging.info("Kanji count: %d" % (n_kanji))
    if n_kanji < 1:
        sys.exit(0)

    fonts = glob.glob("%s/kanji/fonts/*" % (config.test_dataset_root))
    fonts.sort()

    ok = {}
    for font in fonts:
        ok[font] = 0

    rs = db.execute(query)

    dirnme = "%s/kanji/train" % (config.test_dataset_root)
    target_dirnme = "%s/kanji/target" % (config.test_dataset_root)

    logging.info("Be sure that %s and %s are empty" % (dirnme, target_dirnme))
    logging.info("Will continue in 15 seconds")
    time.sleep(5)
    logging.info("Will continue in 10 seconds")
    time.sleep(5)
    logging.info("Will continue in 5 seconds")
    time.sleep(2)
    logging.info("Will continue in 3 seconds")
    time.sleep(1)
    logging.info("Will continue in 2 seconds")
    time.sleep(1)
    logging.info("Will continue in 1 second")
    time.sleep(1)

    fout = open("%s/label_dbindex" % (dirnme), "w")
    fout.write("Folders are named as label_dbindex.\n")
    fout.close()

    lbl = -1
    n_dups = 0
    index_map = []
    targets = []
    for row in rs:
        lbl += 1
        db_idx = row[0]
        character = row[1]
        logging.info("lbl=%d: db_idx=%d %s" % (lbl, db_idx, character))
        exists = False
        for font_idx, font in enumerate(fonts):
            font_ok = False
            transforms = set()
            for i in range(N_TRANSFORMS):
                while True:
                    angle = -ANGLE + numpy.random.rand() * (ANGLE * 2)
                    sx = SCALE + numpy.random.rand() * (1.0 / SCALE - SCALE)
                    sy = SCALE + numpy.random.rand() * (1.0 / SCALE - SCALE)
                    key = "%.1f_%.2f_%.2f" % (angle, sx, sy)
                    if key in transforms:
                        n_dups += 1
                        logging.info("Same transform found, will retry")
                        continue
                    transforms.add(key)
                    break
                img = do_plot(font, character, SY, angle, sx, sy, True,
                              SX, SY)
                if img is None:
                    #logging.info("Not found for font %s" % (font))
                    continue
                a = img.astype(numpy.float32)
                formats.normalize(a)
                outdir = "%s/%05d_%05d" % (dirnme, lbl, db_idx)
                try:
                    os.mkdir(outdir)
                except OSError:
                    pass
                sample_number = len(index_map)
                fnme = "%s/%07d" % (outdir, sample_number)
                scipy.misc.imsave("%s.png" % (fnme), img)
                pickle_fnme = "%s.%d.pickle" % (fnme, 3 if six.PY3 else 2)
                fout = open(pickle_fnme, "wb")
                pickle.dump({"angle": angle,
                             "lbl": lbl,
                             "data": a,
                             "db_idx": db_idx,
                             "sample_number": sample_number,
                             "sx": sx,
                             "sy": sy}, fout)
                fout.close()
                if not font_ok:
                    if not font_idx:  # writing to target
                        img = do_plot(font, character, TARGET_SY, 0, 1.0, 1.0,
                                      False, TARGET_SX, TARGET_SY)
                        fnme = "%s/%05d.png" % (target_dirnme, lbl)
                        scipy.misc.imsave(fnme, img)
                        a = img.astype(numpy.float32)
                        formats.normalize(a)
                        targets.append(a)
                    font_ok = True
                index_map.append(pickle_fnme[len(dirnme) + 1:])
            if font_ok:
                ok[font] += 1
                exists = True
        if not exists:
            raise Exception("Glyph does not exists in the supplied fonts")

    fout = open("%s/targets.%d.pickle" % (target_dirnme, 3 if six.PY3 else 2),
                "wb")
    pickle.dump(targets, fout)
    fout.close()

    fout = open("%s/index_map.%d.pickle" % (dirnme, 3 if six.PY3 else 2), "wb")
    pickle.dump(index_map, fout)
    fout.close()

    for font, n in ok.items():
        logging.info("%s: %d (%.2f%%)" % (font, n, 100.0 * n / n_kanji))

    logging.info("Retried transforms %d times" % (n_dups))
    logging.info("Generated %d samples" % (len(index_map)))

    logging.info("End of job")
    sys.exit(0)
