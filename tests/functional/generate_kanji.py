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
add_path("%s/../.." % (this_dir,))
add_path("%s/../../../src" % (this_dir,))


add_path("/usr/local/lib/python3.3/dist-packages/freetype")
add_path("/usr/local/lib/python3.3/dist-packages/freetype/ft_enums")


from freetype import *
import numpy
import sqlite3
import xml.etree.ElementTree as et
import struct
import glob
import config


SX = 32
SY = 32
TARGET_SX = 24
TARGET_SY = 24
N_TRANSFORMS = 100
KANJI_COUNT = 200


class BMPWriter(object):
    """Writes bmp file.

    Attributes:
        rgbquad: color table for gray scale.
    """
    def __init__(self):
        self.rgbquad = numpy.zeros([256, 4], dtype=numpy.uint8)
        for i in range(0, 256):
            self.rgbquad[i] = (i, i, i, 0)

    def write_gray(self, fnme, a):
        """Writes bmp as gray scale.

        Parameters:
            fnme: file name.
            a: numpy array with 2 dimensions.
        """
        if len(a.shape) != 2:
            raise Exception("a should be 2-dimensional, got: %s" % (
                str(a.shape),))

        fout = open(fnme, "wb")

        header_size = 54 + 256 * 4
        file_size = header_size + a.size

        header = struct.pack("<HIHHI"
                             "IiiHHIIiiII",
                             19778, file_size, 0, 0, header_size,
                             40, a.shape[1], -a.shape[0], 1, 8, 0, a.size,
                             0, 0, 0, 0)
        fout.write(header)

        self.rgbquad.tofile(fout)

        a.astype(numpy.uint8).tofile(fout)

        fout.close()


def do_plot(fontPath, text, size, angle, sx, sy,
            randomizePosition, SX, SY):
    face = Face(bytes(fontPath, 'UTF-8'))
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
    img[y:y + height, x: x + width] = numpy.array(bitmap.buffer,
        dtype=numpy.uint8).reshape(height, width)
    if img.max() == img.min():
        logging.info("Font %s returned empty glyph" % (fontPath,))
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

    db = sqlite3.connect("%s/kanji/kanji.db" % (config.test_dataset_root,))

    try:
        rs = db.execute("select count(*) from kanji")
    except sqlite3.OperationalError:
        create_tables(db)
        rs = db.execute("select count(*) from kanji")
    n_kanji = rs.fetchone()[0]
    if not n_kanji:
        fill_tables(db)
    query = ("select idx, literal from kanji where grade <> 0 "
             "order by grade asc, freq desc, idx asc limit %d" % (
                                                        KANJI_COUNT,))
    rs = db.execute("select count(*) from (%s)" % (query,))
    n_kanji = rs.fetchone()[0]
    logging.info("Kanji count: %d" % (n_kanji,))
    if n_kanji < 1:
        sys.exit(0)

    fonts = glob.glob("%s/kanji/fonts/*" % (config.test_dataset_root,))
    fonts.sort()

    ok = {}
    for font in fonts:
        ok[font] = 0

    rs = db.execute(query)

    bmp = BMPWriter()

    dirnme = "%s/kanji/train" % (config.test_dataset_root,)
    target_dirnme = "/%s/kanji/target" % (config.test_dataset_root,)
    files = glob.glob("%s/*.bmp" % (dirnme,))
    i = 0
    for file in files:
        try:
            os.unlink(file)
            i += 1
        except FileNotFoundError:
            pass
    if i:
        logging.info("Unlinked %d files" % (i,))
    files = glob.glob("%s/*.bmp" % (target_dirnme,))
    i = 0
    for file in files:
        try:
            os.unlink(file)
            i += 1
        except FileNotFoundError:
            pass
    if i:
        logging.info("Unlinked %d files" % (i,))
    del files

    ii = 0
    for row in rs:
        ii += 1
        logging.info("%d: %d %s" % (ii, row[0], row[1]))
        exists = False
        for idx, font in enumerate(fonts):
            font_ok = False
            for i in range(0, N_TRANSFORMS):
            #for s in ((1.0, 1.0), (0.89, 1.0 / 0.89), (1.0 / 0.89, 0.89)):
            #    for angle in (0.0, 12.5, -12.5):
                angle = -12.0 + numpy.random.rand() * 24.0
                sx = 0.7 + numpy.random.rand() * (1.0 / 0.7 - 0.7)
                sy = 0.7 + numpy.random.rand() * (1.0 / 0.7 - 0.7)
                img = do_plot(font, row[1], SY, angle, sx, sy, True,
                              SX, SY)
                if img == None:
                    #logging.info("Not found for font %s" % (font, ))
                    continue
                fnme = "%s/%05d.%.1fx%.1f_%.0f.%02d.bmp" % (dirnme,
                    row[0], sx, sy, angle, idx)
                bmp.write_gray(fnme, img)
                if not font_ok:
                    if not idx:  # writing to target
                        img = do_plot(font, row[1], TARGET_SY, 0, 1.0, 1.0,
                                      False, TARGET_SX, TARGET_SY)
                        fnme = "%s/%05d.bmp" % (target_dirnme, row[0])
                        bmp.write_gray(fnme, img)
                    font_ok = True
            if font_ok:
                ok[font] += 1
                exists = True
        if not exists:
            raise Exception("Glyph does not exists in the supplied fonts")

    for font, n in ok.items():
        logging.info("%s: %d (%.2f%%)" % (font, n, 100.0 * n / n_kanji))

    sys.exit(0)
