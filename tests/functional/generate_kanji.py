#!/usr/bin/python3.3
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


add_path("/usr/local/lib/python3.3/dist-packages/freetype")
add_path("/usr/local/lib/python3.3/dist-packages/freetype/ft_enums")


from freetype import *
import numpy
import sqlite3
import xml.etree.ElementTree as et
import struct
import os
import glob


SX = 32
SY = 32


"""
typedef struct tagBITMAPFILEHEADER {
  WORD  bfType;
  DWORD bfSize;
  WORD  bfReserved1;
  WORD  bfReserved2;
  DWORD bfOffBits;
} BITMAPFILEHEADER, *PBITMAPFILEHEADER;

Members

bfType

    The file type; must be BM.
bfSize

    The size, in bytes, of the bitmap file.
bfReserved1

    Reserved; must be zero.
bfReserved2

    Reserved; must be zero.
bfOffBits

    The offset, in bytes, from the beginning of the BITMAPFILEHEADER structure to the bitmap bits.


typedef struct tagBITMAPINFOHEADER {
  DWORD biSize;
  LONG  biWidth;
  LONG  biHeight;
  WORD  biPlanes;
  WORD  biBitCount;
  DWORD biCompression;
  DWORD biSizeImage;
  LONG  biXPelsPerMeter;
  LONG  biYPelsPerMeter;
  DWORD biClrUsed;
  DWORD biClrImportant;
} BITMAPINFOHEADER, *PBITMAPINFOHEADER;

Members

biSize

    The number of bytes required by the structure.
biWidth

    The width of the bitmap, in pixels.

    If biCompression is BI_JPEG or BI_PNG, the biWidth member specifies the width of the decompressed JPEG or PNG image file, respectively.
biHeight

    The height of the bitmap, in pixels. If biHeight is positive, the bitmap is a bottom-up DIB and its origin is the lower-left corner. If biHeight is negative, the bitmap is a top-down DIB and its origin is the upper-left corner.

    If biHeight is negative, indicating a top-down DIB, biCompression must be either BI_RGB or BI_BITFIELDS. Top-down DIBs cannot be compressed.

    If biCompression is BI_JPEG or BI_PNG, the biHeight member specifies the height of the decompressed JPEG or PNG image file, respectively.
biPlanes

    The number of planes for the target device. This value must be set to 1.
biBitCount

    The number of bits-per-pixel. The biBitCount member of the BITMAPINFOHEADER structure determines the number of bits that define each pixel and the maximum number of colors in the bitmap. This member must be one of the following values.
    Value    Meaning
    0    The number of bits-per-pixel is specified or is implied by the JPEG or PNG format.
    1    The bitmap is monochrome, and the bmiColors member of BITMAPINFO contains two entries. Each bit in the bitmap array represents a pixel. If the bit is clear, the pixel is displayed with the color of the first entry in the bmiColors table; if the bit is set, the pixel has the color of the second entry in the table.
    4    The bitmap has a maximum of 16 colors, and the bmiColors member of BITMAPINFO contains up to 16 entries. Each pixel in the bitmap is represented by a 4-bit index into the color table. For example, if the first byte in the bitmap is 0x1F, the byte represents two pixels. The first pixel contains the color in the second table entry, and the second pixel contains the color in the sixteenth table entry.
    8    The bitmap has a maximum of 256 colors, and the bmiColors member of BITMAPINFO contains up to 256 entries. In this case, each byte in the array represents a single pixel.
    16    The bitmap has a maximum of 2^16 colors. If the biCompression member of the BITMAPINFOHEADER is BI_RGB, the bmiColors member of BITMAPINFO is NULL. Each WORD in the bitmap array represents a single pixel. The relative intensities of red, green, and blue are represented with five bits for each color component. The value for blue is in the least significant five bits, followed by five bits each for green and red. The most significant bit is not used. The bmiColors color table is used for optimizing colors used on palette-based devices, and must contain the number of entries specified by the biClrUsed member of the BITMAPINFOHEADER.

    If the biCompression member of the BITMAPINFOHEADER is BI_BITFIELDS, the bmiColors member contains three DWORD color masks that specify the red, green, and blue components, respectively, of each pixel. Each WORD in the bitmap array represents a single pixel.

    When the biCompression member is BI_BITFIELDS, bits set in each DWORD mask must be contiguous and should not overlap the bits of another mask. All the bits in the pixel do not have to be used.
    24    The bitmap has a maximum of 2^24 colors, and the bmiColors member of BITMAPINFO is NULL. Each 3-byte triplet in the bitmap array represents the relative intensities of blue, green, and red, respectively, for a pixel. The bmiColors color table is used for optimizing colors used on palette-based devices, and must contain the number of entries specified by the biClrUsed member of the BITMAPINFOHEADER.
    32    The bitmap has a maximum of 2^32 colors. If the biCompression member of the BITMAPINFOHEADER is BI_RGB, the bmiColors member of BITMAPINFO is NULL. Each DWORD in the bitmap array represents the relative intensities of blue, green, and red for a pixel. The value for blue is in the least significant 8 bits, followed by 8 bits each for green and red. The high byte in each DWORD is not used. The bmiColors color table is used for optimizing colors used on palette-based devices, and must contain the number of entries specified by the biClrUsed member of the BITMAPINFOHEADER.

    If the biCompression member of the BITMAPINFOHEADER is BI_BITFIELDS, the bmiColors member contains three DWORD color masks that specify the red, green, and blue components, respectively, of each pixel. Each DWORD in the bitmap array represents a single pixel.

    When the biCompression member is BI_BITFIELDS, bits set in each DWORD mask must be contiguous and should not overlap the bits of another mask. All the bits in the pixel do not need to be used.

     
biCompression

    The type of compression for a compressed bottom-up bitmap (top-down DIBs cannot be compressed). This member can be one of the following values.
    Value    Description
    BI_RGB    An uncompressed format.
    BI_RLE8    A run-length encoded (RLE) format for bitmaps with 8 bpp. The compression format is a 2-byte format consisting of a count byte followed by a byte containing a color index. For more information, see Bitmap Compression.
    BI_RLE4    An RLE format for bitmaps with 4 bpp. The compression format is a 2-byte format consisting of a count byte followed by two word-length color indexes. For more information, see Bitmap Compression.
    BI_BITFIELDS    Specifies that the bitmap is not compressed and that the color table consists of three DWORD color masks that specify the red, green, and blue components, respectively, of each pixel. This is valid when used with 16- and 32-bpp bitmaps.
    BI_JPEG    Indicates that the image is a JPEG image.
    BI_PNG    Indicates that the image is a PNG image.

     
biSizeImage

    The size, in bytes, of the image. This may be set to zero for BI_RGB bitmaps.

    If biCompression is BI_JPEG or BI_PNG, biSizeImage indicates the size of the JPEG or PNG image buffer, respectively.
biXPelsPerMeter

    The horizontal resolution, in pixels-per-meter, of the target device for the bitmap. An application can use this value to select a bitmap from a resource group that best matches the characteristics of the current device.
biYPelsPerMeter

    The vertical resolution, in pixels-per-meter, of the target device for the bitmap.
biClrUsed

    The number of color indexes in the color table that are actually used by the bitmap. If this value is zero, the bitmap uses the maximum number of colors corresponding to the value of the biBitCount member for the compression mode specified by biCompression.

    If biClrUsed is nonzero and the biBitCount member is less than 16, the biClrUsed member specifies the actual number of colors the graphics engine or device driver accesses. If biBitCount is 16 or greater, the biClrUsed member specifies the size of the color table used to optimize performance of the system color palettes. If biBitCount equals 16 or 32, the optimal color palette starts immediately following the three DWORD masks.

    When the bitmap array immediately follows the BITMAPINFO structure, it is a packed bitmap. Packed bitmaps are referenced by a single pointer. Packed bitmaps require that the biClrUsed member must be either zero or the actual size of the color table.
biClrImportant

    The number of color indexes that are required for displaying the bitmap. If this value is zero, all colors are required.
"""


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
                str(a.shape), ))

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


def do_plot(fontPath, text, size, angle=0.0, sx=1.0, sy=1.0,
            randomizePosition=True):
    face = Face(fontPath)
    #face.set_char_size(48 * 64)
    face.set_pixel_sizes(0, size)

    c = text[0]

    angle = (angle / 180.0) * numpy.pi

    mx_r = numpy.array([[numpy.cos(angle), -numpy.sin(angle)],
                        [numpy.sin(angle),  numpy.cos(angle)]],
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
            #print("Set pixel size for font %s to %d" % (fontPath, size - j))
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
        print("Font %s returned empty glyph" % (fontPath, ))
        return None
    return img


def create_tables(db):
    print("Will create tables...")
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
    print("done")


def fill_tables(db):
    print("Will fill tables...")
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
    print("done")


if __name__ == '__main__':
    db = sqlite3.connect("kanji.db")

    try:
        rs = db.execute("select count(*) from kanji")
    except sqlite3.OperationalError:
        create_tables(db)
        rs = db.execute("select count(*) from kanji")
    n_kanji = rs.fetchone()[0]
    if not n_kanji:
        fill_tables(db)
    query = ("select idx, literal from kanji where grade <> 0 "
             "order by grade asc, freq desc, idx asc limit 100")
    rs = db.execute("select count(*) from (%s)" % (query, ))
    n_kanji = rs.fetchone()[0]
    print("Kanji count: ", n_kanji)
    if n_kanji < 1:
        sys.exit(0)

    fonts = {b'simfang.ttf': 0,  # that will be target
             b'ipam.ttf': 1,
             b'ipag.ttf': 2,
             b'simhei.ttf': 3,
             b'msjh.ttf': 4,
             b'msyh.ttf': 5,
             b'eduSong_Unicode.ttf': 6,
             b'edukai-3.ttf': 7,
             b'ukai.ttc': 8,
             b'uming.ttc': 9,
             b'wqy-microhei.ttc': 10,
             b'VL-Gothic-Regular.ttf': 11}

    ok = {}
    for font in fonts.keys():
        ok[font] = 0

    rs = db.execute(query)

    bmp = BMPWriter()

    dirnme = "/home/ajk/Projects/kanji"
    target_dirnme = "/home/ajk/Projects/kanji_target"
    files = glob.glob("%s/*.bmp" % (dirnme, ))
    i = 0
    for file in files:
        try:
            os.unlink(file)
            i += 1
        except FileNotFoundError:
            pass
    if i:
        print("Unlinked %d files" % (i, ))
    files = glob.glob("%s/*.bmp" % (target_dirnme, ))
    i = 0
    for file in files:
        try:
            os.unlink(file)
            i += 1
        except FileNotFoundError:
            pass
    if i:
        print("Unlinked %d files" % (i, ))
    del files

    for row in rs:
        print(row[0], row[1])
        exists = False
        for font, idx in fonts.items():
            font_ok = False
            for i in range(0, 100):  # № of transformations for each character
            #for s in ((1.0, 1.0), (0.89, 1.0 / 0.89), (1.0 / 0.89, 0.89)):
            #    for angle in (0.0, 12.5, -12.5):
                angle = -12.0 + numpy.random.rand() * 24.0
                sx = 0.7 + numpy.random.rand() * (1.0 / 0.7 - 0.7)
                sy = 0.7 + numpy.random.rand() * (1.0 / 0.7 - 0.7)
                img = do_plot(font, row[1], SY, angle, sx, sy, True)
                if img == None:
                    #print("Not found for font %s" % (font, ))
                    continue
                fnme = "%s/%05d.%.1fx%.1f_%.0f.%02d.bmp" % (dirnme,
                    row[0], sx, sy, angle, idx)
                bmp.write_gray(fnme, img)
                if not font_ok:
                    if not idx:  # writing to target
                        img = do_plot(font, row[1], SY, 0, 1.0, 1.0, False)
                        fnme = "%s/%05d.bmp" % (target_dirnme, row[0])
                        bmp.write_gray(fnme, img)
                    font_ok = True
            if font_ok:
                ok[font] += 1
                exists = True
        if not exists:
            raise Exception("Glyph does not exists in the supplied fonts")

    for font, n in ok.items():
        print("%s: %d (%.2f%%)" % (font, n, 100.0 * n / n_kanji))

    sys.exit(0)
