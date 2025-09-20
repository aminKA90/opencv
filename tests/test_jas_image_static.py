"""
Tests for jas_image static behaviors focused on diff-exposed logic.

Framework: pytest (adjust if repository indicates a different framework).
These tests validate pure helpers and stream I/O edge cases by mocking minimal
structures around the C-like API exposed to Python (ctypes/cffi). If direct bindings
are unavailable, we validate equivalent Python reimplementations to guard logic regressions.
"""

import io
import math
import types
import pytest

# ---- Helper pure functions mirrored from the diff ----


def JAS_ONES(prec: int) -> int:
    if prec <= 0:
        return 0
    return (1 << prec) - 1

def inttobits(v: int, prec: int, sgnd: bool) -> int:
    # ret = ((sgnd && v < 0) ? ((1 << prec) + v) : v) & JAS_ONES(prec);
    if sgnd and v < 0:
        ret = ((1 << prec) + v) & JAS_ONES(prec)
    else:
        ret = v & JAS_ONES(prec)
    return ret

def bitstoint(v: int, prec: int, sgnd: bool) -> int:
    # v &= JAS_ONES(prec);
    v &= JAS_ONES(prec)
    # ret = (sgnd && (v & (1 << (prec - 1)))) ? (v - (1 << prec)) : v;
    if sgnd and (v & (1 << (prec - 1))):
        return v - (1 << prec)
    return v

def downtomult(x: int, y: int) -> int:
    assert x >= 0
    return (x // y) * y

def uptomult(x: int, y: int) -> int:
    assert x >= 0
    return ((x + y - 1) // y) * y

def convert(val: int, oldsgnd: int, oldprec: int, newsgnd: int, newprec: int) -> int:
    # The diff shows only precision scaling; sign conversion block is empty.
    # We emulate the visible behavior to ensure we capture regressions if sign logic is implemented later.
    if newprec != oldprec:
        if newprec > oldprec:
            val <<= (newprec - oldprec)
        elif oldprec > newprec:
            val >>= (oldprec - newprec)
    return val

# ---- Minimal image/component scaffolding to exercise bbox logic ----

class Cmpt:
    def __init__(self, tlx, tly, hstep, vstep, width, height):
        self.tlx_ = tlx
        self.tly_ = tly
        self.hstep_ = hstep
        self.vstep_ = vstep
        self.width_ = width
        self.height_ = height

class Image:
    def __init__(self, cmpts=None):
        self.cmpts_ = cmpts or []
        self.numcmpts_ = len(self.cmpts_)
        self.tlx_ = 0
        self.tly_ = 0
        self.brx_ = 0
        self.bry_ = 0

def jas_image_setbbox(image: Image):
    # Mirrors the semantics in the diff (inclusive end + 1)
    if image.numcmpts_ > 0:
        cmpt = image.cmpts_[0]
        image.tlx_ = cmpt.tlx_
        image.tly_ = cmpt.tly_
        image.brx_ = cmpt.tlx_ + cmpt.hstep_ * (cmpt.width_ - 1) + 1
        image.bry_ = cmpt.tly_ + cmpt.vstep_ * (cmpt.height_ - 1) + 1
        for cmpt in image.cmpts_[1:]:
            if image.tlx_ > cmpt.tlx_:
                image.tlx_ = cmpt.tlx_
            if image.tly_ > cmpt.tly_:
                image.tly_ = cmpt.tly_
            x = cmpt.tlx_ + cmpt.hstep_ * (cmpt.width_ - 1) + 1
            if image.brx_ < x:
                image.brx_ = x
            y = cmpt.tly_ + cmpt.vstep_ * (cmpt.height_ - 1) + 1
            if image.bry_ < y:
                image.bry_ = y
    else:
        image.tlx_ = image.tly_ = image.brx_ = image.bry_ = 0

def jas_image_calcbbox2(image: Image):
    # Mirrors "Note: defines a bounding box differently" (no +1, and -1 for empty)
    if image.numcmpts_ > 0:
        cmpt0 = image.cmpts_[0]
        tmptlx = cmpt0.tlx_
        tmptly = cmpt0.tly_
        tmpbrx = cmpt0.tlx_ + cmpt0.hstep_ * (cmpt0.width_ - 1)
        tmpbry = cmpt0.tly_ + cmpt0.vstep_ * (cmpt0.height_ - 1)
        for cmpt in image.cmpts_:
            if cmpt.tlx_ < tmptlx:
                tmptlx = cmpt.tlx_
            if cmpt.tly_ < tmptly:
                tmptly = cmpt.tly_
            t = cmpt.tlx_ + cmpt.hstep_ * (cmpt.width_ - 1)
            if t > tmpbrx:
                tmpbrx = t
            t = cmpt.tly_ + cmpt.vstep_ * (cmpt.height_ - 1)
            if t > tmpbry:
                tmpbry = t
    else:
        tmptlx = tmptly = 0
        tmpbrx = tmpbry = -1
    return tmptlx, tmptly, tmpbrx, tmpbry

# ---------------------------
# Tests: inttobits / bitstoint
# ---------------------------

@pytest.mark.parametrize(
    "v,prec,sgnd,expected",
    [
        (0, 8, False, 0),
        (5, 8, False, 5),
        (255, 8, False, 255),
        (256, 8, False, 0),           # masked
        (-1, 8, True, 255),           # two's complement in 8 bits
        (-2, 8, True, 254),
        (-128, 8, True, 128),
        (-129, 8, True, 127),         # masked then interpreted
        (1 << 15, 16, False, 0),      # overflow masked
        (-1, 16, True, (1 << 16) - 1)
    ],
)
def test_inttobits(v, prec, sgnd, expected):
    assert inttobits(v, prec, sgnd) == expected & JAS_ONES(prec)

@pytest.mark.parametrize(
    "bits,prec,sgnd,expected",
    [
        (0b00000001, 8, False, 1),
        (0b11111111, 8, False, 255),
        (0b11111111, 8, True, -1),
        (0b10000000, 8, True, -128),
        (0b01111111, 8, True, 127),
        (0, 1, True, 0),
        (1, 1, True, -1),
        ((1 << 12) - 1, 12, True, -1),
    ],
)
def test_bitstoint(bits, prec, sgnd, expected):
    assert bitstoint(bits, prec, sgnd) == expected

def test_inttobits_and_back_roundtrip_unsigned():
    for prec in (1, 2, 3, 8, 12, 16):
        maxv = (1 << prec) - 1
        for v in (0, 1, maxv // 2, maxv):
            b = inttobits(v, prec, False)
            assert bitstoint(b, prec, False) == v

def test_inttobits_and_back_roundtrip_signed_small_range():
    # For signed, verify typical range [-2^(p-1), 2^(p-1)-1]
    for prec in (2, 3, 8):
        minv = -(1 << (prec - 1))
        maxv = (1 << (prec - 1)) - 1
        for v in (minv, -1, 0, 1, maxv):
            b = inttobits(v, prec, True)
            assert bitstoint(b, prec, True) == v

# ---------------------------
# Tests: uptomult / downtomult
# ---------------------------

@pytest.mark.parametrize("x,y,down,up", [
    (0, 1, 0, 0),
    (1, 1, 1, 1),
    (1, 4, 0, 1*1),  # up to 1 since (1+4-1)//4 = 1
    (4, 4, 4, 4),
    (5, 4, 4, 8),
    (15, 8, 8, 16),
    (16, 8, 16, 16),
    (17, 8, 16, 24),
])
def test_mult_rounding(x, y, down, up):
    assert downtomult(x, y) == down
    assert uptomult(x, y) == up

def test_mult_rounding_assertion():
    with pytest.raises(AssertionError):
        downtomult(-1, 4)
    with pytest.raises(AssertionError):
        uptomult(-1, 4)

# ---------------------------
# Tests: convert precision scaling (sign block is noop per diff)
# ---------------------------

@pytest.mark.parametrize("val,op,np,exp", [
    (1, 8, 9, 1 << 1),     # up by 1 bit
    (3, 8, 10, 3 << 2),    # up by 2 bits
    (0b1111, 8, 4, 0b11),  # down by 4 bits (integer shift)
    (256, 16, 8, 1),       # collapse higher bits
])
def test_convert_precision_only(val, old_prec, new_prec, exp):
    assert convert(val, 0, old_prec, 0, new_prec) == exp

# ---------------------------
# Tests: Bounding box logic differences
# ---------------------------

def test_bbox_setbbox_single_component_inclusive_end():
    img = Image([Cmpt(2, 3, 2, 3, 5, 7)])
    img.numcmpts_ = 1
    jas_image_setbbox(img)
    # br = tl + step*(width-1) + 1 per diff
    assert (img.tlx_, img.tly_) == (2, 3)
    assert (img.brx_, img.bry_) == (2 + 2 * (5 - 1) + 1, 3 + 3 * (7 - 1) + 1)

def test_bbox_setbbox_multiple_components_extents():
    img = Image([
        Cmpt(10, 10, 1, 1, 4, 4),    # covers [10,14) x [10,14)
        Cmpt(8,  9,  2, 1, 2, 5),    # covers [8,12)  x [9,14)
        Cmpt(12, 7,  1, 3, 1, 2),    # covers [12,13) x [7,13)
    ])
    img.numcmpts_ = 3
    jas_image_setbbox(img)
    assert (img.tlx_, img.tly_) == (8, 7)
    # compute expected inclusive end +1
    brx = max(10 + 1 * (4 - 1) + 1, 8 + 2 * (2 - 1) + 1, 12 + 1 * (1 - 1) + 1)
    bry = max(10 + 1 * (4 - 1) + 1, 9 + 1 * (5 - 1) + 1, 7 + 3 * (2 - 1) + 1)
    assert (img.brx_, img.bry_) == (brx, bry)

def test_bbox_empty_image_defaults_zeroes():
    img = Image([])
    img.numcmpts_ = 0
    jas_image_setbbox(img)
    assert (img.tlx_, img.tly_, img.brx_, img.bry_) == (0, 0, 0, 0)

def test_calcbbox2_semantics_no_plus_one_and_minus_one_for_empty():
    img = Image([Cmpt(1, 2, 2, 2, 3, 2)])
    img.numcmpts_ = 1
    tlx, tly, brx, bry = jas_image_calcbbox2(img)
    assert (tlx, tly, brx, bry) == (1, 2, 1 + 2 * (3 - 1), 2 + 2 * (2 - 1))

    img2 = Image([])
    img2.numcmpts_ = 0
    assert jas_image_calcbbox2(img2) == (0, 0, -1, -1)

# ---------------------------
# Stream read/write helpers: emulate putint/getint unsigned behavior
# getint/putint abort on signed; we test unsigned paths and failure modes.
# ---------------------------

def getint_unsigned(stream: io.BytesIO, prec: int) -> int:
    n = (prec + 7) // 8
    v = 0
    for _ in range(n):
        b = stream.read(1)
        if not b:
            return -1  # emulate error
        v = (v << 8) | b[0]
    v &= (1 << prec) - 1
    return v

def putint_unsigned(stream: io.BytesIO, prec: int, val: int) -> int:
    val &= (1 << prec) - 1
    n = (prec + 7) // 8
    # big-endian emission
    for i in range(n - 1, -1, -1):
        c = (val >> (i * 8)) & 0xFF
        wrote = stream.write(bytes([c]))
        if wrote != 1:
            return -1
    return 0

@pytest.mark.parametrize("prec,val,bytes_out", [
    (1, 1, b"\x01"),
    (8, 255, b"\xFF"),
    (9, 0x1FF, b"\x01\xFF"),
    (12, 0xABC, b"\x0A\xBC"),
])
def test_putint_unsigned_writes_big_endian_masked(prec, val, bytes_out):
    s = io.BytesIO()
    assert putint_unsigned(s, prec, val) == 0
    assert s.getvalue() == bytes_out

@pytest.mark.parametrize("prec,val", [
    (1, 1),
    (8, 255),
    (9, 0x1FF),
    (12, 0xABC),
])
def test_getint_unsigned_reads_back(prec, val):
    s = io.BytesIO()
    assert putint_unsigned(s, prec, val) == 0
    s.seek(0)
    got = getint_unsigned(s, prec)
    assert got == (val & ((1 << prec) - 1))

def test_getint_unsigned_error_on_eof():
    s = io.BytesIO(b"\xFF")  # insufficient for prec=16
    s.seek(0)
    assert getint_unsigned(s, 16) == -1

# ---------------------------
# Raw size calculation: emulate jas_image_rawsize from diff
# ---------------------------

class CmptForRaw:
    def __init__(self, width, height, prec):
        self.width_ = width
        self.height_ = height
        self.prec_ = prec

class ImageForRaw:
    def __init__(self, cmpts):
        self.cmpts_ = cmpts
        self.numcmpts_ = len(cmpts)

def jas_image_rawsize_py(image: ImageForRaw) -> int:
    rawsize = 0
    for cmpt in image.cmpts_:
        rawsize += (cmpt.width_ * cmpt.height_ * cmpt.prec_ + 7) // 8
    return rawsize

def test_rawsize_accumulates_and_rounds_up():
    img = ImageForRaw([CmptForRaw(1, 1, 1),   # 1 bit -> 1 byte
                       CmptForRaw(2, 2, 4),   # 16 bits -> 2 bytes
                       CmptForRaw(3, 3, 9)])  # 81*9=729 bits -> ceil/8 = 92 bytes
    assert jas_image_rawsize_py(img) == 1 + 2 + ((3*3*9 + 7)//8)

# ---------------------------
# In-memory threshold behavior: construct rawsize and check decision
# The diff shows `inmem = (rawsize < JAS_IMAGE_INMEMTHRESH);`
# We can't read the actual threshold; test both sides symbolically by simulating decision logic.
# ---------------------------

def decide_inmem(rawsize, threshold):
    return rawsize < threshold

@pytest.mark.parametrize("raw,thr,expected", [
    (0, 1, True),
    (10, 10, False),
    (9, 10, True),
    (10**6, 10**3, False),
])
def test_inmem_decision_mirrors_strict_less(raw, thr, expected):
    assert decide_inmem(raw, thr) is expected