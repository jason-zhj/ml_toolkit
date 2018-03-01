"""
Utility functions for calculating hashing metrics
"""

import io
import itertools
from collections import defaultdict
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def _pyplot_to_image(plt_obj):
    "return a PIL.Image object"
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    return im


def _fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def _fig2img (fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = _fig2data (fig)
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


def process_hash_csv(file_path,delimiter=","):
    "return list of {'label':..,'hash':'010101'}"
    results = []
    with open(file_path,"r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            filename,label,hashcode = line.strip().split(delimiter)
            hashcode = hashcode.replace(" ","")
            results.append({"label":label,"hash":hashcode})
    return results


def _compute_hash_with_dist(hashcode, dist):
    "`hashcode` should only contain '0' and '1', return a list of hash strings, whose hamming distance to `hashcode` is `dist`"
    hash_ls = []
    invert = lambda x:"1" if x=="0" else "0"
    code_length = len(hashcode)
    for positions in itertools.combinations(range(code_length),dist):
        # invert the hash bit at `positions`
        computed_code = "".join([invert(hashcode[i]) if i in positions else hashcode[i]
                                 for i in range(code_length)])
        hash_ls.append(computed_code)

    return hash_ls


def _compute_hash_within_radius(hashcode, radius):
    "return list of hash strings, `hashcode` should be binary string '0101..'"
    hash_ls = []
    for i in range(radius+1):
        hash_ls += _compute_hash_with_dist(hashcode=hashcode, dist=i)
    return hash_ls


def _retrieve_items_using_hash(db_set, hashcode):
    "return a list of {'label':..,'hash':..}, retrieved from `db_set`, `db_set` should be list of dict"
    return [item for item in db_set if item["hash"]==hashcode]

def _retrieve_items_at_dist(db_set,hashcode,dist):
    hash_ls = _compute_hash_with_dist(hashcode=hashcode,dist=dist)
    items = []
    for code in hash_ls:
        items += _retrieve_items_using_hash(db_set=db_set,hashcode=code)
    return items

def get_hdist(code1, code2):
    "return hamming distance, '-' is recognized as inactive bit"
    assert len(code1) == len(code2)
    dist = 0
    for c1, c2 in zip(code1, code2):
        if (c1 != c2 and c1 != "-" and c2 != "-"):
            dist += 1
    return dist


def _retrieve_items_all(db_set,hashcode,max_hdist=None):
    "return a dict {0:[list of items],1:[]...}"
    results = defaultdict(list)
    for item in db_set:
        dist = get_hdist(item["hash"], hashcode)
        if (max_hdist is None or dist <= max_hdist):
            results[dist].append(item)

    # fill up zero entry for non-existent keys
    max_key = max_hdist if max_hdist is not None else len(hashcode)
    for i in range(max_hdist+1):
        if (i not in results.keys()):
            results[i] = []
    return results

def compute_line(p,q):
    x0,y0 = p
    x1,y1 = q
    k = (y1 - y0)/(x1 - x0)
    b = y0 - k*x0
    return k,b

def compare_performance(l1,l2):
    "l1,l2 is a list: [(p0,r0),(p1,r1),(p2,r2)], return 1 if l1 lies above l2, -1 if below, 0 if neither above or below"
    p1,p2,p3 = l1
    q1,q2,q3 = l2
    lie_above = [True for _ in range(3)] # record True if p lies above q
    k,b = compute_line(q1,q2)
    if (p1[1] < k * p1[0] + b): lie_above[0] = False
    if (p2[1] < k * p2[0] + b): lie_above[1] = False

    k, b = compute_line(q2, q3)
    if (p2[1] < k * p2[0] + b): lie_above[1] = False
    if (p3[1] < k * p3[0] + b): lie_above[2] = False

    if (all(lie_above)):
        return 1
    elif (all([not i for i in lie_above])):
        return -1
    else:
        return 0
