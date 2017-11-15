"""
Utility functions for calculating hashing metrics
"""

import io
import itertools

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

def _get_hdist(code1, code2):
    "return hamming distance"
    assert len(code1) == len(code2)
    return len([1 for i in range(len(code1)) if code1[i]!=code2[i]])