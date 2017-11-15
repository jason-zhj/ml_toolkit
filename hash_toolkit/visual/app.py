"""
Simple GUI for demo hash retrieval
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button
from tkinter import filedialog
from PIL import Image
from ml_toolkit.hash_toolkit.metrics.utils import _retrieve_items_at_dist

def gallery(array, ncols=10):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols + 1
    if (nindex < nrows * ncols):
        array = np.vstack([
            array,np.zeros(shape=[nrows*ncols-nindex] + list(array.shape[1:]))
        ])
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def make_array(filenames):
    return np.array([np.asarray(Image.open(fn).convert('RGB')) for fn in filenames])


def parse_csv(csv_path,delimiter="\t"):
    "return a list of {filename:...,label:...,hash:..}"
    lines = open(csv_path).readlines()
    items = []
    for line in lines:
        filename,label,hash = line.strip().split(delimiter)
        items.append({"filename":filename,"label":label,"hash":hash})
    return items



class HashDemoGUI:

    def __init__(self, master, db_items, query_items,query_file_path,search_radius=2):
        "hash_db is a list of {hash:...,label:...,filename:...}"
        self.master = master
        master.title("Hash Demo")

        self.greet_button = Button(master, text="Search", command=self.retrieve)
        self.greet_button.pack()

        self.file_button = Button(master, text="Open file", command=self.open_file)
        self.file_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

        self.query_file = None
        self.hash_db = db_items
        self.query_items = query_items
        self.query_file_dir = query_file_path
        self.search_radius = search_radius

    def _check_same_path(self,p1,p2):
        return os.path.normcase(os.path.normpath(p1)) == os.path.normcase(os.path.normpath(p2))

    def retrieve(self):
        assert self.query_file is not None
        # locate the query item
        query_item = list(filter(lambda x:self._check_same_path(p1=self.query_file,p2=x["filename"]),self.query_items))
        assert len(query_item) == 1
        query_item = query_item[0]

        # retrieve from db items
        for dist in range(self.search_radius+1):
            items = _retrieve_items_at_dist(hashcode=query_item["hash"],db_set=self.hash_db,dist=dist)
            if (items):
                array = make_array(filenames=[item["filename"] for item in items])
                result = gallery(array)
                plt.figure(dist)
                plt.imshow(result)
            else:
                print("No items retrieved as dist={}".format(dist))
        plt.show()

    def open_file(self):
        filename = filedialog.askopenfilename(initialdir = self.query_file_dir)
        self.query_file = filename
        print(filename)



def run_hash_demo(query_csv,db_csv,query_file_path,csv_delimiter="\t"):
    """
    :param query_csv: should be in the format of filename, label,hash
    :param db_csv: ...
    :param query_file_path: path where query files are located
    """
    root = Tk()
    query_items = parse_csv(csv_path=query_csv,delimiter=csv_delimiter)
    db_items = parse_csv(csv_path=db_csv,delimiter=csv_delimiter)
    my_gui = HashDemoGUI(root, db_items=db_items, query_file_path=query_file_path,query_items=query_items)
    root.mainloop()


if __name__ == "__main__":
    run_hash_demo(query_csv="G:/generated_hash/query.csv",db_csv="G:/generated_hash/db.csv",
                  query_file_path="F:/data/mnist_m/mini/query-db-split/query")