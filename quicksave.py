import os
import pickle
import json
import csv
from datetime import datetime
import re

class QuickSaver():
    """
    Saving args:
        obj (depends on function): thing to save
        name (str): if not None, use this as the base for making a savename 
        replace (bool): if True, will replace 'name.ext' file if it already exists.
            If False, will add a '_n' to the end of the filename, with 'n' higher 
            than any currently existing 'n' for that name base
        inc_date (bool): whether or not to add date to filename

    Loading args:
        name (str): name of file to load in self.fileloc, including extension
    """
    def __init__(self, file_loc='quick_saves', subfolder=None):
        self.file_loc = file_loc
        os.makedirs(self.file_loc, exist_ok=True)

        if subfolder is not None:
            subfolder = self.inc_name(subfolder, ext='')
            self.file_loc = os.path.join(self.file_loc, subfolder)
            os.makedirs(self.file_loc, exist_ok=True)

    def inc_name(self, name, ext):
        all_dirs = os.listdir(self.file_loc)
        p = re.compile(name + '_' + "[0-9]+" + ext)
        rel_dirs = [d for d in all_dirs if re.fullmatch(p, d)]
        if len(ext) > 0:
            ns = [int(d[len(name)+1:-len(ext)]) for d in rel_dirs] + [0]
        else:
            ns = [int(d[len(name)+1:]) for d in rel_dirs] + [0]

        n = max(ns)
        name = name + "_{}"
        sloc = name + ext
        while sloc.format(n) in all_dirs:
            n += 1
        return name.format(n)
        
    def get_save_name(self, name, ext, replace=False, inc_date=False):
        name = 'tmp' if name is None else name

        ext_ind = name.find('.')
        if ext_ind >= 0:
            if name[name.find('.'):] != ext:
                print("Overriding given extension '{}' to correct one".format(name[name.find('.'):]))
            name = name[:ext_ind]

        if inc_date:
            name += '_' + datetime.strftime(datetime.now(), "%m-%d")
        if not replace:
            name = self.inc_name(name, ext)
        name += ext
        return name

    def get_save_loc(self, name, ext, replace=False, inc_date=False):
        name = self.get_save_name(name, ext, replace, inc_date)
        return os.path.join(self.file_loc, name)

    ### pickling: works on generic python objects, not readable ###
    def save_pickle(self, obj, name=None, replace=False, inc_date=False):
        loc = self.get_save_loc(name, '.pkl', replace, inc_date)

        with open(loc, 'wb') as f:
            pickle.dump(obj, f)

    def load_pickle(self, name):
        with open(os.path.join(self.file_loc, name), 'rb') as f:
           obj = pickle.load(f)
        return obj

    ### txt files: save str() of obj in .txt (or, optionally .py) ###
    def save_txt(self, obj, name=None, replace=False, inc_date=False, as_py=False):
        loc = self.get_save_loc(name, '.txt' if not as_py else '.py', replace, inc_date)

        with open(loc, 'w') as f:
            f.write(str(obj))

    def load_txt(self, name):
        with open(os.path.join(self.file_loc, name), 'r') as f:
           string = f.read()
        return string

    def txt_to_obj(self, name):
        text = self.load_txt(name)
        return eval(text) 

    ### json, works with dicts ###
    def save_json(self, obj, name=None, replace=False, inc_date=False):
        loc = self.get_save_loc(name, '.json', replace, inc_date)

        js = json.dumps(obj, sort_keys=True, indent=4, separators=(',', ':'))
        with open(loc, 'w') as f:
            f.write(js)

    def load_json(self, name):
        with open(os.path.join(self.file_loc, name), 'r') as f:
            return json.load(f)

    def load_json_path(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    ### csv, for a list of lists ###
    def save_csv(self, rows, name=None, replace=False, inc_date=False):
        """
        Each list is a new row
        """
        loc = self.get_save_loc(name, '.csv', replace, inc_date)

        with open(loc, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def load_csv(self, name):
        """
        Everythin is loaded as strings
        """
        loc = os.path.join(self.file_loc, name)
        with open(loc, 'r') as f:
            lines = []
            reader = csv.reader(f)
            for line in reader:
                # l = list(map(lambda x: float(x), line))
                lines.append(line)
            return lines

    def switch_cols_rows(self, data):
        ''' For saving CSVs, change list of columns to list of rows.
        Also works for going back (rows to cols) '''
        new_len = range(len(data[0]))
        new_data = [[c[i] for c in data] for i in new_len]
        return new_data

    ### logging ###
    def print_and_log(self, text, name=''):
        if name != '':
            name = '_' + name
        print(text)
        self.save_txt(text, 'log' + name, inc_date=True)