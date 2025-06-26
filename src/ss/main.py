from utils import *
import os

script_dir = os.path.abspath(os.getcwd())
fname = os.path.abspath(os.path.join(script_dir, 'data', 'AF700_740bp_0001.hdf5'))

tpsfs1, inten1, tpsfs2, inten2 = utils.ss3_read_hdf5_file(fname)    
output = utils.process_single_shot_lifetime(fname, gate=61, gate_width=3)