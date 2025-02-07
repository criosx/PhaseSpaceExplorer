import glob
import os
from PIL import Image
import pandas as pd
import shutil
import streamlit as st
import subprocess

from support import gp


def run_pse(pse_pars, pse_dir, acq_func="variance", optimizer='gpcam', gpcam_iterations=50):
    gpo = gp.Gp(exp_par=pse_pars, storage_path=pse_dir, acq_func=acq_func, optimizer=optimizer,
                gpcam_iterations=gpcam_iterations, resume=True)
    gpo.run()
