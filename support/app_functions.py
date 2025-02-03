import glob
import os
from PIL import Image
import pandas
import shutil
import streamlit as st
import subprocess

import gp


def resume_pse(psedir, pse_pars, acq_func="variance", optimizer='gpcam'):
    gpo = gp.Gp(exp_par=pse_pars, storage_path=psedir, acq_func=acq_func, optimizer=optimizer, resume=True)
    gpo.run()


def run_pse(psedir, pse_pars, acq_func="variance", optimizer='gpcam'):
    gpo = gp.Gp(exp_par=pse_pars, storage_path=psedir, acq_func=acq_func, optimizer=optimizer)
    gpo.run()
