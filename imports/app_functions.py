import glob
import os
from PIL import Image
import pandas
from sasmodels.data import load_data
from scattertools.support import molstat
from scattertools.support import api_sasview
import shutil
import streamlit as st
import subprocess


def get_info_from_runfile(model_name, model_dir, file_dir, fit_dir):
    runfile = os.path.join(model_dir, model_name)
    # extract name of data files from runfile
    datafile_names = api_sasview.extract_data_filenames_from_runfile(runfile=runfile)

    # strip any long path from filename and retain only the basename, write back to file
    datafile_names = [os.path.basename(file) for file in datafile_names]
    api_sasview.write_data_filenames_to_runfile(runfile=runfile, filelist=datafile_names)

    # check if data files are in user folder, if not then create dummy files
    datafile_names_user = [os.path.join(file_dir, os.path.basename(file)) for file in datafile_names]

    molstat.prepare_fit_directory(fitdir=fit_dir, runfile=runfile, datafile_names=datafile_names_user)

    # if datafiles exist in user filedir, use those; otherwise create dummy files
    for filename in datafile_names_user:
        if os.path.isfile(filename):
            shutil.copyfile(filename, os.path.join(fit_dir, os.path.basename(filename)))
        else:
            api_sasview.write_dummy_sans_file(os.path.join(fit_dir, os.path.basename(filename)))

    os.chdir(fit_dir)
    fitobj = molstat.CMolStat(
        fitsource="SASView",
        spath=fit_dir,
        mcmcpath="MCMC",
        runfile=os.path.basename(runfile),
        state=None,
        problem=None,
    )
    df_pars = pandas.DataFrame.from_dict(fitobj.fnLoadParameters())
    li_allpars = list(fitobj.Interactor.problem.model_parameters().keys())

    return df_pars, li_allpars, datafile_names, fitobj


@st.cache_data
def load_SANS_data(uploaded_file, file_save_dir):
    try:
        with open(os.path.join(file_save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        ds, tb_output, file_name = load_sans_file(uploaded_file.name, file_save_dir)
        if file_name is None:
            uploaded_file = None
    except IOError:
        ds = None
        tb_output = ["Invalid SANS data file."]
        uploaded_file = None
    return ds, tb_output, uploaded_file


@st.cache_data
def load_sans_file(file_name, file_dir):
    try:
        dso = load_data(os.path.join(file_dir, file_name))
        Q = dso.x
        Iq = dso.y
        dI = dso.dy
        dQ = dso.dx
        ds = pandas.DataFrame(
            {'Q': Q,
             'I': Iq,
             'dI': dI,
             'dQ': dQ
             }
        )
        tb_output = ""
    except IOError:
        ds = None
        tb_output = ["Invalid SANS data file."]
        file_name = None

    return ds, tb_output, file_name


@st.cache_data
def load_sans_files(filelist, file_dir):
    for file in filelist:
        try:
            with open(os.path.join(file_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        except IOError:
            pass


def monitor_jobs(job_dir):
    status_path = os.path.join(job_dir, 'status.json')

    if not os.path.isfile(status_path):
        status = 'idle'
    else:
        status_df = pandas.read_json(status_path)
        status = status_df['status'].values[0]

    # last modification to top experimental optimization folder
    list_of_files = glob.glob(job_dir)
    latest_file = max(list_of_files, key=os.path.getctime)
    jobtime = os.path.getctime(latest_file)

    return jobtime, status


def run_fit(fitdir=None, runfile=None, datafile_names=None, datafile_names_uploaded=None, file_dir=None, model_dir=None,
            burn=1000, steps=200):
    # save current working directory
    olddir = os.getcwd()

    # check if all data files are in place
    breakflag = False
    for file in datafile_names:
        if file not in datafile_names_uploaded:
            if os.path.isfile(os.path.join(file_dir, file)):
                infostr = 'Data file ' + file + ' not uploaded. Using copy from user file folder.'
                st.info(infostr)
            else:
                errorstr = 'Data file ' + file + ' not uploaded.'
                st.error(errorstr)
                breakflag = True

    if breakflag:
        return

    datafpaths = [os.path.join(file_dir, file) for file in datafile_names]
    molstat.prepare_fit_directory(fitdir=fitdir, runfile=os.path.join(model_dir, runfile), datafile_names=datafpaths)

    st.info("Starting the fit ...")
    runfile = os.path.splitext(runfile)[0]
    os.chdir(fitdir)
    with open('run_fit.py', 'w') as file:
        file.write('from scattertools.support import molstat\n')
        file.write('\n')
        file.write("if __name__ == '__main__':\n")
        file.write('    fitobj = molstat.CMolStat(\n')
        file.write('        fitsource="SASView",\n')
        file.write('        spath="' + fitdir + '",\n')
        file.write('        mcmcpath="MCMC",\n')
        file.write('        runfile="' + runfile + '",\n')
        file.write('        state=None,\n')
        file.write('        problem=None,\n')
        file.write('    )\n')
        file.write('\n')
        file.write('    fitobj.Interactor.fnRunMCMC(' + str(burn) + ', ' + str(steps) + ', batch=False)\n')
        file.write('\n')

    # bumps uses multithreading, which collides with Streamlits requirement to register the thread context
    subprocess.call(['python', 'run_fit.py'])

    fitobj = molstat.CMolStat(
        fitsource="SASView",
        spath=fitdir,
        mcmcpath="MCMC",
        runfile=runfile,
        state=None,
        problem=None,
    )

    st.info("Analyzing the fit ...")
    fitobj.fnRestoreFit()
    results = fitobj.fnAnalyzeStatFile(fConfidence=-1)
    st.write('Results:')
    st.write(results)
    for file in os.listdir("MCMC"):
        if file.endswith(".png"):
            image = Image.open(os.path.join('MCMC', file))
            st.image(image)

    os.chdir(olddir)
    return


def run_optimization(optdir=None, runfile=None, file_dir=None, model_dir=None, burn=1000, steps=200):

    datafile_names = api_sasview.extract_data_filenames_from_runfile(runfile=runfile)
    for file in datafile_names:
        dfile = os.path.join(file_dir, file)
        if not os.path.isfile(dfile):
            infostr = 'Data file ' + file + ' not in user file folder. Please set up complete fit under Models and Fit'
            st.info(infostr)
            return

    datafpaths = [os.path.join(file_dir, file) for file in datafile_names]
    molstat.prepare_fit_directory(fitdir=optdir, runfile=os.path.join(model_dir, runfile), datafile_names=datafpaths)

    # TODO: copy optimization specific files
