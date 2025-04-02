from support import gp


def run_pse(pse_pars, pse_dir, acq_func="variance", optimizer='gpcam', gpcam_iterations=50, parallel_measurements=1):
    """
    Initializes and runs the Gaussian Process phase space exploration, PSE (also supports grid search). Results are
    saved in the pse_dir directory. Returns a success flag.
    :param pse_pars: (dict) PSE parameters, those that are being optimized and those that are not.
    :param pse_dir: (str) Path to the directory where the results are saved.
    :param acq_func: (str) gpCAM acquisition function
    :param optimizer: (str) PSE optimization algorithm (gpcam or grid)
    :param gpcam_iterations: (int) number of Gaussian Process iterations
    :param parallel_measurements: (int) number of parallel measurements per iteration
    :return: (Bool) success flag.
    """
    gpo = gp.Gp(exp_par=pse_pars, storage_path=pse_dir, acq_func=acq_func, optimizer=optimizer,
                gpcam_iterations=gpcam_iterations, parallel_measurements=parallel_measurements, resume=True)
    # success flag currently unused
    success = gpo.run()

    return success

