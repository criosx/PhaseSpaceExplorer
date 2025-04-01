from support import gp


def run_pse(pse_pars, pse_dir, acq_func="variance", optimizer='gpcam', gpcam_iterations=50, parallel_measurements=1):
    gpo = gp.Gp(exp_par=pse_pars, storage_path=pse_dir, acq_func=acq_func, optimizer=optimizer,
                gpcam_iterations=gpcam_iterations, parallel_measurements=parallel_measurements, resume=True)
    gpo.run()
