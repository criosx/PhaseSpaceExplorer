from gpcam.autonomous_experimenter import AutonomousExperimenterGP

import numpy as np
class gp:
    def __init__(self):
        pass

    def run_optimization_gpcam(self):
        # Using the gpCAM global optimizer, follows the example from the gpCAM website

        # initialization
        # feel free to try different acquisition functions, e.g. optional_acq_func, "covariance", "shannon_ig"
        # note how costs are defined in for the autonomous experimenter
        parlimits = []
        for row in self.steppar.iterrows():
            parlimits.append([row[1].l_sim, row[1].u_sim])
        parlimits = np.array(parlimits)
        numpars = len(parlimits)

        x = self.gpCAMstream['position']
        y = self.gpCAMstream['value']
        v = self.gpCAMstream['variance']

        if len(x) >= 1:
            # use any previously computed results
            x = np.array(x)
            y = np.array(y)
            v = np.array(v)
            self.gpiteration = len(x)
            bFirstEval = False
        else:
            x = None
            y = None
            v = None
            self.gpiteration = 0
            bFirstEval = True

        hyperpars = np.ones([numpars + 1])
        # the zeroth hyper bound is associated with a signal variance
        # the others with the length scales of the parameter inputs
        hyper_bounds = np.array([[0.001, 100]] * (numpars + 1))
        for i in range(len(parlimits)):
            delta = parlimits[i][1] - parlimits[i][0]
            hyper_bounds[i + 1] = [delta * 1e-3, delta * 1e1]

        self.my_ae = AutonomousExperimenterGP(parlimits, hyperpars, hyper_bounds,
                                              init_dataset_size=self.gpcam_init_dataset_size,
                                              instrument_func=self.gpcam_instrument,
                                              acq_func=self.acq_func,  # optional_acq_func,
                                              # cost_func = optional_cost_function,
                                              # cost_update_func = optional_cost_update_function,
                                              x=x, y=y, v=v,
                                              # cost_func_params={"offset": 5.0, "slope": 10.0},
                                              kernel_func=None, use_inv=True,
                                              communicate_full_dataset=False, ram_economy=True)

        # save and evaluate initial data set if it has been freshly calculate
        if bFirstEval:
            self.save_results_gpcam(self.spath)
            self.gpcam_prediction(self.my_ae)

        while len(self.my_ae.x) < self.gpcam_iterations:
            print("length of the dataset: ", len(self.my_ae.x))
            self.my_ae.train(method="global", max_iter=10000)  # or not, or both, choose "global","local" and "hgdl"
            # update hyperparameters in case they are optimized asynchronously
            self.my_ae.train(method="local")  # or not, or both, choose between "global","local" and "hgdl"
            # training and client can be killed if desired and in case they are optimized asynchronously
            # self.my_ae.kill_training()
            if self.gpcam_step is not None:
                target_iterations = len(self.my_ae.x) + self.gpcam_step
                retrain_async_at = []
            else:
                target_iterations = self.gpcam_iterations
                retrain_async_at = np.logspace(start=np.log10(len(self.my_ae.x)),
                                               stop=np.log10(self.gpcam_iterations / 2), num=3, dtype=int)
            # run the autonomous loop
            self.my_ae.go(N=target_iterations,
                          retrain_async_at=retrain_async_at,
                          retrain_globally_at=[],
                          retrain_locally_at=[],
                          acq_func_opt_setting=lambda number: "global" if number % 2 == 0 else "local",
                          training_opt_max_iter=20,
                          training_opt_pop_size=10,
                          training_opt_tol=1e-6,
                          acq_func_opt_max_iter=20,
                          acq_func_opt_pop_size=20,
                          acq_func_opt_tol=1e-6,
                          number_of_suggested_measurements=1,
                          acq_func_opt_tol_adjust=0.1)

            # training and client can be killed if desired and in case they are optimized asynchronously
            if self.gpcam_step is None:
                self.my_ae.kill_training()
            self.save_results_gpcam(self.spath)
            self.gpcam_prediction(self.my_ae)

    def run_optimization_grid(self):
        # Grid search
        # every index has at least one result before re-analyzing any data point (refinement)
        bRefinement = False
        while True:
            bWorkedOnAnyIndex = self.gridsearch_iterate_over_all_indices(bRefinement)
            if not bWorkedOnAnyIndex:
                if not bRefinement:
                    # all indices have the minimum number of iterations -> start refinement
                    bRefinement = True
                else:
                    # done with refinement
                    break

            if self.bClusterMode or self.bFetchMode:
                # never repeat iterations on cluster or when just calculating entropies
                break

        # wait for all jobs to finish
        if self.bClusterMode:
            while self.joblist:
                self.waitforjob(bFinish=True)
