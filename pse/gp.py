from gpcam.autonomous_experimenter import AutonomousExperimenterGP
from gpcam import GPOptimizer
from os import path, mkdir
import concurrent.futures
import json
import math
import matplotlib.pyplot as plt
#from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue
import numpy as np
import os.path
import pickle
import pandas as pd
import time


def nice_interval(start=0, stop=1, step=None, numsteps=10):
    """
    Paul Kienzle's method to obtain nicely spaced intervals for plot axes
    """

    if step is None:
        step = (stop-start)/numsteps

    sign = 1.0
    if step < 0:
        sign = -1.0
        step = step * (-1)

    if step == 0:
        return [start, stop+1]

    exponent = math.floor(np.log(step)/np.log(10))
    mantisse = step / pow(10, exponent)

    new_mantisse = 1
    if math.fabs(mantisse-2) < math.fabs(mantisse-new_mantisse):
        new_mantisse = 2
    if math.fabs(mantisse-5) < math.fabs(mantisse-new_mantisse):
        new_mantisse = 5
    if math.fabs(mantisse-10) < math.fabs(mantisse-new_mantisse):
        new_mantisse = 10

    new_step = sign * new_mantisse * pow(10, exponent)
    new_start = math.floor(start/new_step) * new_step
    new_stop = math.ceil(stop/new_step) * new_step

    return np.arange(new_start, new_stop+0.05*new_step, new_step)


def save_plot_1d(x, y, dy=None, xlabel='', ylabel='', color='blue', filename="plot", ymin=None, ymax=None, levels=5,
                 niceticks=False, keep_plots=False):
    import matplotlib.pyplot as plt
    import matplotlib

    if ymin is None:
        ymin = np.amin(y)
    if ymax is None:
        ymax = np.amax(y)

    font = {'family': 'sans-serif', 'weight': '200', 'size': 14}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots()
    if dy is None:
        ax.plot(x, y, color=color)
    else:
        ax.errorbar(x, y, dy, color=color)
    if niceticks:
        bounds = nice_interval(start=ymin, stop=ymax, numsteps=levels)
        ax.set_yticks(bounds)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(scilimits=(-3, 3), useMathText=True)

    plt.tight_layout()

    if keep_plots:
        i = 0
        while os.path.isfile(filename + str(i) + '.png'):
            i += 1
        filename = filename + str(i)

    plt.savefig(filename + '.pdf')
    plt.savefig(filename + '.png')
    plt.close("all")


def save_plot_2d(x, y, z, xlabel, ylabel, color, filename='plot', zmin=None, zmax=None, levels=20, mark_maximum=False,
                 keep_plots=False, support_points=None):
    import matplotlib.pyplot as plt
    import matplotlib

    if zmin is None:
        zmin = np.amin(z)
    if zmax is None:
        zmax = np.amax(z)

    bounds = nice_interval(start=zmin, stop=zmax, numsteps=levels)

    font = {'family': 'sans-serif', 'weight': '200', 'size': 14}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots()
    cs = ax.contourf(x, y, z, cmap=color, vmin=bounds[0], vmax=bounds[-1], levels=bounds, extend='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(scilimits=(-3, 3), useMathText=True)
    fig.colorbar(cs)

    if support_points is not None:
        plt.scatter(support_points[:, 1], support_points[:, 0], s=10, c='k')

    if mark_maximum:
        index = np.unravel_index(z.argmax(), z.shape)
        plt.text(x[index[1]], y[index[0]], 'x', horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()

    if keep_plots:
        i = 0
        while os.path.isfile(filename + str(i) + '.png'):
            i += 1
        filename = filename + str(i)

    plt.savefig(filename + '.pdf')
    plt.savefig(filename + '.png')
    plt.close("all")


class Gp:
    def __init__(self, exp_par, storage_path=None, acq_func="variance", gpcam_iterations=50,
                 gpcam_init_dataset_size=20, gpcam_step=1, keep_plots=False, miniter=1, optimizer='gpcam',
                 parallel_measurements=1, resume=False, signal_estimate=10, show_support_points=True,
                 train_global_every=None, gp_discrete_points=None):
        """
        Initialize the GP class.
        :param exp_par: (Pandas dataframe or json or dict) Exploration parameter dataframe with rows: "name", "type",
                        "value", "lower_opt", "upper_opt", "step_opt"
        :param optimizer: (string) Optimizer name 'gpcam', 'gpCAM' (redundant), or 'grid'
        :param gp_discrete_points: (ndarray or list) of shape V x D, where D is the length of the input vector that
                                   defines the grid of possible measurement points. If gp_discrete points is provided,
                                   there still needs to be an exp_par dataframe for plotting and naming.
        :param resume: (bool, default False) loads previous results from the storage path.
        :param signal_estimate: (float) estimated max signal (max - min) for gp hyperparameter setting
        """
        self.acq_func = acq_func
        self.gpcam_iterations = gpcam_iterations
        self.gpcam_init_dataset_size = gpcam_init_dataset_size
        self.gpcam_step = gpcam_step
        self.gpiteration = 0
        # global flag for an occuring measurement failure
        self.measurement_aborted = False
        self.keep_plots = keep_plots
        self.miniter = miniter
        self.optimizer = optimizer
        if self.optimizer == 'gpCAM':
            self.optimizer = 'gpcam'
        self.parallel_measurements = parallel_measurements
        self.signal_estimate = signal_estimate
        self.show_support_points = show_support_points
        # status dict of the type {"status": "running", "progress": "0%", "cancelled": False}
        self.task_dict = {}
        if train_global_every is not None:
            self.train_global_every = train_global_every
        else:
            self.train_global_every = self.parallel_measurements


        self.my_ae = None

        # directory for storing results
        if storage_path is None:
            storage_path = os.getcwd()
        if not path.isdir(storage_path):
            mkdir(storage_path)
        self.spath = storage_path

        result_path = os.path.join(self.spath, 'results')
        if not path.isdir(result_path):
            mkdir(result_path)
        plot_path = os.path.join(self.spath, 'plots')
        if not path.isdir(plot_path):
            mkdir(plot_path)

        # Pandas dataframe of exploration parameters
        if not isinstance(exp_par, pd.DataFrame):
            exp_par = pd.DataFrame(exp_par)

        self.all_par = exp_par
        self.exp_par = exp_par
        # keep only rows that are being explored
        self.exp_par = self.exp_par[self.exp_par['optimize']]
        columns_to_keep = ['name', 'type', 'value', 'lower_opt', 'upper_opt', 'step_opt']
        self.exp_par = self.exp_par[columns_to_keep]

        # List of exploration steps and axes (for plotting in gpcam or for the gridsearch)
        self.steplist = []
        self.axes = []
        for row in self.exp_par.itertuples():
            steps = int((row.upper_opt - row.lower_opt) / row.step_opt) + 1
            self.steplist.append(steps)
            axis = []
            for i in range(steps):
                axis.append(row.lower_opt + i * row.step_opt)
            self.axes.append(axis)

        # result queue for communicating with the measurment processes
        self.measurement_results_queue = Queue()
        self.measurement_inprogress = []

        # initialize new run if result dir is empty
        if resume and not os.listdir(path.join(self.spath, 'results')):
            resume = False

        if self.optimizer == 'grid':
            if not resume:
                self.results = np.full(self.steplist, np.nan)
                self.variances = np.full(self.steplist, np.nan)
                self.n_iter = np.zeros(self.steplist)
                # do a first save of results
                self.results_io(load=False)
            else:
                self.results_io(load=True)

        elif self.optimizer == 'gpcam':
            # create discrete points for the GPOptimizer, if none were provided
            if gp_discrete_points is not None:
                self.gp_discrete_points = gp_discrete_points
            else:
                grids = np.meshgrid(*self.axes, indexing='ij')
                self.gp_discrete_points = np.stack(grids, axis=-1).reshape(-1, len(self.axes))
                # make this numpy array into a list of numpy arrays along axis 0
                self.gp_discrete_points = [self.gp_discrete_points[i] for i in range(self.gp_discrete_points.shape[0])]
            self.hyper_bounds = None

            if not resume:
                columns = ['parameter names', 'position', 'value', 'variance']
                self.gpCAMstream = pd.DataFrame(columns=columns)
                self.gpiteration = 0
            else:
                self.results_io(load=True)

        self.prediction_gpcam = np.zeros(self.steplist)
        self.prediction_var_gpcam = np.zeros(self.steplist)

    def do_measurement(self, optpars, it_label, entry, q):
        """
        This function performs the actual measurement and needs to be implemented in each subclass. Here, a test
        function providing virtual data is provided
        :param optpars: specific set of parameters for the measurement
        :param it_label: a label for the current iteration
        :entry (dict) the record without the results that will be deposited in the result queue
        :param q: (multiprocessing.Queue) the result queue
        :return: (result, variance) measurement result
        """

        def ackley_nd(x, a=20, b=0.2, c=2 * np.pi):
            """
            Computes the n-dimensional Ackley function as a test for gp.
            :param x: np.ndarray
                      A 1D array of shape (n,) representing a single point in n-dimensional space,
                      or a 2D array of shape (m, n) representing m points in n-dimensional space.
            :param a, b, c: float
                            Standard Ackley function parameters.
            :return: float or np.ndarray
                     The Ackley function value(s) at the input point(s).
            """
            x = np.atleast_2d(x)
            n = x.shape[1]

            # Compute sum of squares and sum of cosines
            sum_sq_term = np.sum(x ** 2, axis=1)
            cos_term = np.sum(np.cos(c * x), axis=1)

            # Compute Ackley function
            term1 = -a * np.exp(-b * np.sqrt(sum_sq_term / n))
            term2 = -np.exp(cos_term / n)
            result = term1 + term2 + a + np.e

            # Return scalar if input was 1D
            return result[0] if result.shape[0] == 1 else result

        result = 0
        testpars = []
        for i, par in enumerate(optpars):
            ackley_par = (optpars[par] - self.axes[i][0])
            ackley_par /= (self.axes[i][-1] - self.axes[i][0])
            testpars.append(ackley_par)
        result = float(ackley_nd(np.array(testpars)))
        variance = 0.001
        # add noise term
        result += np.random.normal(loc=0.0, scale=np.sqrt(variance))
        time.sleep(1)

        # THESE THREE LINES NEED DO BE PRESENT IN EVERY DERIVED METHOD
        # TODO: Make this post-logic seemless for inheritance
        entry['value'] = result
        entry['variance'] = variance
        q.put(entry)

        return result, variance

    def gpcam_init_ae(self, just_gpcamstream=False):
        if not just_gpcamstream:
            # Compute Input Ranges
            parlimits = self.exp_par[['lower_opt', 'upper_opt', 'step_opt']].to_numpy()
            ranges = [p[1] - p[0] for p in parlimits]

            # Adjust Length Scales
            #     A good starting point for each length scale is ~20–30% of the input range
            #     A good bound is typically [1% to 100%][1% to 100%] of the input range
            length_scale_init = [0.2 * r for r in ranges]
            length_scale_bounds = [(0.01 * r, r) for r in ranges]

            # Amplitude (signal variance) and Noise
            # These don’t depend on the input range but on the output scale of your function. If you have rough estimates
            # of the function's values:
            #     Use amplitude_init ≈ std(f(x))
            #     Use noise_init ≈ measurement error variance (or small, if deterministic)
            amplitude_init = 0.1 * self.signal_estimate
            amplitude_bounds = (1e-2, self.signal_estimate)
            noise_init = 1e-6
            noise_bounds = (1e-8, 1e-2)

            hyperpars = np.array([amplitude_init] + length_scale_init)
            self.hyper_bounds = np.array([amplitude_bounds] + length_scale_bounds)

            self.my_ae = GPOptimizer(
                init_hyperparameters=hyperpars,
                gp2Scale=False,
                calc_inv=False,
                ram_economy=False,
                args=None
            )

        # those are Pandas dataframe colunns
        x = self.gpCAMstream['position'].to_numpy()
        if x.size != 0 and x.dtype == object:
            x = np.stack(x)
        y = self.gpCAMstream['value'].to_numpy()
        v = self.gpCAMstream['variance'].to_numpy()

        if len(x) >= 1:
            # use any previously computed results
            x = np.array(x)
            y = np.array(y)
            v = np.array(v)
            self.gpiteration = len(x)
            # tell the optimizer the data that was already collected
            self.my_ae.tell(x, y, v, append=False)
        else:
            x = None
            y = None
            v = None
            self.gpiteration = 0
        return

    def gpcam_optimization_loop(self):
        def collect_measurement(gpcam_initialized=False):
            result = self.measurement_results_queue.get()
            x = np.array([result['position']])
            y = np.array([result['value']])
            v = np.array([result['variance']])
            print('Colected measurement results x, y, v: {}, {}, {}'.format(x, y, v))
            print('\n')
            # This is replacing the point that was previously preset with predicted value
            self.gpCAMstream.loc[len(self.gpCAMstream)] = result
            self.gpcam_init_ae(just_gpcamstream=True)
            self.results_io()
            # remove the current task from the in-progress list
            self.measurement_inprogress = [item for item in self.measurement_inprogress
                                           if not np.array_equal(item[1], result['position'])]
            progress = float(len(self.gpCAMstream)) / float(self.gpcam_iterations)
            self.task_dict['progress'] = '{:.2f}%'.format(progress * 100)
            if gpcam_initialized:
                if len(self.gpCAMstream) % self.train_global_every == 0:
                    # reinitialize gp from gpCAM stream in case double-measured points were
                    # eliminated due to the blocking scheme with prediction data
                    self.gpcam_init_ae()
                    self.gpcam_train(method='global')
                else:
                    self.gpcam_train(method='local')
                self.gpcam_prediction()
                self.gpcam_plot()
            return

        # Using the gpCAM global optimizer
        self.gpcam_init_ae()

        # save and evaluate initial data set if it has been freshly calculate
        print('Checking if sufficient initial measurements.')
        while self.gpiteration < self.gpcam_init_dataset_size and not self.task_dict.get("cancelled", False):

            if not self.measurement_results_queue.empty():
                collect_measurement(gpcam_initialized=False)
            elif len(self.measurement_inprogress) < self.parallel_measurements:
                print('Preparing initial measurement #{}.'.format(self.gpiteration))
                if self.gpiteration == 0:
                    next_point = self.gp_discrete_points[0]
                else:
                    next_point = self.gp_discrete_points[int(np.random.random() * len(self.gp_discrete_points))]
                self.work_on_iteration(next_point, self.gpiteration)
                self.gpiteration += 1
            else:
                # nothing to do
                time.sleep(5)

        print("Continue to gpCAM measurments...")

        self.gpcam_train(method='global')
        self.gpcam_prediction()
        self.gpcam_plot()

        while len(self.my_ae.x_data) < self.gpcam_iterations and not self.task_dict.get("cancelled", False):
            # print('gpCAM main loop with abortion signal {}'.format(self.task_dict.get("cancelled", False)))
            # print("length of the dataset: ", len(self.my_ae.x_data))

            # first collect any finished measurements as they update the model
            if not self.measurement_results_queue.empty():
                collect_measurement(gpcam_initialized=True)
            # can we start another measurment?
            elif len(self.measurement_inprogress) < self.parallel_measurements:
                # update hyperparameters
                print('Hyperparamters:')
                print(self.my_ae.get_hyperparameters())
                n = self.parallel_measurements - len(self.measurement_inprogress)
                n_max = self.parallel_measurements

                submit_counter = 0
                for i in range(n_max):
                    next_points = self.my_ae.ask(
                        self.gp_discrete_points,
                        n=1,
                        method='global',
                        acquisition_function=self.acq_func,
                        info=True,
                    )
                    self.work_on_iteration(next_points['x'][0], self.gpiteration)
                    self.gpiteration += 1
                    submit_counter += 1
                    # immediately block this point by updating the gp with the theoretical result
                    # will be later replaced
                    next_point = np.array(next_points['x'][0])
                    pred_points = next_point.reshape(1, -1)
                    pred_mean = self.my_ae.posterior_mean(pred_points)["f(x)"]
                    pred_var = np.array([self.signal_estimate * 1e-7])
                    self.my_ae.tell(pred_points, pred_mean, pred_var, append=True)
                    self.gpcam_train(method='local')
                    if submit_counter == n:
                        break

            else:
                if not self.measurement_inprogress:
                    # delete iterations in progress log file
                    self.iterations_inprogress_delete_file()
                # nothing to do
                time.sleep(5)

    def gpcam_plot(self):
        path1 = path.join(self.spath, 'plots')
        if not path.isdir(path1):
            mkdir(path1)
        # self.results_plot(self.prediction_gpcam, filename=path.join(path1, 'prediction_gpcam'), mark_maximum=True)

        if self.show_support_points:
            support_points = self.gpCAMstream['position'].to_numpy()
            if support_points.dtype == object:
                support_points = np.stack(support_points)
        else:
            support_points = None

        self.results_plot(self.prediction_gpcam,
                          filename=path.join(path1, 'prediction_gpcam'), mark_maximum=True,
                          support_points=support_points)

    def gpcam_prediction(self):
        """
        Creates a gp model prediction on all input points defined by the axes and steps of the optimization
        problem defined during object intialization.
        :return: no return value
        """

        # TODO: This creates all input positions of the range defined by the axes and stepsizes, if there is a non-
        #   Eucledian input space for the gpOptimizer provided, this is ignored here. The idea is that non-Eucledian
        #   plotting is a nightmare in this module. However, some filtering against the non-Euclidean input might be
        #   desirable.

        mesh = np.meshgrid(*self.axes, indexing='ij')
        stacked = np.stack(mesh, axis=-1)
        prediction_positions = np.array(stacked.reshape(-1, len(self.axes)), dtype=np.float32)

        mean = self.my_ae.posterior_mean(prediction_positions)["f(x)"]
        var = self.my_ae.posterior_covariance(prediction_positions, variance_only=True, add_noise=False)["v(x)"]
        self.prediction_gpcam = mean.reshape(self.steplist)
        self.prediction_var_gpcam = var.reshape(self.steplist)

    def gpcam_train(self, method='mcmc'):
        self.my_ae.train(
            hyperparameter_bounds=self.hyper_bounds,
            method=method,
            max_iter=10000
        )

    def gpcam_train_async(self):
        opt_obj = self.my_ae.train_async(
            hyperparameter_bounds=self.hyper_bounds,
            max_iter=10000
        )
        #self.my_ae.update_hyperparameters(opt_obj)
        return opt_obj

    def gridsearch_iterate_over_all_indices(self, refinement=False):
        def collect_measurement():
            result = self.measurement_results_queue.get()
            x = np.array([result['position']])
            y = np.array([result['value']])
            v = np.array([result['variance']])
            print('Colected measurement results x, y, v: {}, {}, {}'.format(x, y, v))
            print('\n')

            # recreate indices
            xflat = np.asarray(x).flatten()
            indices = [np.where(np.asarray(axis) == value)[0][0] for axis, value in zip(self.axes, xflat)]
            # the tuple() avoids fancy indexing since indices is a numpy array
            self.results[tuple(indices)] = y
            self.variances[tuple(indices)] = v
            self.n_iter[tuple(indices)] += 1

            # remove the current task from the in-progress list
            self.measurement_inprogress = [item for item in self.measurement_inprogress
                                           if not np.array_equal(item[1], result['position'])]
            progress = float(np.ravel_multi_index(itindex, self.steplist)) / float(np.prod(self.steplist))
            self.task_dict['progress'] = '{:.2f}%'.format(progress * 100)

            self.results_io()
            path1 = path.join(self.spath, 'plots')
            filename = path.join(path1, 'prediction_gpcam')
            self.results_plot(np.nan_to_num(self.results, nan=0), arr_variance=np.nan_to_num(self.variances, nan=0.0),
                              filename=filename)
            return

        bWorkedOnIndex = False
        # the complicated iteration syntax is due the unknown dimensionality of the results space / arrays
        it = np.nditer(self.results, flags=['multi_index'])

        while not it.finished and not self.measurement_aborted:
            # first collect any finished measurements
            if not self.measurement_results_queue.empty():
                collect_measurement()
            elif len(self.measurement_inprogress) < self.parallel_measurements:
                itindex = it.multi_index
                print('index : {}'.format(itindex))
                # run iteration if it is first time or the value in results is nan
                invalid_result = np.isnan(self.results[itindex])
                insufficient_iterations = self.n_iter[itindex] < self.miniter
                print('Result present: {}, Insufficient iterations: {}'.format(not invalid_result,
                                                                               insufficient_iterations))
                # Do we need to work on this particular index?
                if invalid_result or insufficient_iterations:
                    bWorkedOnIndex = True
                    position = [self.axes[n][itindex[n]] for n in range(len(self.axes))]
                    itlabel = np.ravel_multi_index(itindex, self.steplist)
                    self.work_on_iteration(position, itlabel)
                it.iternext()
            else:
                if not self.measurement_inprogress:
                    # delete iterations in progress log file
                    self.iterations_inprogress_delete_file()
                time.sleep(5)

        return bWorkedOnIndex

    def iterations_inprogress_delete_file(self):
        """
        Deletes any existing current iterations file in the results directory of the active project.
        :return: no return value
        """
        if os.path.exists(path.join(self.spath, 'results', 'current_iterations.pkl')):
            os.remove(path.join(self.spath, 'results', 'current_iterations.pkl'))

    def iterations_inprogress_save_to_file(self):
        """
        Saves the currently worked on iterations parameters to file for visualization. The data is obtained
        from the measurment in progress queue.
        :return: None
        """
        output_df = pd.DataFrame()
        for argument in self.measurement_inprogress:
            _, position, itlabel = argument
            optpars = {}
            # cycle through all parameters
            for isim, row in enumerate(self.exp_par.itertuples()):
                optvalue = position[isim]
                optpars[row.name] = optvalue
            optpars['storage label'] = itlabel
            new_row = pd.DataFrame([optpars])
            output_df = pd.concat([output_df, new_row], ignore_index=True)

        with open(path.join(self.spath, 'results', 'current_iterations.pkl'), 'wb') as file:
            pickle.dump(output_df, file)

    def run(self, task_dict):
        self.task_dict = task_dict
        self.task_dict['status'] = 'running'

        if self.optimizer == 'grid':
            self.run_optimization_grid()
        elif self.optimizer == 'gpcam':
            self.gpcam_optimization_loop()
        else:
            self.task_dict['status'] = 'failure'
            raise NotImplementedError('Unknown optimization method')

        if self.measurement_aborted:
            if self.task_dict['status'] == 'failure':
                return False

        # stopping or running status reset to idle
        self.task_dict['status'] = 'idle'
        return True

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

    def results_io(self, load=False):
        def pack(nparray):
            return_data = {
                "array": nparray.tolist(),
                "dtype": str(nparray.dtype),
                "shape": nparray.shape
            }
            return return_data

        if self.optimizer == 'grid':
            if load:
                with open(path.join(self.spath, 'results', 'pse_grid_results.pkl'), 'rb') as file:
                    self.results = pickle.load(file)
                with open(path.join(self.spath, 'results', 'pse_grid_variances.pkl'), 'rb') as file:
                    self.variances = pickle.load(file)
                with open(path.join(self.spath, 'results', 'pse_grid_iterations.pkl'), 'rb') as file:
                    self.n_iter = pickle.load(file)
            else:
                with open(path.join(self.spath, 'results', 'pse_grid_results.pkl'), 'wb') as file:
                    pickle.dump(self.results, file)
                with open(path.join(self.spath, 'results', 'pse_grid_variances.pkl'), 'wb') as file:
                    pickle.dump(self.variances, file)
                with open(path.join(self.spath, 'results', 'pse_grid_iterations.pkl'), 'wb') as file:
                    pickle.dump(self.n_iter, file)
                # create a json output
                results_data = pack(self.results)
                variances_data = pack(self.variances)
                iterations_data = pack(self.n_iter)
                json_out = {'results': results_data, 'variances': variances_data, 'iterations': iterations_data}
                with open(path.join(self.spath, 'results', 'pse_grid_results.json'), 'w') as file:
                    json.dump(json_out, file)
        elif self.optimizer == 'gpcam':
            if load:
                with open(path.join(self.spath, 'results', 'gpCAMstream.pkl'), 'rb') as file:
                    self.gpCAMstream = pickle.load(file)
            else:
                with open(path.join(self.spath, 'results', 'gpCAMstream.pkl'), 'wb') as file:
                    pickle.dump(self.gpCAMstream, file)
                # create a json output
                path_name = path.join(self.spath, 'results', 'gpCAMstream.json')
                self.gpCAMstream.to_json(path_name, orient="records", indent=2)
        else:
            raise NotImplementedError('Unknown optimization method')

    def results_plot(self, arr_value, arr_variance=None, filename='plot', mark_maximum=False, valmin=None, valmax=None,
                     levels=20, niceticks=False, vallabel='z', support_points=None):

        # onecolormaps = [plt.cm.Greys, plt.cm.Purples, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds]
        ec = plt.colormaps['coolwarm']

        path1 = path.join(self.spath, 'plots')

        if len(arr_value.shape) == 1:
            ax0 = self.axes[0]
            sp0 = self.exp_par['name'].tolist()[0]
            if arr_variance is not None:
                dy = np.sqrt(arr_variance)
            else:
                dy = None
            save_plot_1d(ax0, arr_value, dy=dy, xlabel=sp0, ylabel=vallabel, filename=path.join(path1, filename),
                         ymin=valmin, ymax=valmax, levels=levels, niceticks=niceticks, keep_plots=self.keep_plots)

        elif len(arr_value.shape) == 2:
            # numpy array and plot axes are reversed
            ax1 = self.axes[0]
            ax0 = self.axes[1]
            sp1 = self.exp_par['name'].tolist()[0]
            sp0 = self.exp_par['name'].tolist()[1]
            save_plot_2d(ax0, ax1, arr_value, xlabel=sp0, ylabel=sp1, color=ec,
                         filename=path.join(path1, filename), zmin=valmin, zmax=valmax, levels=levels,
                         mark_maximum=mark_maximum, keep_plots=self.keep_plots, support_points=support_points)

        elif len(arr_value.shape) == 3 and arr_value.shape[0] < 6:
            ax2 = self.axes[1]
            ax1 = self.axes[2]
            sp2 = self.exp_par['name'].tolist()[1]
            sp1 = self.exp_par['name'].tolist()[2]
            for slice_n in range(arr_value.shape[0]):
                save_plot_2d(ax1, ax2, arr_value[slice_n], xlabel=sp1, ylabel=sp2, color=ec,
                             filename=path.join(path1, filename+'_'+str(slice_n)), zmin=valmin, zmax=valmax,
                             levels=levels, mark_maximum=mark_maximum, keep_plots=self.keep_plots)

        if len(arr_value.shape) >= 3:
            # plot projections onto two parameters at a time
            for i in range(len(self.exp_par)):
                for j in range(i):
                    ax2 = self.axes[i]
                    ax1 = self.axes[j]
                    sp2 = self.exp_par['name'].tolist()[i]
                    sp1 = self.exp_par['name'].tolist()[j]
                    projection = np.empty((self.steplist[i], self.steplist[j]))
                    for k in range(self.steplist[i]):
                        for ll in range(self.steplist[j]):
                            projection[k, ll] = np.take(np.take(arr_value, indices=k, axis=i), indices=ll, axis=j).max()
                    save_plot_2d(ax1, ax2, projection, xlabel=sp1, ylabel=sp2, color=ec,
                                 filename=path.join(path1, filename+'_'+sp1+'_'+sp2), zmin=valmin, zmax=valmax,
                                 levels=levels, mark_maximum=mark_maximum, keep_plots=self.keep_plots)

    def work_on_iteration(self, position, itlabel):
        """
        Performs a single interation (measurement) of either a grid search or gpcam
        :param position: (tuple, list) parameter values to measure next
        :param itlabel: (any) some iteration label to identify the current iteration
        :return: (float, float) result and variance of the performed measurement
        """
        current_task_data = (None, position, itlabel)
        self.measurement_inprogress.append(current_task_data)
        self.iterations_inprogress_save_to_file()

        print(self.task_dict)
        if self.task_dict['cancelled']:
            self.task_dict['status'] = 'stopping'
            self.measurement_aborted = True
            return None, None

        optpars = {}
        # cycle through all parameters
        for isim, row in enumerate(self.exp_par.itertuples()):
            optvalue = position[isim]
            optpars[row.name] = optvalue

        print(itlabel, optpars)
        try:
            q = self.measurement_results_queue
            entry = {'parameter names': self.exp_par['name'].to_list(), 'position': current_task_data[1],
                     'value': None, 'variance': None}
            p = Thread(
                target=self.do_measurement,
                args=(optpars, itlabel, entry, q)
            )
            p.start()
        except RuntimeError as e:
            print('Measurement failed outside of GP {}'.format(e))
            self.task_dict['status'] = 'failure'
            self.measurement_aborted = True
            self.measurement_inprogress.remove(current_task_data)

        return




