from gpcam.autonomous_experimenter import AutonomousExperimenterGP
from gpcam import GPOptimizer
from os import path, mkdir
import concurrent.futures
import json
import math
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
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
                 parallel_measurements=1, resume=False, show_support_points=True, gp_discrete_points=None):
        """
        Initialize the GP class.
        :param exp_par: (Pandas dataframe or json or dict) Exploration parameter dataframe with rows: "name", "type",
                        "value", "lower_opt", "upper_opt", "step_opt"
        :param optimizer: (string) Optimizer name 'gpcam', 'gpCAM' (redundant), or 'grid'
        :param gp_discrete_points: (ndarray or list) of shape V x D, where D is the length of the input vector that
                                   defines the grid of possible measurement points. If gp_discrete points is provided,
                                   there still needs to be an exp_par dataframe for plotting and naming.
        :param resume: (bool, default False) loads previous results from the storage path.
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
        self.show_support_points = show_support_points
        # status dict of the type {"status": "running", "progress": "0%", "cancelled": False}
        self.task_dict = {}

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
        self.measurement_inprogress = None
        self.measurement_requests_queue = Queue()

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
            self.hyper_bounds = None

            if not resume:
                columns = ['parameter names', 'position', 'value', 'variance']
                self.gpCAMstream = pd.DataFrame(columns=columns)
                self.gpiteration = 0
            else:
                self.results_io(load=True)

        self.prediction_gpcam = np.zeros(self.steplist)

    def do_measurement(self, optpars, it_label):
        """
        This function performs the actual measurement and needs to be implemented in each subclass. Here, a test
        function providing virtual data is provided
        :param optpars: specific set of parameters for the measurement
        :param it_label: a label for the current iteration
        :return: (result, variance) measurement result
        """
        result = 0
        for par in optpars:
            result += optpars[par] * 2 * np.pi
        variance = np.abs(result * 0.025) + 1e-7

        '''
        argument = 0
        valid = True
        last_par = None

        for par in optpars:
            if last_par is None:
                last_par = optpars[par]
            else:
                if optpars[par] > last_par:
                    valid = False
                last_par = optpars[par]

            d = 1 - argument
            if optpars[par] > 1:
                p = 1
            elif optpars[par] < 0:
                p = 0
            else:
                p = optpars[par]

            argument += d * p
        if valid:
            result = np.sin(argument*6)
            variance = np.abs(result * 0.025) + 0.0000001
        else:
            result = 0
            variance = 0.0000001
        '''

        time.sleep(0.1)
        return result, variance

    def gpcam_init_ae(self):
        parlimits = self.exp_par[['lower_opt', 'upper_opt']].to_numpy()
        numpars = len(parlimits)
        hyperpars = np.ones([numpars + 2])
        # the zeroth hyper bound is associated with a signal variance for the kernel
        # the others with the length scales of the parameter inputs
        self.hyper_bounds = np.array([[0.001, 100]] * (numpars + 2))
        for i in range(len(parlimits)):
            delta = parlimits[i][1] - parlimits[i][0]
            self.hyper_bounds[i + 1] = [delta * 1e-3, delta * 1e1]

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
            firsteval = False
        else:
            x = None
            y = None
            v = None
            self.gpiteration = 0
            firsteval = True

        '''
        Left-over arguments from ae initialization:
                init_dataset_size=self.gpcam_init_dataset_size,
                instrument_function=self.gpcam_instrument,
                # cost_func = optional_cost_function,
                # cost_update_func = optional_cost_update_function,
                x_data=x, y_data=y, noise_variances=v,
                # cost_func_params={"offset": 5.0, "slope": 10.0},
                kernel_function=None, calc_inv=True,
                communicate_full_dataset=False, ram_economy=True)
        '''

        return firsteval

    def gpcam_instrument(self, request_queue, in_progress_list, result_queue, gp_iteration):
        """
        The gpcam instrument function that will receive a single data point for measurement. It saves the in-progress
        information and the measurement values and variances to a log list and a result queue, respectively. As this
        function is meant to run in its separate process, these three objects are the only ones shared with the parent.
        :param request_queue: (multiprocessing.Queue) a queue to guarantee to contain a measurement point for evaluation
        :param in_progress_list: (list) list of in_progress measurement points
        :param result_queue: (multiprocessing.Queue) a queue to store the measurement results
        :param gp_iteration: (int) the current iteration
        :return: no return value
        """

        # print("This is the current length of the data received by gpCAM: ", len(data))
        # print("Suggested by gpCAM: ", data)

        data = request_queue.get()
        current_task_data = (None, data, gp_iteration)
        in_progress_list.append(current_task_data)
        self.iterations_inprogress_save_to_file()

        # do the measurment
        value, variance = self.work_on_iteration(current_task_data)

        in_progress_list.remove(current_task_data)

        # Entries for None results will not be returned as they would raise a GPCam error
        # abortion can be due to measurement failure or task cancellation
        if value is not None:
            entry = {'parameter names': self.exp_par['name'].to_list(), 'position': data, 'value': value,
                     'variance': variance}
            result_queue.put(entry)

        return

    def gpcam_optimization_loop(self):
        # Using the gpCAM global optimizer, follows the example from the gpCAM website
        bFirstEval = self.gpcam_init_ae()

        # save and evaluate initial data set if it has been freshly calculate
        if bFirstEval:
            self.results_io()
            self.gpcam_prediction()
            self.gpcam_plot()

        opt_obj = self.gpcam_train_async()

        while len(self.my_ae.x_data) < self.gpcam_iterations and not self.task_dict.get("cancelled", False):
            print('gpCAM main loop with abortion signal {}'.format(self.task_dict.get("cancelled", False)))

            # can we start another measurment?
            if not self.measurement_requests_queue.empty() and (len(self.measurement_inprogress) <
                                                                self.parallel_measurements):
                # update hyperparameters
                self.my_ae.update_hyperparameters(opt_obj)
                print('Hyperparamters:')
                print(self.my_ae.get_hyperparameters())

                # get the next measurement point based on the discrete input grid
                next_point = self.my_ae.ask(
                    self.gp_discrete_points,
                    n=1,
                    acquisition_function=self.acq_func
                )
                self.measurement_requests_queue.put(next_point['x'])
                self.gpiteration += 1

                p = Process(
                    target=self.gpcam_instrument,
                    args=(
                        self.measurement_requests_queue,
                        self.measurement_inprogress,
                        self.measurement_results_queue,
                        self.gpiteration
                    )
                )
                p.start()

            elif not self.measurement_results_queue.empty():
                result = self.measurement_results_queue.get()
                x = result['position']
                y = result['value']
                v = result['variance']
                self.my_ae.tell(x, y, v, append=True)
                self.gpCAMstream.loc[len(self.gpCAMstream)] = result
                self.results_io()
                self.gpcam_prediction()
                self.gpcam_plot()

            elif not self.measurement_inprogress:
                # delete iterations in progress log file
                self.iterations_inprogress_delete_file()

            else:
                # nothing to do
                time.sleep(1)

        self.my_ae.stop_training(opt_obj)

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
        # create a flattened array of all positions to be evaluated, maximize the use of numpy
        prediction_positions = np.array(self.axes[0])
        for i in range(1, len(self.axes)):
            a = np.array([prediction_positions] * len(self.axes[i]))
            # transpose the first two axes of a only
            newshape = np.linspace(0, len(a.shape) - 1, len(a.shape), dtype=int)
            newshape[0] = 1
            newshape[1] = 0
            a = np.transpose(a, newshape)
            b = np.array([self.axes[i]] * prediction_positions.shape[0])
            prediction_positions = np.dstack((a, b))
            # now flatten the first two dimensions
            newshape = list(prediction_positions.shape[1:])
            newshape[0] = newshape[0] * prediction_positions.shape[0]
            newshape = tuple(newshape)
            prediction_positions = np.reshape(prediction_positions, newshape)

        res = self.my_ae.posterior_mean(prediction_positions)
        f = res["f(x)"]
        self.prediction_gpcam = f.reshape(self.steplist)

    def gpcam_train(self, method='mcmc'):
        print("length of the dataset: ", len(self.my_ae.x_data))
        self.my_ae.train(
            hyperparameter_bounds=self.hyper_bounds,
            method=method,
            max_iter=10000
        )

    def gpcam_train_async(self,):
        print("length of the dataset: ", len(self.my_ae.x_data))
        opt_obj = self.my_ae.train(
            hyperparameter_bounds=self.hyper_bounds,
            max_iter=10000
        )
        return opt_obj

    def gridsearch_iterate_over_all_indices(self, refinement=False):
        bWorkedOnIndex = False
        # the complicated iteration syntax is due the unknown dimensionality of the results space / arrays
        it = np.nditer(self.results, flags=['multi_index'])
        work_on_it_list = []
        work_on_itindex_list = []
        while not it.finished and not self.measurement_aborted:
            itindex = it.multi_index
            # run iteration if it is first time or the value in results is nan
            print('index : {}'.format(itindex))
            invalid_result = np.isnan(self.results[itindex])
            insufficient_iterations = self.n_iter[itindex] < self.miniter
            # Do we need to work on this particular index?
            if invalid_result or insufficient_iterations:
                bWorkedOnIndex = True
                work_on_it_list.append((it.copy(), None, None))
                work_on_itindex_list.append(itindex)

            it.iternext()
            if (it.finished and work_on_it_list) or len(work_on_it_list) == self.parallel_measurements:
                print('parallel measurements: {}'.format(self.parallel_measurements))
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    print('submit jobs ...')
                    results = list(executor.map(self.work_on_iteration, work_on_it_list))
                    print('receive jobs ...')
                for i, entry in enumerate(results):
                    if entry[0] is not None:
                        self.results[work_on_itindex_list[i]] = entry[0]
                        self.variances[work_on_itindex_list[i]] = entry[1]
                        self.n_iter[work_on_itindex_list[i]] += 1
                    else:
                        self.measurement_aborted = True
                self.results_io()
                path1 = path.join(self.spath, 'plots')
                filename = path.join(path1, 'prediction_gpcam')
                self.results_plot(np.nan_to_num(self.results, nan=0), arr_variance=np.nan_to_num(self.variances, nan=0.0),
                                  filename=filename)
                work_on_it_list = []
                work_on_itindex_list = []

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
            optpars, itlabel = self.yield_optpars_label(argument)
            optpars['storage label'] = itlabel
            new_row = pd.DataFrame([optpars])
            output_df = pd.concat([output_df, new_row], ignore_index=True)

        with open(path.join(self.spath, 'results', 'current_iterations.pkl'), 'wb') as file:
            pickle.dump(output_df, file)

    def run(self, task_dict, measurements_inprogress, print_queue=None):
        self.task_dict = task_dict
        self.task_dict['status'] = 'running'
        self.measurement_inprogress = measurements_inprogress

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

    def work_on_iteration(self, arguments):
        """
        Performs a single interation (measurement) of either a grid search or gpcam
        :param arguments: (tuple) it object for grid search, position (x_value) for gpcam, gplabel for gpcam. Not
                          applying fields can be none, see also yield_optpars_label
        :return: (float, float) result and variance of the performed measurement
        """
        print(self.task_dict)
        if self.task_dict['cancelled']:
            self.task_dict['status'] = 'stopping'
            self.measurement_aborted = True
            return None, None

        optpars, itlabel = self.yield_optpars_label(arguments)
        print(itlabel, optpars)
        try:
            result, variance = self.do_measurement(optpars, itlabel)
        except RuntimeError as e:
            print('Measurement failed outside of GP {}'.format(e))
            self.task_dict['status'] = 'failure'
            self.measurement_aborted = True
            result = None
            variance = None

        return result, variance

    def yield_optpars_label(self, arguments):
        """
        Helper function that creates a dictionary of parameter names and their values plus a label for the current
        iteration.
        :param arguments: (tuple) it object for grid search, position (x_value) for gpcam, gplabel for gpcam. Not
                          applying fields can be none
        :return: (dict, label) dictionary with parameter names as keys and their values as values and a label used
                 for storage of the iteration
        """

        it, position, gpiteration = arguments

        if it is not None:
            # provide some running index for grid search for possible use as storage label
            itindex = it.multi_index
            itlabel = it.iterindex
        else:
            # gpcammode
            itlabel = gpiteration

        optpars = {}
        # cycle through all parameters
        for isim, row in enumerate(self.exp_par.itertuples()):
            lopt = self.exp_par.loc[self.exp_par['name'] == row.name, 'lower_opt'].iloc[0]
            stepopt = self.exp_par.loc[self.exp_par['name'] == row.name, 'step_opt'].iloc[0]
            # value = self.exp_par.loc[self.exp_par['name'] == row.name, 'value'].iloc[0]

            if it is not None:
                # grid mode, calculate value from evaluation grid and index
                optvalue = lopt + stepopt * it.multi_index[isim]
                # print(self.steppar['unique_name'], '  ', simvalue, '\n')
            else:
                # gpcam mode, use position suggested by gpcam
                optvalue = position[isim]

            optpars[row.name] = optvalue

        return optpars, itlabel



