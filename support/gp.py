import os.path

from gpcam.autonomous_experimenter import AutonomousExperimenterGP
from os import path, mkdir

import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd


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


class gp:
    def __init__(self, exp_par, storage_path=None, acq_func="variance", calc_symmetric=False, gpcam_iterations=50,
                 gpcam_init_dataset_size=20, gpcam_step=None, keep_plots=False, miniter=1, optimizer='gpcam',
                 previous_results=None, show_support_points=False):
        """
        Initialize the GP class.
        :param exp_par: (Pandas dataframe) Exploration parameter dataframe with rows: "name", "type", "value",
                        "lower_opt", "upper_opt", "step_opt"
        :param optimizer: (string) Optimizer name 'gpcam', 'gpCAM' (redundant), or 'grid'
        :param previous_results: (DataFrame or numpy array) Previous results for gpcam or grid optimizer, respectively
        """
        self.acq_func = acq_func
        self.calc_symmetric = calc_symmetric
        self.gpcam_iterations = gpcam_iterations
        self.gpcam_init_dataset_size = gpcam_init_dataset_size
        self.gpcam_step = gpcam_step
        self.gpiteration = 0
        self.keep_plots = keep_plots
        self.miniter = miniter
        self.show_support_points = show_support_points

        self.my_ae = None

        # directory for storing results
        if storage_path is None:
            storage_path = os.getcwd()
        if not path.isdir(storage_path):
            mkdir(storage_path)
        self.spath = storage_path

        # Pandas dataframe of exploration parameters
        self.exp_par = exp_par
        columns_to_keep = ['name', 'type', 'value', 'lower_opt', 'upper_opt', 'step_opt']
        self.exp_par = self.exp_par[columns_to_keep]

        # List of exploration steps and axes (for plotting in gpcam or for the gridsearch)
        self.steplist = []
        self.axes = []
        for row in self.exp_par.itertuples():
            steps = int((row.u_sim - row.l_sim) / row.step_sim) + 1
            self.steplist.append(steps)
            axis = []
            for i in range(steps):
                axis.append(row.l_sim + i * row.step_sim)
            self.axes.append(axis)

        if optimizer == 'grid':
            if previous_results is None:
                self.results = np.full(self.steplist, np.nan)
            else:
                self.results = previous_results
            self.n_iter = np.zeros(self.steplist)

        elif optimizer == 'gpcam' or optimizer == 'gpCAM':
            if previous_results is None:
                columns = ['position', 'value', 'variance']
                self.gpCAMstream = pd.DataFrame(columns=columns)
            else:
                self.gpCAMstream = previous_results

        self.prediction_gpcam = np.zeros(self.steplist)

    def gpcam_instrument(self, data, Test=False):
        print("This is the current length of the data received by gpCAM: ", len(data))
        print("Suggested by gpCAM: ", data)
        for entry in data:
            if Test:
                # value = np.sin(np.linalg.norm(entry["position"]))
                # value = np.array(entry['position']).sum() / 1000
                value = (entry['position'][0] - entry['position'][1]) ** 2
                time0 = entry['position'][2]
                time1 = entry['position'][3]
                time2 = entry['position'][4]
                tf = 14400 / (time0 + time1 + time2)
                value += np.log10(time0 * tf) * 0.5
                value += np.log10(time1 * tf) * 1.5
                value += np.log10(time2 * tf) * 1
                entry['value'] = value
                print('Value: ', entry['value'])
                variance = None  # 0.01 * np.abs(entry['value'])
                entry['variance'] = variance
            else:
                # TODO: Implement return of experimental value here
                value = 1
                variance = None
                entry["value"] = value
                entry['variance'] = variance
                # entry["cost"]  = [np.array([0,0]),entry["position"],np.sum(entry["position"])]

            self.gpCAMstream['position'].append(entry['position'])
            self.gpCAMstream['value'].append(value)
            self.gpCAMstream['variance'].append(variance)
            self.save_results_gpcam()
            self.gpiteration += 1
        return data

    def gpcam_prediction(self, my_ae):
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

        res = my_ae.gp_optimizer.posterior_mean(prediction_positions)
        f = res["f(x)"]
        self.prediction_gpcam = f.reshape(self.steplist)

        path1 = path.join(self.spath, 'plots')
        if not path.isdir(path1):
            mkdir(path1)
        # self.plot_arr(self.prediction_gpcam, filename=path.join(path1, 'prediction_gpcam'), mark_maximum=True)

        if self.show_support_points:
            support_points = np.array(self.gpCAMstream['position'])
        else:
            support_points = None

        self.plot_arr(self.prediction_gpcam,
                      filename=path.join(path1, 'prediction_gpcam'), mark_maximum=True,
                      support_points=support_points)

    def gridsearch_iterate_over_all_indices(self, refinement=False):
        bWorkedOnIndex = False
        # the complicated iteration syntax is due the unknown dimensionality of the results space / arrays
        it = np.nditer(self.results, flags=['multi_index'])
        while not it.finished:
            itindex = it.multi_index
            # parameter grids can be symmetric, and only one of the symmetry-related indices
            # will be calculated unless calc_symmetric is True
            if all(itindex[i] <= itindex[i + 1] for i in range(len(itindex) - 1)) or self.calc_symmetric:
                # run iteration if it is first time or the value in results is nan
                invalid_result = np.isnan(self.results[itindex])
                insufficient_iterations = self.n_iter[itindex] < self.miniter
                # Do we need to work on this particular index?
                if invalid_result or insufficient_iterations:
                    bWorkedOnIndex = True
                    _ = self.work_on_iteration(it=it)
            it.iternext()
        return bWorkedOnIndex

    def plot_arr(self, arr_value, arr_variance=None, filename='plot', mark_maximum=False, valmin=None, valmax=None,
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

    def run_optimization_gpcam(self):
        # Using the gpCAM global optimizer, follows the example from the gpCAM website

        # initialization
        # feel free to try different acquisition functions, e.g. optional_acq_func, "covariance", "shannon_ig"
        # note how costs are defined in for the autonomous experimenter
        parlimits = self.exp_par[['l_sim', 'u_sim']].to_numpy()
        numpars = len(parlimits)

        # those are Pandas dataframe colunns
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
                                              instrument_function=self.gpcam_instrument,
                                              acquisition_function=self.acq_func,  # optional_acq_func,
                                              # cost_func = optional_cost_function,
                                              # cost_update_func = optional_cost_update_function,
                                              x_data=x, y_data=y, noise_variances=v,
                                              # cost_func_params={"offset": 5.0, "slope": 10.0},
                                              kernel_function=None, calc_inv=True,
                                              communicate_full_dataset=False, ram_economy=True)

        # save and evaluate initial data set if it has been freshly calculate
        if bFirstEval:
            self.save_results_gpcam()
            self.gpcam_prediction(self.my_ae)

        while len(self.my_ae.x_data) < self.gpcam_iterations:
            print("length of the dataset: ", len(self.my_ae.x_data))
            self.my_ae.train(method="global", max_iter=10000)  # or not, or both, choose "global","local" and "hgdl"
            # update hyperparameters in case they are optimized asynchronously
            self.my_ae.train(method="local")  # or not, or both, choose between "global","local" and "hgdl"
            # training and client can be killed if desired and in case they are optimized asynchronously
            # self.my_ae.kill_training()
            if self.gpcam_step is not None:
                target_iterations = len(self.my_ae.x_data) + self.gpcam_step
                retrain_async_at = []
            else:
                target_iterations = self.gpcam_iterations
                retrain_async_at = np.logspace(start=np.log10(len(self.my_ae.x_data)),
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
            self.save_results_gpcam()
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

            if self.bClusterMode:
                # never repeat iterations on cluster or when just calculating entropies
                break

        # wait for all jobs to finish
        if self.bClusterMode:
            while self.joblist:
                self.waitforjob(bFinish=True)

    def save_results_gpcam(self):
        with open(path.join(self.spath, 'results', 'gpCAMstream.pkl'), 'wb') as file:
            pickle.dump(self.gpCAMstream, file)

    def work_on_iteration(self, it=None, position=None, gpiteration=None):

        if it is not None:
            # grid mode, calculate which iteration we are on from itindex
            # Recover itindex from 'it'. This is not an argument to the function anymore, as work_on_index might
            # be used independently of iterate_over_all_indices
            itindex = it.multi_index
            if self.calc_symmetric:
                iteration = it.iterindex
            else:
                # TODO: This is potentially expensive. Find better method for symmetry-conscious calculation
                it2 = np.nditer(self.results, flags=['multi_index'])
                iteration = 0
                while not it2.finished:
                    itindex2 = it2.multi_index
                    if all(itindex2[i] <= itindex2[i + 1] for i in range(len(itindex2) - 1)):
                        # iterations are only increased if this index is not dropped because of symmetry
                        iteration += 1
                    it2.iternext()
        else:
            # gpcam mode
            iteration = gpiteration
            itindex = None

        # most relevant result for a particular index to return for general use of this function
        result = 0



        configurations = self.set_sim_pars_for_iteration(it, position)
        avg_gmm_marginal = self.calc_entropy_for_iteration(molstat_iter, itindex=itindex)



        return avg_gmm_marginal

