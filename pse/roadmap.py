import copy
import datetime
import json
import numpy as np
import time

from pathlib import Path
from pprint import pprint
from scipy.special import erf
from threading import Event
from uuid import uuid4

from pse.manager import MANAGER_ADDRESS, ManagerInterface
from pse.gp import Gp
from gpcam import GPOptimizer

from lh_manager.liquid_handler.bedlayout import Composition
from lh_manager.liquid_handler.formulation import SoluteFormulation
from lh_manager.liquid_handler.roadmapmethods import (ROADMAP_QCMD_MakeBilayer,
                                                      ROADMAP_QCMD_RinseLoopInjectandMeasure,
                                                      ROADMAP_QCMD_RinseDirectInjectandMeasure,
                                                      ROADMAP_DirectInjecttoQCMD,
                                                      ROADMAP_QCMD_DirectInjectandMeasure,
                                                      QCMDRecordTag,
                                                      Formulation,
                                                      TransferWithRinse,
                                                      MixWithRinse,
                                                      InferredWellLocation,
                                                      ROADMAP_DirectInjecttoQCMD)
                                                      
from lh_manager.liquid_handler.samplelist import Sample, MethodList

def acq_variance_target(x: np.ndarray, gpoptimizer: GPOptimizer):

    #print(x, x.shape)
    #print(gpoptimizer.posterior_covariance(x, variance_only=True)['v(x)'], gpoptimizer.posterior_mean(x)['f(x)'])
    tolerance = 5
    retval = np.array(gpoptimizer.posterior_covariance(x, variance_only=True)['v(x)']) / ((np.array(gpoptimizer.posterior_mean(x)['f(x)']) + 25) ** 2 + tolerance ** 2)
    #print(retval)
    return retval

def acq_variance_target_add(x: np.ndarray, gpoptimizer: GPOptimizer):

    #print(x, x.shape)
    #print(gpoptimizer.posterior_covariance(x, variance_only=True)['v(x)'], gpoptimizer.posterior_mean(x)['f(x)'])
    retval = 3 * np.array(gpoptimizer.posterior_covariance(x, variance_only=True)['v(x)']) + 1.0 / (np.array(gpoptimizer.posterior_mean(x)['f(x)']) + 25) ** 2
    #print(retval)
    return retval

def collect_data_sleep(manager: ManagerInterface,
                 bilayer_composition: Composition,
                 sample_name: str,
                 description: str,
                 channel: int = 0,
                 control: bool = True,
                 stop_event: Event = Event()) -> tuple[float, float]:
    """Performs a bilayer formation measurement with a lipid composition and total lipid concentration.
    
        TODO: for now, assumes a single channel = 0, but in the future could implement a channel selector
                based on the inputs (substrate) or a rolling counter

    Args:
        lipids (dict[str, float]): dictionary with lipid name as the key and concentration of that lipid as the value
        concentration (float): total lipid concentration in mg/mL
        sample_name (str): name of sample
        description (str): description of data set
        control (bool, optional): if True, collect a control measurement. Default True.

    Returns:
        tuple[float, float]: normalized change in QCMD frequency and error
    """

    # Initiate sample
    sample = Sample(name=sample_name,
        description=repr(bilayer_composition) + ', started ' + description,
        channel=channel,
        stages={'methods': MethodList()})

    sample = manager.new_sample(sample=sample)        

    time.sleep(30)

    # parse and combine the results
    return {'control': {}, 'measure': {}}

def collect_data(manager: ManagerInterface,
                 bilayer_composition: Composition,
                 exchange_flow_rate: float,
                 sample_name: str,
                 description: str,
                 channel: int = 0,
                 control: bool = True,
                 stop_event: Event = Event()) -> tuple[float, float]:
    """Performs a bilayer formation measurement with a lipid composition and total lipid concentration.
    
    Args:
        lipids (dict[str, float]): dictionary with lipid name as the key and concentration of that lipid as the value
        concentration (float): total lipid concentration in mg/mL
        sample_name (str): name of sample
        description (str): description of data set
        control (bool, optional): if True, collect a control measurement. Default True.

    Returns:
        tuple[float, float]: normalized change in QCMD frequency and error
    """

    water = Composition(solvents=[manager.solvent_from_material('H2O', fraction=1)], solutes=[])
    isopropanol = Composition(solvents=[manager.solvent_from_material('isopropanol', fraction=1)], solutes=[])

    buffer_composition = Composition(solvents=[manager.solvent_from_material('H2O', fraction=1)],
                                    solutes=[manager.solute_from_material('NaCl', concentration=0.15, units='M'),
                                            manager.solute_from_material('tris', concentration=10.0, units='mM')])

    # rinses
    ethanol_rinse = ROADMAP_QCMD_RinseLoopInjectandMeasure(Target_Composition=Composition(solvents=[manager.solvent_from_material('ethanol', fraction=1)], solutes=[]),
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=True,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=2,
                                                      Measurement_Time=1)

    isopropanol_rinse = ROADMAP_QCMD_RinseLoopInjectandMeasure(Target_Composition=isopropanol,
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=True,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=2,
                                                      Measurement_Time=1)
    
    water_rinse = ROADMAP_QCMD_RinseLoopInjectandMeasure(Target_Composition=water,
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=2,
                                                      Measurement_Time=1)
    
    buffer_control = ROADMAP_QCMD_RinseLoopInjectandMeasure(id=str(uuid4()),
                                                            Target_Composition=buffer_composition,
                                                      Volume=1.5,
                                                      Injection_Flow_Rate=0.1,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=5,
                                                      Measurement_Time=3)
                                                          
   
    second_water_rinse = ROADMAP_QCMD_RinseLoopInjectandMeasure(
                                                      Target_Composition=water,
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=2,
                                                      Measurement_Time=1)

    second_ethanol_rinse = ROADMAP_QCMD_RinseLoopInjectandMeasure(Target_Composition=Composition(solvents=[manager.solvent_from_material('ethanol', fraction=1)], solutes=[]),
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=True,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=2,
                                                      Measurement_Time=1)


    make_bilayer = ROADMAP_QCMD_MakeBilayer(id=str(uuid4()),
                                            Bilayer_Composition=bilayer_composition,
                                            Bilayer_Solvent=isopropanol,
                                            Use_Rinse_System_for_Solvent=True,
                                            Lipid_Injection_Volume=1.0,
                                            Buffer_Composition=buffer_composition,
                                            Buffer_Injection_Volume=1.5,
                                            Use_Rinse_System_for_Buffer=True,
                                            Extra_Volume=0.1,
                                            Rinse_Volume=2.0,
                                            Flow_Rate=2.0,
                                            Exchange_Flow_Rate=max(exchange_flow_rate, 0.1),
                                            Equilibration_Time=5.0,
                                            Measurement_Time=3.0)
    
    # set up running protocol. Note that measurement methods must have an ID
    methods = [
        water_rinse,
        ethanol_rinse,
        ]
    if control:
        methods += [
            isopropanol_rinse,
            buffer_control,
            second_water_rinse,
            second_ethanol_rinse,
        ]
    methods += [
        make_bilayer
    ]

    # Initiate sample
    sample = Sample(name=sample_name,
        description=repr(bilayer_composition) + ', started ' + description,
        channel=channel,
        stages={'methods': MethodList()})

    sample = manager.new_sample(sample=sample)        

    for method in methods:
        sample.stages['methods'].add(method)
    _, sample = manager.update_sample(sample)
    _, sample = manager.run_sample(sample.id)

    # Rinses and control measurement, if control is None, otherwise use provided control
    #sample.stages['methods'].add(ethanol_rinse)
    #sample.stages['methods'].add(isopropanol_rinse)

    if control:
        print('Waiting for control measurement result...')
        control_result = manager.wait_for_result(sample, buffer_control.id, stop_event)
    else:
        control_result = None

    print('Waiting for measurement result...')

    # make the bilayer
    measure_result = manager.wait_for_result(sample, make_bilayer.id, stop_event)

    # archive the sample to clean up the GUI
    #manager.archive_sample(sample)

    # parse and combine the results
    return {'control': control_result, 'measure': measure_result}

def reduce_qcmd(meas: dict, control: dict, harmonic_power: float = 1) -> float:
    """Data reduction for measurement and control. Performs the following steps:
        o Calculates the frequency difference for each harmonic
        o Normalizes each frequency difference to the harmonic number
        o Selects all harmonics except the fundamental
        o Calculates the average and the variance using a mixture distribution

    Args:
        meas (dict): measurement result dictionary
        control (dict): control result dictionary
        harmonic_power (float): exponent to normalize frequencies (1 for most applications, 1/2 if looking at solvent properties)

    Returns:
        tuple(float, float): normalized frequency difference, variance
    """

    try:
        mtag = meas['result']['result']['tags'][0]
        ctag = control['result']['result']['tags'][0]
    except (KeyError, ValueError, TypeError):
        print(f'Warning: measurement result does not have correct format: {meas}')
        return None, None
    harmonics = [1, 3, 5, 7, 9]

    diffs = []
    diff_errs = []
    for imf, icf, h in zip(mtag['f_averages'], ctag['f_averages'], harmonics):
        diffs.append((imf[0] - icf[0]) / (h ** harmonic_power))
        diff_errs.append(np.sqrt(imf[2] ** 2 + icf[2] ** 2) / (h ** harmonic_power))

    diffs = np.array(diffs)[1:]
    diff_errs = np.array(diff_errs)[1:]
    weights = 1. / diff_errs ** 2
    print(diffs, diff_errs)

    # weights are the individual variances
    average = np.average(diffs, weights=weights)
    mix_variance = np.average(diff_errs ** 2 + diffs ** 2, weights=weights) - average ** 2
    
    # apply Bessel correction https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    # does this apply to a mixture distribution?
    #corrected_variance = mix_variance / (1 - sum(weights ** 2) / sum(weights) ** 2)

    return average, mix_variance

def collect_wateripa(manager: ManagerInterface, ipa_fraction: float, sample_name: str, description: str, control: bool = True) -> tuple[float, float]:
    """Performs a bilayer formation measurement with a lipid composition and total lipid concentration.
    
        TODO: for now, assumes a single channel = 0, but in the future could implement a channel selector
                based on the inputs (substrate) or a rolling counter

    Args:
        manager (ManagerInterface): lh_manager interface to use
        ipa_fraciton (float): total ipa fraction
        sample_name (str): name of sample
        description (str): description of data set
        control (bool, optional): if True, collect a control measurement. Default True.

    Returns:
        tuple[float, float]: normalized change in QCMD frequency and error
    """

    water = Composition(solvents=[manager.solvent_from_material('H2O', fraction=1)], solutes=[])
    solvents = []
    if ipa_fraction > 0:
        solvents.append(manager.solvent_from_material('isopropanol', fraction=ipa_fraction))
    if ipa_fraction < 1:
        solvents.append(manager.solvent_from_material('H2O', fraction=(1 - ipa_fraction)))
    mix = Composition(solvents=solvents, solutes=[])
    
    water_control = ROADMAP_QCMD_RinseDirectInjectandMeasure(id=str(uuid4()),
                                                            Target_Composition=water,
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=3,
                                                      Measurement_Time=3)
    
    mix_well = InferredWellLocation(rack_id='Mix', expected_composition=mix)
    water_ipa_mix = Formulation(id=str(uuid4()),
                                target_composition=mix,
                                target_volume=2.0,
                                Target=mix_well,
                                transfer_template=TransferWithRinse(Use_Liquid_Level_Detection=False),
                                mix_template=MixWithRinse(Repeats=3,
                                                          Extra_Volume=0.1,
                                                          Use_Liquid_Level_Detection=False))

    water_ipa_inject = ROADMAP_DirectInjecttoQCMD(Source=mix_well,
                        Volume=1.0,
                        Aspirate_Flow_Rate=2.0,
                        Load_Flow_Rate=2.0,
                        Injection_Flow_Rate=1.0,
                        Outside_Rinse_Volume=0.5,
                        Extra_Volume=0.1,
                        Air_Gap=0.15,
                        Use_Liquid_Level_Detection=False,
                        Use_Bubble_Sensors=True,
                        )

    water_ipa_measure = QCMDRecordTag(id=str(uuid4()),
                                      record_time=3 * 60,
                                sleep_time=3 * 60,
                                tag_name=repr(mix))

    # Initiate sample
    sample = Sample(name=sample_name,
        description=repr(mix) + ', started ' + description,
        channel=0,
        stages={'methods': MethodList()})

    sample = manager.new_sample(sample=sample)    

    # set up running protocol. Note that measurement methods must have an ID
    methods = [
        #ethanol_rinse,
        #isopropanol_rinse
        ]
    if control:
        methods += [
            water_control
        ]
    methods += [
        water_ipa_mix,
        water_ipa_inject,
        water_ipa_measure
    ]

    for method in methods:
        sample.stages['methods'].add(method)
    _, sample = manager.update_sample(sample)
    _, sample = manager.run_sample(sample.id)
    time.sleep(1)

    # Rinses and control measurement, if control is None, otherwise use provided control
    #sample.stages['methods'].add(ethanol_rinse)
    #sample.stages['methods'].add(isopropanol_rinse)

    if control:
        print('Waiting for control measurement result...')
        control_result = manager.wait_for_result(sample, water_control.id)
    else:
        control_result = None

    print('Waiting for measurement result...')

    #  Normally we don't have to wait to do the injection but in this case we do
    #manager.get_samples()
    #sample = manager.rehydrate_sample(sample.id)
    #sample.stages['methods'].add(water_ipa_inject)
    #sample.stages['methods'].add(water_ipa_measure)
    #_, sample = manager.update_sample(sample)
    #_, sample = manager.run_sample(sample.id)
    measure_result = manager.wait_for_result(sample, water_ipa_measure.id)

    # archive sample
    manager.archive_sample(sample)

    # parse and combine the results
    return {'control': control_result, 'measure': measure_result}

class ROADMAP_Gp(Gp):

    def __init__(self, exp_par, storage_path=None, acq_func="variance", gpcam_iterations=50, gpcam_init_dataset_size=4, gpcam_step=1, keep_plots=False, miniter=1, optimizer='gpcam', parallel_measurements=1, resume=False, signal_estimate=10, show_support_points=True, train_global_every=None, gp_discrete_points=None, project_name=None):

        # ensure that init dataset size is no smaller than number of parallel measurements
        gpcam_init_dataset_size = max(gpcam_init_dataset_size, parallel_measurements)

        # set number of channels
        self.n_channels = parallel_measurements
        self.channels = [{'busy': False, 'count': 0} for i in range(self.n_channels)]

        super().__init__(exp_par, storage_path, acq_func, gpcam_iterations, gpcam_init_dataset_size, gpcam_step, keep_plots, miniter, optimizer, parallel_measurements, resume, signal_estimate, show_support_points, train_global_every, gp_discrete_points, project_name)

        """
            Parameters are fractions of the *volume* of the stock solutions of the optimized lipids. In other words, a parameter set of {'DOPC': 0.1, 'DOPE': 0.2}
                would create a formulation with a standard total volume V, of which 0.1V is of the DOPC stock solution, 0.2V is of the DOPE stock solution, and 0.7V is
                the diluent. This allows the parameters to vary from 0 to 1 with no issues.

            Plotting this in a reasonable concentration/composition space has to be done later. The stock solution concentrations are looked up in the layout.
            TODO: stock solutions need to be stored.
        """

        # set acquisition function
        self.acq_func = acq_variance_target

        # control measurement
        self.controls: dict[str, dict] = self.load_controls()
        self.current_control: list[str] = [None for _ in range(self.n_channels)]
        for channel in range(self.n_channels):
            if len(self.controls[str(channel)]):
                # take most recent control (should always be one)
                self.current_control[channel] = list(self.controls[str(channel)].keys())[-1]
            else:
                self.current_control[channel] = None
        
        # how often to collect control measurements
        self.control_cycle = 4

        # connect to manager
        self.manager = ManagerInterface(address=MANAGER_ADDRESS)
        self.manager.initialize()

        # create a dictionary of lipid names and stock solution concentrations
        optimized_lipids = list(self.all_par[self.all_par['type']=='compound']['name'])
        #pprint(optimized_lipids)
        self._layout = self.manager.get_layout()
        self.units = 'mg/mL'

        # protect against missing stock solutions
        self.lipids = {}
        for name in optimized_lipids:
            stock_conc = self._find_stock(name)
            if stock_conc is None:
                #print(f'WARNING: cannot find stock solution of {name}. Ignoring...')
                raise RuntimeError('cannot find stock solution of {name}. Ignoring...')
            else:
                self.lipids.update({name: stock_conc})

        self._filter_discrete_points()

    def _filter_discrete_points(self):

        init_time = time.time()
        pts = np.array(self.gp_discrete_points, dtype=float)
        print(pts.shape)

        lipid_idxs = []
        for lipid, stock in self.lipids.items():
            lipid_idxs.append(self.all_par.index[self.all_par['name'] == lipid].tolist()[0])

        print(lipid_idxs)

        source_matrix = np.diag(list(self.lipids.values()))
        print(source_matrix, source_matrix.shape)

        target_vectors = np.take_along_axis(pts, np.array([lipid_idxs]), axis=1)
        print(target_vectors, target_vectors.shape)

        sol = np.linalg.solve(source_matrix, target_vectors.T).T
        print(sol, sol.shape)
        print(f'Elapsed time: {time.time() - init_time}')

        # efficient filtering of results
        # sum of elements has to be less than 1
        mask_sum = np.sum(sol, axis=1) <= 1.0

        # no elements can be less than zero
        mask_zero = ~np.sum(sol < 0.0, axis=1, dtype=bool)

        filtered_sol = sol[mask_zero & mask_sum]
        print(f'Elapsed time: {time.time() - init_time}')
        print(filtered_sol.shape)

        pts = pts[mask_zero & mask_sum]

        self.gp_discrete_points = [pts[i] for i in range(pts.shape[0])]

        if len(self.gp_discrete_points) > 40000:
            raise RuntimeError(f'Maximum length of filtered points is 40000, requested {len(self.gp_discrete_points)}')

    def _update_layout(self):

        self._layout = self.manager.get_layout()

    def _find_stock(self, compound: str) -> float:

        # find wells that have only the compound as a stock solution
        well = next((w for w in self._layout.racks['Stock'].wells if [s.name for s in w.composition.solutes]==[compound]), None)

        if well is not None:
            return well.composition.solutes[0].convert_units(self.units)
        else:
            return None

    def load_controls(self) -> dict:

        controls = {str(i): {} for i in range(self.n_channels)}
        storage_path = Path(self.spath) / 'results' / 'results.json'
        if storage_path.exists():
            with open(storage_path, 'r') as f:
                current_results: dict = json.load(f)

            controls = current_results.get('controls', controls)

        return controls

    def save_result(self, it_label: str, parameters: dict, raw_result: dict, control_id: str, reduced_result: dict):

        storage_path = Path(self.spath) / 'results' / 'results.json'
        if storage_path.exists():

            with open(storage_path, 'r') as f:
                current_results = json.load(f)
        else:
            current_results = {}

        current_results.update({'controls': self.controls,
                                str(it_label): dict(parameters=parameters,
                                                    control=control_id,
                                                   raw_result=raw_result,
                                                   reduced_result=reduced_result)})
        with open(storage_path, 'w') as f:
            json.dump(current_results, f)

    def do_measurement_test(self, optpars, it_label, entry, q):
        
        print(f'Starting measurement {it_label}')
        lipid_dict = {}
        #frac_remaining = 1.0
        for compound, stock_conc in self.lipids.items():
            frac = optpars.get(compound, 0.0)
            lipid_dict[compound] = frac * stock_conc
            #frac_remaining *= (1.0 - frac)

        total_concentration = sum(f for f in lipid_dict.values())

        result = -25 * (0.5 * (1 + erf((lipid_dict.get('DOPC', 0.0) - 0.5) / np.sqrt(2 * 0.6 ** 2))) + \
                        lipid_dict.get('DOPE', 0.0) + \
                        0.25 * (1 + erf((lipid_dict.get('POPG', 0.0) - 0.5) / np.sqrt(2 * 0.3 ** 2))))
        #variance = np.abs(result)
        variance = 10.0
        result += np.sqrt(variance) * np.random.randn()

        sleep_time = 60 * np.random.random()
        print(f'{it_label} Sleeping {sleep_time:0.0f} s...')

        time.sleep(sleep_time)

        print(f'{it_label} Waking...')

        # THESE THREE LINES NEED DO BE PRESENT IN EVERY DERIVED METHOD
        # TODO: Make this post-logic seemless and still working with multiprocessing.Process
        entry['value'] = result
        entry['variance'] = variance
        q.put(entry)        

        return result, variance            

    def gpcam_instrument(self, data):
        if self.gpiteration == 0:
            data[0]['x_data'] = np.zeros(len(self.lipids))

        return super().gpcam_instrument(data)

    def do_measurement_old(self, optpars: dict, it_label: str, entry: dict, q):

        # determine channel number
        channel = next(i for i, ch in enumerate(self.channels) if not ch['busy'])
        self.channels[channel]['busy'] = True
        #channel = int(it_label) % self.n_channels

        # Configure a particular problem with a set of N lipids. Then there are N-1 keywords describing the composition, plus 1 for the total concentration.
        # Calculate the composition we expect from optpars.
        lipid_dict = {compound: optpars.get(compound, 0.0) for compound in self.lipids.keys()}

        total_concentration = sum(f for f in lipid_dict.values())

        print(f'Starting measurement {it_label} with total concentration {total_concentration} and lipid concentrations {lipid_dict}: ')

        if False:
            result = -25 * (0.5 * (1 + erf((total_concentration - 0.6) / np.sqrt(2 * 0.3 ** 2))) + 0.5 * (1 + erf((lipid_dict['DOPE'] - 1.5) / np.sqrt(2 * 0.3 ** 2))))
            variance = np.sqrt(np.abs(result))
            time.sleep(5)
            return result, variance

        # check that the composition can be made with the current bed layout
        self._update_layout()
        diluent = Composition(solvents=[self.manager.solvent_from_material('isopropanol', fraction=1)], solutes=[])
        bilayer_composition = Composition(solvents=[],
                                    solutes=[self.manager.solute_from_material(name, conc, self.units) for (name, conc) in lipid_dict.items() if conc > 0])
        
        # in special case that there are no lipids, specify that we're just measuring the diluent
        if bilayer_composition.is_empty:
            bilayer_composition = diluent
            real_composition = diluent
        else:
        
            # attempt the formulation. If it fails, stop. Most likely is that one of the stocks on the bed ran out.
            bilayer_formulation = SoluteFormulation(target_composition=bilayer_composition, target_volume=1.0, diluent=diluent)
            _, _, success = bilayer_formulation.formulate(self._layout)

            if success:
                real_composition = bilayer_formulation.get_expected_composition(self._layout)
                print('Real composition: ' + repr(real_composition))
            else:
                raise RuntimeError('Cannot make composition ' + repr(bilayer_formulation))
        
        # collect data, including control if necessary, and increment control counter
        channel_counter = self.channels[channel]['count']
        new_control = (((channel_counter % self.control_cycle) == 0) | (len(self.controls[str(channel)])==0))
        print(f'Iteration {it_label} in channel {channel} with channel-specific counter {channel_counter}')
        
        sample_name = f'{self.project_name} point {it_label}'
        description = datetime.datetime.now().strftime("%Y%m%d %H.%M.%S")

        # start new manager client in thread (can't use self.manager because not thread-safe)
        manager = ManagerInterface(address=self.manager.address)
        manager.initialize()

        res: dict = collect_data(manager, bilayer_composition, optpars.get('flow_rate', 0.1), sample_name, description, channel=channel, control=new_control)
        harmonic_power = 1.0

        # replace current value of control if applicable
        if new_control:
            new_id = str(uuid4())
            self.current_control[channel] = new_id
            self.controls[str(channel)][new_id] = res['control']

        parameters = {'optpars': optpars,
                      'actual_composition': real_composition.model_dump(),
                      'concentration': total_concentration,
                      'lipid_concentrations': lipid_dict,
                      'lipid_fractions': {k: (v / total_concentration if total_concentration > 0 else 0) for k, v in lipid_dict.items()},
                      'harmonic_power': harmonic_power}

        # reduce data with most recent control
        results, variance = None, None
        if res['measure'] is not None:
            if len(res['measure']):
                results, variance = reduce_qcmd(res['measure'], self.controls[str(channel)][self.current_control[channel]], harmonic_power)

        #res = {'measure': None, 'control': None}
        #results, variance = 0.0, 0.01

        self.save_result(it_label, parameters, raw_result=res, control_id=self.current_control[channel], reduced_result=dict(results=results,
                                                                                variance=variance))

        self.channels[channel]['busy'] = False
        self.channels[channel]['count'] += 1

        # THESE THREE LINES NEED DO BE PRESENT IN EVERY DERIVED METHOD
        # TODO: Make this post-logic seemless and still working with multiprocessing.Process
        entry['value'] = results
        entry['variance'] = variance
        q.put(entry)        

        return results, variance

if __name__ == '__main__':

    import json

    #pprint.pprint(manager.materials)
    manager = ManagerInterface(address=MANAGER_ADDRESS)
    manager.initialize()

    if True:
        res = collect_data(manager, {'DOPC': 0.1}, 0.5, 'hello', 'goodbye', True)
        #res = collect_wateripa(manager, 0.2, 'hello', 'goodbye', False)
        #with open('control.json', 'w') as f:
        #    json.dump(res, f, indent=2)
        pprint(res)

    else:

        with open('control.json', 'r') as f:
            res = json.load(f)

        result, variance = reduce_qcmd(res['measure'], res['control'])
        print(result, np.sqrt(variance))

        



