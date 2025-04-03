import copy
import datetime
import numpy as np
import threading
import time
from pprint import pprint
from uuid import uuid4

from .manager import MANAGER_ADDRESS, ManagerInterface
from ..gp import Gp

from lh_manager.liquid_handler.bedlayout import Composition
from lh_manager.liquid_handler.methods import BaseMethod
from lh_manager.liquid_handler.roadmapmethods import (ROADMAP_QCMD_MakeBilayer,
                                                      ROADMAP_QCMD_RinseLoopInjectandMeasure,
                                                      ROADMAP_QCMD_RinseDirectInjectandMeasure,
                                                      ROADMAP_QCMD_DirectInjectandMeasure,
                                                      ROADMAP_DirectInjecttoQCMD,
                                                      QCMDRecordTag,
                                                      Formulation,
                                                      TransferWithRinse,
                                                      MixWithRinse,
                                                      InferredWellLocation,
                                                      ROADMAP_DirectInjecttoQCMD)
                                                      
from lh_manager.liquid_handler.samplelist import Sample, MethodList

def extract_ids(sample: Sample, method_id: str):
    
    method: BaseMethod = next((m for m in sample.stages['methods'].active if m.id == method_id), None)
    if method is None:
        print(f'Warning: no active tasks found with sample {sample.id} and method {method_id}')
        return None, None
    
    task = method.tasks[-1]
    subtask = task.task['tasks'][-1]

    return task.id, subtask.get('id', None)

def execute_measurement(manager: ManagerInterface, sample: Sample, measure_method_id: str) -> dict:

    task_id, subtask_id = extract_ids(sample, measure_method_id)
    print(f'\tSubtask id: {subtask_id}')
    # wait until measurement is complete and then read the result
    thread_result = {'result': None}
    monitor_thread = threading.Thread(target=manager.monitor_task, args=(task_id, thread_result), daemon=True)
    monitor_thread.start()
    monitor_thread.join()
    if thread_result['result'].get('success', None) is not None:
        return manager.get_task_result(task_id, subtask_id)
    else:
        pprint(thread_result)

def collect_data(manager: ManagerInterface, lipids: dict[str, float], concentration: float, sample_name: str, description: str, control: bool = True) -> tuple[float, float]:
    """Performs a bilayer formation measurement with a lipid composition and total lipid concentration.
    
        TODO: for now, assumes a single channel = 0, but in the future could implement a channel selector
                based on the inputs (substrate) or a rolling counter

    Args:
        lipids (dict[str, float]): dictionary with lipid name as the key and fraction of that lipid as the value
        concentration (float): total lipid concentration in mg/mL
        sample_name (str): name of sample
        description (str): description of data set
        control (bool, optional): if True, collect a control measurement. Default True.

    Returns:
        tuple[float, float]: normalized change in QCMD frequency and error
    """

    sum_fractions = sum(lipids.values())
    units = 'mg/mL'
    water = Composition(solvents=[manager.solvent_from_material('H2O', fraction=1)], solutes=[])
    bilayer_composition = Composition(solvents=[manager.solvent_from_material('isopropanol', fraction=1)],
                                      solutes=[manager.solute_from_material(name, frac * concentration / sum_fractions, units) for (name, frac) in lipids.items() if frac > 0])

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
                                                      Equilibration_Time=0,
                                                      Measurement_Time=0)

    isopropanol_rinse = ROADMAP_QCMD_RinseLoopInjectandMeasure(Target_Composition=Composition(solvents=[manager.solvent_from_material('isopropanol', fraction=1)], solutes=[]),
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=True,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=0,
                                                      Measurement_Time=0)
    
    water_rinse = ROADMAP_QCMD_RinseLoopInjectandMeasure(Target_Composition=Composition(solvents=[manager.solvent_from_material('H2O', fraction=1)], solutes=[]),
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=0,
                                                      Measurement_Time=0)
    """
    buffer_control = ROADMAP_QCMD_RinseLoopInjectandMeasure(id=str(uuid4()),
                                                            Target_Composition=buffer_composition,
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=0.1,
                                                      Measurement_Time=0.5)
    """
    buffer_control = ROADMAP_QCMD_RinseDirectInjectandMeasure(id=str(uuid4()),
                                                            Target_Composition=water,
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=0.1,
                                                      Measurement_Time=0.5)

                                                      
    make_bilayer = ROADMAP_QCMD_RinseDirectInjectandMeasure(id=str(uuid4()),
                                                            Target_Composition=water,
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=0.1,
                                                      Measurement_Time=0.5)
    
    second_water_rinse = ROADMAP_QCMD_RinseLoopInjectandMeasure(
                                                      Target_Composition=water,
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=0,
                                                      Measurement_Time=0)
    """
    make_bilayer = ROADMAP_QCMD_MakeBilayer(id=str(uuid4()),
                                            Bilayer_Composition=bilayer_composition,
                                            Bilayer_Solvent=Composition(solvents=[manager.solvent_from_material('isopropanol', fraction=1)]),
                                            Use_Rinse_System_for_Solvent=True,
                                            Lipid_Injection_Volume=1.0,
                                            Buffer_Composition=buffer_composition,
                                            Buffer_Injection_Volume=2.0,
                                            Use_Rinse_System_for_Buffer=True,
                                            Extra_Volume=0.1,
                                            Rinse_Volume=2.0,
                                            Flow_Rate=3.0,
                                            Exchange_Flow_Rate=0.1,
                                            Equilibration_Time=3.0,
                                            Measurement_Time=5.0)
    """
    # Initiate sample
    sample = Sample(name=sample_name,
        description=repr(bilayer_composition) + ', started ' + description,
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
            #water_rinse,
            buffer_control
        ]
    methods += [
        #second_water_rinse,
        make_bilayer
    ]

    for method in methods:
        sample.stages['methods'].add(method)
    _, sample = manager.update_sample(sample)
    _, sample = manager.run_sample(sample.id)

    # Rinses and control measurement, if control is None, otherwise use provided control
    #sample.stages['methods'].add(ethanol_rinse)
    #sample.stages['methods'].add(isopropanol_rinse)

    if control:
        print('Waiting for control measurement result...')
        control_result = execute_measurement(manager, sample, buffer_control.id)
    else:
        control_result = None

    print('Waiting for measurement result...')

    # make the bilayer
    measure_result = execute_measurement(manager, sample, make_bilayer.id)

    # parse and combine the results
    return {'control': control_result, 'measure': measure_result}

def reduce_qcmd(meas: dict, control: dict) -> float:
    """Data reduction for measurement and control. Performs the following steps:
        o Calculates the frequency difference for each harmonic
        o Normalizes each frequency difference to the harmonic number
        o Selects all harmonics except the fundamental
        o Calculates the average and the variance using a mixture distribution

    Args:
        meas (dict): measurement result dictionary
        control (dict): control result dictionary

    Returns:
        tuple(float, float): normalized frequency difference, variance
    """

    try:
        mtag = meas['result']['result']['tags'][0]
        ctag = control['result']['result']['tags'][0]
    except (KeyError, ValueError):
        print(f'Warning: measurement result does not have correct format: {meas}')
    harmonics = [1, 3, 5, 7, 9]

    diffs = []
    diff_errs = []
    for imf, icf, h in zip(mtag['f_averages'], ctag['f_averages'], harmonics):
        diffs.append((imf[0] - icf[0]) / h)
        diff_errs.append(np.sqrt(imf[2] ** 2 + icf[2] ** 2) / h)

    diffs = np.array(diffs)[1:]
    diff_errs = np.array(diff_errs)[1:]
    print(diffs, diff_errs)

    average = np.average(diffs)
    mix_variance = np.average(diff_errs ** 2 + diffs ** 2) - average ** 2

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
                                                      Measurement_Time=5)
    
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
                                      record_time=5 * 60,
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
        control_result = execute_measurement(manager, sample, water_control.id)
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
    measure_result = execute_measurement(manager, sample, water_ipa_measure.id)

    # parse and combine the results
    return {'control': control_result, 'measure': measure_result}

class ROADMAP_Gp(Gp):

    def __init__(self, exp_par, storage_path=None, acq_func="variance", gpcam_iterations=50, gpcam_init_dataset_size=20, gpcam_step=1, keep_plots=False, miniter=1, optimizer='gpcam', parallel_measurements=1, resume=False, show_support_points=True, project_name=''):
        super().__init__(exp_par, storage_path, acq_func, gpcam_iterations, gpcam_init_dataset_size, gpcam_step, keep_plots, miniter, optimizer, parallel_measurements, resume, show_support_points, project_name)

        # sort compounds into optimized and non-optimized
        optimized_lipids = []
        non_optimized_lipids = {}
        for _, par in self.all_par.iterrows():
            if par['type'] == 'compound':
                if par['optimize']:
                    optimized_lipids.append(par['name'])
                else:
                    non_optimized_lipids[par['name']] = par['value']

        self.optimized_lipids = optimized_lipids
        self.non_optimized_lipids = non_optimized_lipids

        # get default value of concentration
        if 'lipid concentration' in self.all_par.columns:
            concentration_index = self.all_par[self.all_par['name'] == 'lipid concentration'].index.values[0]
            self.concentration = self.all_par.iloc[concentration_index]['value']
        else:
            self.concentration = None

        # control measurement
        self.control: dict | None = None
        self.control_counter: int = 0
        
        # how often to collect control measurements
        self.control_cycle = 3

        # connect to manager
        self.manager = ManagerInterface(address=MANAGER_ADDRESS)
        self.manager.initialize()

    def do_measurement(self, optpars: dict, it_label: str):

        # Configure a particular problem with a set of N lipids. Then there are N-1 keywords describing the composition, plus 1 for the total concentration.

        # break out all non-lipid parameters
        conc = optpars.pop('lipid concentration', self.concentration)

        # cycle through optimized lipids and determine absolute fraction. Each lipid is expressed as a fraction of the remainder
        lipid_dict = {}
        sum_remainder = 0.0
        for lipid in self.optimized_lipids:
            new_fraction = optpars[lipid] * (1 - sum_remainder)
            lipid_dict[lipid] = float(new_fraction)
            sum_remainder += new_fraction

        # fill in the remainder with the non-optimized lipids in the ratio given in the table values
        sum_non_optimized = sum(v for v in self.non_optimized_lipids.values())
        for lipid, v in self.non_optimized_lipids.items():
            lipid_dict[lipid] = float(v * (1 - sum_remainder) / sum_non_optimized)

        print(f'Starting measurement {it_label} with concentration {conc} and lipids {lipid_dict}: ')

        # collect data, including control if necessary, and increment control counter
        new_control = (((self.control_counter % self.control_cycle) == 0) | (self.control is None))
        
        sample_name = f'{self.project_name} point {it_label}'
        description = datetime.datetime.now().strftime("%Y%m%d %H.%M.%S")
        if 'ipa_fraction' in optpars:
           res = collect_wateripa(self.manager, optpars['ipa_fraction'], sample_name, description, control=new_control)
        else:
            res = collect_data(self.manager, lipid_dict, conc, sample_name, description, control=new_control)
        self.control_counter += 1

        # replace current value of control if applicable
        if new_control:
            self.control = res['control']

        # reduce data with most recent control
        results, variance = reduce_qcmd(res['measure'], self.control)

        return results, variance

if __name__ == '__main__':

    import json

    #pprint.pprint(manager.materials)
    manager = ManagerInterface(address=MANAGER_ADDRESS)
    manager.initialize()

    if True:
        #res = collect_data(manager, {'DOPC': 0.1, 'DOPE': 0.2}, 0.5, 'hello', 'goodbye')
        res = collect_wateripa(manager, 0.2, 'hello', 'goodbye', False)
        #with open('control.json', 'w') as f:
        #    json.dump(res, f, indent=2)
        pprint(res)

    else:

        with open('control.json', 'r') as f:
            res = json.load(f)

        result, variance = reduce_qcmd(res['measure'], res['control'])
        print(result, np.sqrt(variance))

        



