import datetime
import numpy as np
import threading
import time
from pprint import pprint
from uuid import uuid4

from .manager import MANAGER_ADDRESS, ManagerInterface
from ..support.gp import Gp

from lh_manager.liquid_handler.bedlayout import Composition
from lh_manager.liquid_handler.methods import BaseMethod
from lh_manager.liquid_handler.roadmapmethods import ROADMAP_QCMD_MakeBilayer, ROADMAP_QCMD_RinseLoopInjectandMeasure, QCMDRecordTag
from lh_manager.liquid_handler.samplelist import Sample, MethodList

def collect_data(manager: ManagerInterface, lipids: dict[str, float], concentration: float, description: str, control: bool = True) -> tuple[float, float]:
    """Performs a bilayer formation measurement with a lipid composition and total lipid concentration.
    
        TODO: for now, assumes a single channel = 0, but in the future could implement a channel selector
                based on the inputs (substrate) or a rolling counter

    Args:
        lipids (dict[str, float]): dictionary with lipid name as the key and fraction of that lipid as the value
        concentration (float): total lipid concentration in mg/mL
        description (str): description of data set
        control (bool, optional): if True, collect a control measurement. Default True.

    Returns:
        tuple[float, float]: normalized change in QCMD frequency and error
    """

    sample = Sample(name='PhaseSpaceExplorer',
        description=datetime.datetime.now().strftime("%Y%M%D.%H:%M.%S") + ' ' + description,
        channel=0,
        stages={'methods': MethodList()})
    
    sum_fractions = sum(lipids.values())
    units = 'mg/mL'
    bilayer_composition = Composition(solvents=[manager.solvent_from_material('isopropanol', fraction=1)],
                                      solutes=[manager.solute_from_material(name, frac * concentration / sum_fractions, units) for (name, frac) in lipids.items()])

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
    
    buffer_control = ROADMAP_QCMD_RinseLoopInjectandMeasure(id=str(uuid4()),
                                                            Target_Composition=buffer_composition,
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=3,
                                                      Measurement_Time=3)

    second_water_rinse = ROADMAP_QCMD_RinseLoopInjectandMeasure(Target_Composition=Composition(solvents=[manager.solvent_from_material('H2O', fraction=1)], solutes=[]),
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
                                                      Extra_Volume=0.1,
                                                      Is_Organic=False,
                                                      Use_Bubble_Sensors=True,
                                                      Equilibration_Time=0,
                                                      Measurement_Time=0)

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

    #test_measure = QCMDRecordTag(sleep_time=10, record_time=20, tag_name='testing')

    # Rinses and control measurement, if control is None, otherwise use provided control
    sample.stages['methods'].add(ethanol_rinse)
    sample.stages['methods'].add(isopropanol_rinse)
    if control:
        sample.stages['methods'].add(water_rinse)        
        sample.stages['methods'].add(buffer_control)
        manager.new_sample(sample=sample)
        time.sleep(1)
        print('Starting first method...')
        manager.run_sample(sample.id)

        def extract_ids(sample_id: str, method_id: str):
            sample: Sample = Sample.model_validate(manager.samples[sample_id])
            method: BaseMethod = next(m for m in sample.stages['methods'].active if m.id == method_id)
            task = method.tasks[-1]
            subtask = task.task['tasks'][-1]
            pprint(subtask)

            return task.id, subtask.get('id', None)

        task_id, subtask_id = extract_ids(sample.id, buffer_control.id)
        # wait until measurement is complete and then read the result
        thread_result = {'result': None}
        monitor_thread = threading.Thread(target=manager.monitor_task, args=(task_id, thread_result), daemon=True)
        monitor_thread.start()
        monitor_thread.join()
        if thread_result['result'].get('success', None) is not None:
            control_result = manager.get_task_result(task_id, subtask_id)
        else:
            pprint(thread_result)
            control_result = None
    else:
        control_result = None

    print('Starting second method...')

    # make the bilayer
    manager.get_samples()
    sample = Sample.model_validate(manager.samples[sample.id])
    sample.stages['methods'].add(second_water_rinse)
    sample.stages['methods'].add(make_bilayer)
    manager.update_sample(sample)
    time.sleep(1)
    manager.run_sample(sample.id)

    task_id, subtask_id = extract_ids(sample.id, make_bilayer.id)
    # wait until measurement is complete and then read the result
    thread_result = {'result': None}
    monitor_thread = threading.Thread(target=manager.monitor_task, args=(task_id, thread_result), daemon=True)
    monitor_thread.start()
    monitor_thread.join()
    if thread_result['result'].get('success', None) is not None:
        measure_result = manager.get_task_result(task_id, subtask_id)
    else:
        pprint(thread_result)
        measure_result = None

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

    mtag = meas['result']['result']['tags'][0]
    ctag = control['result']['result']['tags'][0]
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

class ROADMAP_Gp(Gp):

    def __init__(self, exp_par, lipids: list[str] = ['DOPC'], manager_address: str = MANAGER_ADDRESS, control_cycle: int = 3, storage_path=None, acq_func="variance", gpcam_iterations=50, gpcam_init_dataset_size=20, gpcam_step=1, keep_plots=False, miniter=1, optimizer='gpcam', parallel_measurements=1, resume=False, show_support_points=True):
        super().__init__(exp_par, storage_path, acq_func, gpcam_iterations, gpcam_init_dataset_size, gpcam_step, keep_plots, miniter, optimizer, parallel_measurements, resume, show_support_points)

        # expected lipid list
        self.lipids = lipids

        # control measurement
        self.control: dict | None = None
        self.control_counter: int = 0
        
        # how often to collect control measurements
        self.control_cycle = control_cycle

        # connect to manager
        self.manager = ManagerInterface(manager_address)
        self.manager.initialize()

    def do_measurement(self, optpars: dict, it_label: str):

        # Configure a particular problem with a set of N lipids. Then there are N-1 keywords describing the composition, plus 1 for the total concentration.

        # break out all non-lipid parameters
        conc = optpars.pop('concentration')

        # add up all fractions of lipids in optpars
        sum_fractions = sum(v for k, v in optpars.values() if k in self.lipids)

        # add back in the missing lipid
        missing_lipid = next(k for k in optpars if k not in self.lipids)
        optpars[missing_lipid] = 1 - sum_fractions

        # collect data, including control if necessary, and increment control counter
        new_control = (self.control_counter % self.control_cycle)==0 | (self.control is None)
        res = collect_data(self.manager, optpars, conc, it_label, control=new_control)
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

    if False:
        res = collect_data(manager, {}, 0.5)
        with open('control.json', 'w') as f:
            json.dump(res, f, indent=2)
        pprint(res)

    else:

        with open('control.json', 'r') as f:
            res = json.load(f)

        result, variance = reduce_qcmd(res['measure'], res['control'])
        print(result, np.sqrt(variance))

        



