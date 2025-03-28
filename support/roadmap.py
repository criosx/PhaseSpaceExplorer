import numpy as np
import requests
import threading
import time
from pprint import pprint
from urllib.parse import urljoin
from uuid import uuid4
from .gp import Gp

from lh_manager.material_db.db import Material
from lh_manager.liquid_handler.bedlayout import Composition, Solvent, Solute
from lh_manager.liquid_handler.methods import BaseMethod
from lh_manager.liquid_handler.roadmapmethods import ROADMAP_QCMD_MakeBilayer, ROADMAP_QCMD_RinseLoopInjectandMeasure, QCMDRecordTag
from lh_manager.liquid_handler.samplelist import Sample, MethodList
from lh_manager.liquid_handler.samplecontainer import SampleContainer

MANAGER_ADDRESS = 'http://localhost:5001'

class ManagerInterface:
    address: str = MANAGER_ADDRESS
    samples: dict[str, dict] = {}
    sample_ids: list[str] = []
    materials: dict[str, Material] = {}

    def initialize(self):
        self.load_materials()
        self.get_samples()

    def load_materials(self):

        all_materials: list[dict] = requests.get(urljoin(self.address, '/Materials/all/')).json()['materials']

        self.materials = {v['name']: Material(**v) for v in all_materials}

    def get_samples(self):
        
        samples: list[dict] = requests.get(urljoin(self.address, '/GUI/GetSamples/')).json()['samples']

        self.samples = {s['id']: s for s in samples['samples']}

        SampleContainer.model_validate(samples)

    def get_sample_ids(self) -> str:

        return [s['id'] for s in self.samples.values()]

    def new_sample(self, sample: Sample) -> str:

        if sample not in self.sample_ids:
            self.update_sample(sample)
            self.get_sample_ids()
        
    def update_sample(self, sample: Sample) -> str:

        response: dict[str, str] = requests.post(urljoin(self.address, '/GUI/UpdateSample'), data=sample.model_dump_json()).json()

        # should return {'sample added': id} or {}'sample updated': id}

        self.get_samples()

        return list(response.values())[0]
    
    def run_sample(self, sample_id: str) -> str:

        sample = self.samples[sample_id]
        response: dict = requests.post(urljoin(self.address, '/GUI/RunSample/'), json={'name': sample['name'], 'id': sample['id'], 'uuid': sample.get('uuid', None), 'slotID': None, 'stage': ['methods']}).json()
        self.get_samples()
        return response

    def get_task_complete(self, task_id: str) -> str | dict:

        response = requests.get(urljoin(self.address, '/autocontrol/GetTaskStatus'), json={'task_id': task_id})
        if not response.ok:
            return {'error': response.json()}

        if response.json().get('queue', '') == 'history':
            return {'success': 'complete'}
        
        return 'incomplete'

    def monitor_task(self, task_id: str, thread_result: dict, poll_interval: float = 5) -> str:
        current_status = 'incomplete'
        while current_status == 'incomplete':
            time.sleep(poll_interval)
            current_status = self.get_task_complete(task_id=task_id)
        
        thread_result['result'] = current_status

    def get_task_result(self, task_id: str, subtask_id: str) -> dict:

        data = dict(task_id=task_id,
                    subtask_id=subtask_id)

        response: dict = requests.get(urljoin(self.address, '/autocontrol/GetTaskResult'), json=data).json()

        return response

    def solvent_from_material(self, name: str, fraction: float) -> Solvent:
        """Gets solvent properties from material definition

        Args:
            name (str): material name
            fraction (float): fraction

        Returns:
            Solvent: returned solvent object
        """

        material = self.materials[name]

        return Solvent(name=material.name, fraction=fraction)

    def solute_from_material(self, name: str, concentration: float, units: str | None = None) -> Solute:
        """Gets solute properties from material definition

        Args:
            name (str): material name
            concentration (float): concentration
            units (str, optional): units. Defaults to material default units

        Returns:
            Solute: returned solute object
        """

        material = self.materials[name]

        return Solute(name=material.name,
                    molecular_weight=material.molecular_weight,
                    concentration=concentration,
                    units=units if units is not None else material.concentration_units)


manager = ManagerInterface()
manager.initialize()

def collect_data(lipids: dict[str, float], concentration: float) -> tuple[float, float]:
    """Performs a bilayer formation measurement with a lipid composition and total lipid concentration.
    
        TODO: for now, assumes a single channel = 0, but in the future could implement a channel selector
                based on the inputs (substrate) or a rolling counter

    Args:
        lipids (dict[str, float]): dictionary with lipid name as the key and fraction of that lipid as the value
        concentration (float): total lipid concentration in mg/mL

    Returns:
        tuple[float, float]: normalized change in QCMD frequency and error
    """

    sample = Sample(name='PhaseSpaceExplorer',
        description='PhaseSpaceExplorer container sample',
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
                                            Buffer_Composition=Composition(solvents=[manager.solvent_from_material('H2O', fraction=1)],
                                                                           solutes=[manager.solute_from_material('NaCl', concentration=0.15, units='M'),
                                                                                    manager.solute_from_material('tris', concentration=10.0, units='mM')]),
                                            Buffer_Injection_Volume=2.0,
                                            Use_Rinse_System_for_Buffer=True,
                                            Extra_Volume=0.1,
                                            Rinse_Volume=2.0,
                                            Flow_Rate=3.0,
                                            Exchange_Flow_Rate=0.1,
                                            Equilibration_Time=3.0,
                                            Measurement_Time=5.0)

    #test_measure = QCMDRecordTag(sleep_time=10, record_time=20, tag_name='testing')

    # Rinses and control measurement
    sample.stages['methods'].add(ethanol_rinse)
    sample.stages['methods'].add(isopropanol_rinse)
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

def reduce(meas: dict, control: dict) -> float:
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

    def do_measurement(self, optpars: dict, it_label: str):

        # TODO: configure a particular problem with a set of N lipids. Then there are N-1 keywords describing the composition, plus 1 for the total concentration.

        conc = optpars.pop('concentration')
        res = collect_data(optpars, conc, it_label)

        results, variance = reduce(res['measure'], res['control'])

        return results, variance

if __name__ == '__main__':

    import json

    #pprint.pprint(manager.materials)
    if False:
        res = collect_data({}, 0.5)
        with open('control.json', 'w') as f:
            json.dump(res, f, indent=2)
        pprint(res)

    else:

        with open('control.json', 'r') as f:
            res = json.load(f)

        result, variance = reduce(res['measure'], res['control'])
        print(result, np.sqrt(variance))

        



