import datetime
import json
import numpy as np
import pandas as pd
import time

from pathlib import Path
from pprint import pprint
from threading import Event
from uuid import uuid4

from .manager import MANAGER_ADDRESS, ManagerInterface
from ..gp import Gp

from lh_manager.liquid_handler.bedlayout import Composition
from lh_manager.liquid_handler.formulation import SoluteFormulation
from lh_manager.liquid_handler.roadmapmethods import (ROADMAP_QCMD_MakeBilayer,
                                                      ROADMAP_QCMD_RinseLoopInjectandMeasure,
                                                      ROADMAP_QCMD_RinseDirectInjectandMeasure,
                                                      ROADMAP_DirectInjecttoQCMD,
                                                      QCMDRecordTag,
                                                      Formulation,
                                                      TransferWithRinse,
                                                      MixWithRinse,
                                                      InferredWellLocation,
                                                      ROADMAP_DirectInjecttoQCMD)
                                                      
from lh_manager.liquid_handler.samplelist import Sample, MethodList

def collect_data(manager: ManagerInterface,
                 bilayer_composition: Composition,
                 sample_name: str,
                 description: str,
                 control: bool = True,
                 stop_event: Event = Event()) -> tuple[float, float]:
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
                                                      Volume=1,
                                                      Injection_Flow_Rate=2,
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
    
    make_bilayer = ROADMAP_QCMD_MakeBilayer(id=str(uuid4()),
                                            Bilayer_Composition=bilayer_composition,
                                            Bilayer_Solvent=isopropanol,
                                            Use_Rinse_System_for_Solvent=True,
                                            Lipid_Injection_Volume=1.0,
                                            Buffer_Composition=buffer_composition,
                                            Buffer_Injection_Volume=1.2,
                                            Use_Rinse_System_for_Buffer=True,
                                            Extra_Volume=0.1,
                                            Rinse_Volume=2.0,
                                            Flow_Rate=3.0,
                                            Exchange_Flow_Rate=0.1,
                                            Equilibration_Time=5.0,
                                            Measurement_Time=3.0)
    
    # Initiate sample
    sample = Sample(name=sample_name,
        description=repr(bilayer_composition) + ', started ' + description,
        channel=0,
        stages={'methods': MethodList()})

    sample = manager.new_sample(sample=sample)    

    # set up running protocol. Note that measurement methods must have an ID
    methods = [
        ethanol_rinse,
        isopropanol_rinse
        ]
    if control:
        methods += [
            water_rinse,
            buffer_control
        ]
    methods += [
        second_water_rinse,
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
        control_result = manager.wait_for_result(sample, buffer_control.id, stop_event)
    else:
        control_result = None

    print('Waiting for measurement result...')

    # make the bilayer
    measure_result = manager.wait_for_result(sample, make_bilayer.id, stop_event)

    # archive the sample to clean up the GUI
    manager.archive_sample(sample)

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
        return np.nan, np.nan
    harmonics = [1, 3, 5, 7, 9]

    diffs = []
    diff_errs = []
    for imf, icf, h in zip(mtag['f_averages'], ctag['f_averages'], harmonics):
        diffs.append((imf[0] - icf[0]) / (h ** harmonic_power))
        diff_errs.append(np.sqrt(imf[2] ** 2 + icf[2] ** 2) / (h ** harmonic_power))

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

    def __init__(self, exp_par, storage_path=None, acq_func="variance", gpcam_iterations=50, gpcam_init_dataset_size=20, gpcam_step=1, keep_plots=False, miniter=1, optimizer='gpcam', parallel_measurements=1, resume=False, show_support_points=True, project_name='', stop_event: Event = Event()):
        super().__init__(exp_par, storage_path, acq_func, gpcam_iterations, gpcam_init_dataset_size, gpcam_step, keep_plots, miniter, optimizer, parallel_measurements, resume, show_support_points, project_name, stop_event)

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
        self.control: dict | None = self.load_control()
        
        # how often to collect control measurements
        self.control_cycle = 4

        # connect to manager
        self.manager = ManagerInterface(address=MANAGER_ADDRESS)
        self.manager.initialize()
        self._layout = self.manager.get_layout()
        self._diluent = Composition(solvents=[self.manager.solvent_from_material('isopropanol', 1)], solutes=[])

        self.stop_event = stop_event

    def composition_from_parameters(self, args: list) -> Composition:

        labels = list(self.all_par[self.all_par['optimize']]['name'])
        print(labels)
        # break out all non-lipid parameters
        if 'lipid concentration' in labels:
            cidx = labels.index('lipid concentration')
            conc = args.pop(cidx)
            labels.pop(cidx)

        # cycle through optimized lipids and determine absolute fraction. Each lipid is expressed as a fraction of the remainder
        lipid_dict = {}
        sum_remainder = 0.0
        for lipid in self.optimized_lipids:
            frac = args[labels.index(lipid)]
            new_fraction = frac * (1 - sum_remainder)
            lipid_dict[lipid] = float(new_fraction)
            sum_remainder += new_fraction

        # fill in the remainder with the non-optimized lipids in the ratio given in the table values
        sum_non_optimized = sum(v for v in self.non_optimized_lipids.values())
        for lipid, v in self.non_optimized_lipids.items():
            lipid_dict[lipid] = float(v * (1 - sum_remainder) / sum_non_optimized)

        sum_fractions = sum(lipid_dict.values())
        units = 'mg/mL'
        bilayer_composition = Composition(solvents=[self.manager.solvent_from_material('isopropanol', fraction=1)],
                                      solutes=[self.manager.solute_from_material(name, frac * conc / sum_fractions, units) for (name, frac) in lipid_dict.items() if frac > 0])
        
        return bilayer_composition, lipid_dict, conc

    def cost_function(self, origin: np.ndarray, x: np.ndarray, cost_function_parameters):
        # origin has shape V x D
        # x has shape V x D
        # cost_function_parameters can be anything

        print('Calculating cost', x.shape)

        formulation_template = SoluteFormulation(target_composition=Composition(), target_volume=1.5, diluent=self._diluent)

        res = np.ones(x.shape[0])
        for i, ix in enumerate(x):
            formulation_template.target_composition = self.composition_from_parameters(list(ix))[0]
            _, _, success = formulation_template.formulate(self._layout)
            res[i] = formulation_template.estimated_time(self._layout) if success else 1e8
            #print(formulation_template.get_methods(self._layout), [m.estimated_time(self._layout) for m in formulation_template.get_methods(self._layout)], res[i])

        return res

    def load_control(self) -> dict:

        control = None
        storage_path = Path(self.spath) / 'results' / 'results.json'
        if storage_path.exists():
            with open(storage_path, 'r') as f:
                current_results: dict = json.load(f)

            control = current_results.get('control', None)

        return control

    def save_result(self, it_label: str, optpars: dict, raw_result: dict, reduced_result: dict):

        storage_path = Path(self.spath) / 'results' / 'results.json'
        if storage_path.exists():

            with open(storage_path, 'r') as f:
                current_results = json.load(f)
        else:
            current_results = {}

        current_results.update({'control': self.control,
                                str(it_label): dict(optpars=optpars,
                                                   raw_result=raw_result,
                                                   reduced_result=reduced_result)})
        with open(storage_path, 'w') as f:
            json.dump(current_results, f)

    def do_measurement_test(self, optpars, it_label):
        
        total_time = 0
        while (not self.stop_event.is_set()) & (total_time < 6):
            time.sleep(2)
            total_time += 2

        return 0.1, 0.1

    def do_measurement(self, optpars: dict, it_label: str):

        # Configure a particular problem with a set of N lipids. Then there are N-1 keywords describing the composition, plus 1 for the total concentration.

        bilayer_composition, lipid_dict, conc = self.composition_from_parameters(list(optpars.values()))

        cost = self.cost_function(None, np.array([list(optpars.values())]), None)[0]

        print(f'Starting measurement {it_label} with concentration {conc} and lipids {lipid_dict}, cost {cost}: ')

        time.sleep(0.1)
        return (cost, cost ** 0.5) if cost < 1e8 else (0.0, 0.1)

        # collect data, including control if necessary, and increment control counter
        new_control = (((int(it_label) % self.control_cycle) == 0) | (self.control is None))
        
        sample_name = f'{self.project_name} point {it_label}'
        description = datetime.datetime.now().strftime("%Y%m%d %H.%M.%S")
        if 'ipa_fraction' in optpars:
            res = collect_wateripa(self.manager, optpars['ipa_fraction'], sample_name, description, control=new_control, stop_event=self.stop_event)
            harmonic_power = 0.5
        else:
            res = collect_data(self.manager, bilayer_composition, sample_name, description, control=new_control, stop_event=self.stop_event)
            harmonic_power = 1.0

        # replace current value of control if applicable
        if new_control:
            self.control = res['control']

        # reduce data with most recent control
        results, variance = reduce_qcmd(res['measure'], self.control, harmonic_power)

        #res = {'measure': None, 'control': None}
        #results, variance = 0.0, 0.0

        # update layout
        self._layout = self.manager.get_layout()

        self.save_result(it_label, optpars, raw_result=res, reduced_result=dict(results=results,
                                                                                variance=variance))

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

        



