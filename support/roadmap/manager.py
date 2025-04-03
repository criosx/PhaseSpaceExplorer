"""Interface for lh_manager (github.com/roadmap-automation/lh_manager)"""

import json
import requests
import time
from urllib.parse import urljoin

from lh_manager.material_db.db import Material
from lh_manager.liquid_handler.bedlayout import Solvent, Solute
from lh_manager.liquid_handler.samplelist import Sample
from lh_manager.liquid_handler.samplecontainer import SampleContainer

MANAGER_ADDRESS = 'http://localhost:5001'

class ManagerInterface:
    samples: dict[str, dict] = {}
    sample_ids: list[str] = []
    materials: dict[str, Material] = {}

    def __init__(self, address: str = MANAGER_ADDRESS):
        self.address = address

    def initialize(self):
        self.load_materials()
        self.get_samples()

    def load_materials(self):

        all_materials: list[dict] = requests.get(urljoin(self.address, '/Materials/all/')).json()['materials']

        self.materials = {v['name']: Material(**v) for v in all_materials}

    def get_samples(self):
        
        samples: list[dict] = requests.get(urljoin(self.address, '/GUI/GetSamples/')).json()['samples']

        self.samples = {s['id']: s for s in samples['samples']}
        #SampleContainer.model_validate(samples)

    def get_sample_ids(self) -> list[str]:

        return list(self.samples.keys())

    def new_sample(self, sample: Sample) -> Sample:

        if sample not in self.sample_ids:
            _, sample = self.update_sample(sample)

        return sample
        
    def update_sample(self, sample: Sample) -> tuple[str, Sample]:

        response: dict[str, str] = requests.post(urljoin(self.address, '/GUI/UpdateSample'), data=sample.model_dump_json()).json()

        # should return {'sample added': id} or {}'sample updated': id}

        self.get_samples()

        return list(response.values())[0], self.rehydrate_sample(sample.id)

    def rehydrate_sample(self, sample_id: str) -> Sample:

        return Sample.model_validate(self.samples[sample_id])

    def run_sample(self, sample_id: str) -> tuple[str, Sample]:

        sample = self.samples[sample_id]
        response: dict = requests.post(urljoin(self.address, '/GUI/RunSample/'), json={'name': sample['name'], 'id': sample['id'], 'uuid': sample.get('uuid', None), 'slotID': None, 'stage': ['methods']}).json()
        self.get_samples()
        return response, self.rehydrate_sample(sample_id)

    def get_task_complete(self, task_id: str) -> str | dict:

        response = requests.get(urljoin(self.address, '/autocontrol/GetTaskStatus'), json={'task_id': task_id})
        try:
            resp = response.json()
        except json.JSONDecodeError:
            return {'error': 'json could not be decoded'}
        
        if not response.ok:
            return {'error': resp}

        if resp.get('queue', '') == 'history':
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
