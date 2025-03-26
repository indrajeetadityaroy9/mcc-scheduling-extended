from enum import Enum
import random
from typing import Dict, Any

# Global cloud execution times list
cloud_execution_times = [3, 1, 1]

def initialize_edge_execution_times(tasks, num_edge_nodes, num_edge_cores, baseline_cloud_time=5):
    edge_execution_times = {}

    # Determine cloud time from the first task, if available.
    if tasks and hasattr(tasks[0], 'cloud_execution_times') and tasks[0].cloud_execution_times:
        cloud_time = sum(tasks[0].cloud_execution_times)
    else:
        cloud_time = baseline_cloud_time

    for task in tasks:
        task_id = task.id

        # Retrieve local execution times or compute reasonable defaults based on task complexity.
        local_times = getattr(task, 'local_execution_times', None)
        if not local_times:
            complexity_factor = getattr(task, 'complexity', 3.0) / 3.0
            local_times = [9 * complexity_factor, 7 * complexity_factor, 5 * complexity_factor]

        min_local = min(local_times)
        task_type = getattr(task, 'task_type', 'balanced').lower()
        complexity = getattr(task, 'complexity', 3.0)
        data_intensity = getattr(task, 'data_intensity', 1.0)

        # Compute base_edge_time and adjustment_factor based on task type.
        if task_type == 'data':
            base_edge_time = 0.9 * min_local
            adjustment_factor = 1.0 - 0.05 * max(data_intensity - 1.0, 0)
        elif task_type == 'compute':
            base_edge_time = (min_local + cloud_time) / 2.0
            adjustment_factor = 1.0 + 0.05 * max(complexity - 3.0, 0)
        elif task_type == 'balanced':
            base_edge_time = (0.95 * min_local + cloud_time) / 2.0
            adjustment_factor = 1.0 + 0.03 * ((complexity - 3.0) - (data_intensity - 1.0))
        else:
            base_edge_time = (min_local + cloud_time) / 2.0
            adjustment_factor = 1.0

        # Clamp adjustment_factor between 0.5 and 1.5.
        adjustment_factor = max(0.5, min(adjustment_factor, 1.5))
        adjusted_edge_time = base_edge_time * adjustment_factor

        # Ensure task has its own dictionary to store computed times.
        task.edge_execution_times = {}

        # Loop over each edge node and core.
        for edge_id in range(1, num_edge_nodes + 1):
            edge_factor = 1.0 + (edge_id - 1) * 0.1  # 10% degradation per node.
            for core_id in range(1, num_edge_cores + 1):
                core_factor = 1.0 + (core_id - 1) * 0.05  # 5% degradation per core.
                variation = random.uniform(0.97, 1.03)
                computed_time = round(adjusted_edge_time * edge_factor * core_factor * variation, 1)
                task.edge_execution_times[(edge_id, core_id)] = computed_time
                edge_execution_times[(task_id, edge_id, core_id)] = computed_time

    return edge_execution_times


def initialize_device_execution_times(tasks, num_device_cores=3, baseline_cloud_time=5):
    dynamic_core_execution_times = {}

    # Allocate cores into 'big' (performance) cores
    # We only need to identify the big cores - little cores are implied
    if num_device_cores <= 2:
        big_cores = [0]  # First core is performance core
    else:
        num_big = max(1, num_device_cores // 3)
        big_cores = list(range(num_big))  # First num_big cores are performance cores

    # Fixed performance ratios for simplicity
    BIG_LITTLE_RATIO = 2.0  # Standard big/LITTLE performance ratio

    for task in tasks:
        task_id = task.id
        task_type = getattr(task, 'task_type', 'balanced')
        complexity = getattr(task, 'complexity', 3.0)

        # Simplified scaling factors based on task type
        if task_type == 'compute':
            cloud_scaling_factor = 1.5  # Compute tasks run faster locally relative to cloud
            big_little_ratio = BIG_LITTLE_RATIO  # Standard ratio
        elif task_type == 'data':
            cloud_scaling_factor = 0.7  # Data tasks run slower locally relative to cloud
            big_little_ratio = 1.5  # Smaller difference between cores
        else:  # balanced
            cloud_scaling_factor = 1.0  # Equal performance
            big_little_ratio = BIG_LITTLE_RATIO  # Standard ratio

        # Scale based on complexity - linear scaling without normalization
        complexity_factor = complexity / 3.0  # Simple ratio against baseline complexity

        # Calculate base execution time
        base_execution_time = baseline_cloud_time * cloud_scaling_factor * complexity_factor

        # Calculate execution times for each core - no random variation
        execution_times = []
        for idx in range(num_device_cores):
            if idx in big_cores:
                # Performance cores - baseline performance
                execution_times.append(round(base_execution_time, 1))
            else:
                # Efficiency cores - slower by the big/LITTLE ratio
                execution_times.append(round(base_execution_time * big_little_ratio, 1))

        dynamic_core_execution_times[task_id] = execution_times
        task.local_execution_times = execution_times

    return dynamic_core_execution_times


def generate_realistic_power_models(device_type, battery_level, num_edge_nodes, num_cores=3):
    power_models: Dict[str, Dict[Any, Any]] = {'device': {}, 'edge': {}, 'cloud': {}, 'rf': {}}

    if device_type == 'mobile':
        battery_factor = 1.0 if battery_level > 30 else 1.2

        # Dynamic core count support
        for core_id in range(num_cores):
            if core_id == 0:  # Performance core
                power_models['device'][core_id] = {
                    'idle_power': 0.1 * battery_factor,
                    'dynamic_power': lambda load: (0.2 + 1.8 * load) * battery_factor
                }
            elif core_id < num_cores // 3:  # Other performance cores
                power_models['device'][core_id] = {
                    'idle_power': 0.08 * battery_factor,
                    'dynamic_power': lambda load: (0.15 + 1.5 * load) * battery_factor
                }
            else:  # Efficiency cores
                power_models['device'][core_id] = {
                    'idle_power': 0.03 * battery_factor,
                    'dynamic_power': lambda load: (0.05 + 0.95 * load) * battery_factor
                }

        # Simplified RF model - removed signal strength parameters
        power_models['rf'] = {
            'device_to_edge': lambda data_rate, _=None: (0.1 + 0.4 * (data_rate / 10)) * battery_factor,
            'device_to_cloud': lambda data_rate, _=None: (0.15 + 0.6 * (data_rate / 5)) * battery_factor
        }

    elif device_type == 'edge_server':
        # Keep edge computing models with deterministic efficiency calculation
        for edge_id in range(1, num_edge_nodes + 1):
            for core_id in range(1, num_cores + 1):
                # Deterministic efficiency based on node and core ID
                base_efficiency = 1.0 - 0.1 * (edge_id - 1) - 0.05 * (core_id - 1)
                efficiency = max(0.7, base_efficiency)  # Set a minimum efficiency

                power_models['edge'][(edge_id, core_id)] = {
                    'idle_power': 5.0 * efficiency,
                    'dynamic_power': lambda load, eff=efficiency: (3.0 + 12.0 * load) * eff
                }

    # Simplified cloud model - just basic power values needed for reference
    power_models['cloud'] = {
        'idle_power': 50.0,
        'dynamic_power': lambda load: 20.0 + 180.0 * load
    }

    return power_models


def generate_realistic_network_conditions():
    base_upload = {
        'device_to_edge': 10.0,
        'edge_to_edge': 30.0,
        'edge_to_cloud': 50.0,
        'device_to_cloud': 5.0,
    }
    base_download = {
        'edge_to_device': 12.0,
        'cloud_to_edge': 60.0,
        'edge_to_edge': 30.0,
        'cloud_to_device': 6.0,
    }
    random_factor = random.uniform(0.85, 1.15)
    upload_rates = {link: base_rate * random_factor for link, base_rate in base_upload.items()}
    download_rates = {link: base_rate * random_factor for link, base_rate in base_download.items()}

    return upload_rates, download_rates


def add_task_attributes(predefined_tasks, num_edge_nodes, complexity_range, data_intensity_range, task_type_weights= None):
    if task_type_weights is None:
        task_type_weights = {'compute': 0.1, 'data': 0.8, 'balanced': 0.1}

    for task in predefined_tasks:
        task.task_type = random.choices(list(task_type_weights.keys()), weights=list(task_type_weights.values()))[0]

        if task.task_type == 'compute':
            task.complexity = random.uniform(complexity_range[1] * 0.7, complexity_range[1])
        elif task.task_type == 'data':
            task.complexity = random.uniform(complexity_range[0], complexity_range[0] * 2)
        else:
            task.complexity = random.uniform(complexity_range[0], complexity_range[1])

        if task.task_type == 'data':
            task.data_intensity = random.uniform(data_intensity_range[1] * 0.7, data_intensity_range[1])
        elif task.task_type == 'compute':
            task.data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[0] * 2)
        else:
            task.data_intensity = random.uniform(data_intensity_range[0], data_intensity_range[1])

        # Generate dynamic data sizes.
        task.data_sizes = {
            'device_to_cloud': random.uniform(1.0, 5.0),
            'cloud_to_device': random.uniform(1.0, 5.0),
        }
        for i in range(1, num_edge_nodes + 1):
            task.data_sizes[f'device_to_edge{i}'] = random.uniform(0.5, 2.0)
            task.data_sizes[f'edge{i}_to_device'] = random.uniform(0.5, 2.0)
            task.data_sizes[f'edge{i}_to_cloud'] = random.uniform(2.0, 4.0)
            task.data_sizes[f'cloud_to_edge{i}'] = random.uniform(1.0, 3.0)
            for j in range(1, num_edge_nodes + 1):
                if i != j:
                    task.data_sizes[f'edge{i}_to_edge{j}'] = random.uniform(1.0, 3.0)

        print(f"Task {task.id}: Type={task.task_type}, "
              f"Task Complexity={task.complexity:.2f}, Data Intensity={task.data_intensity:.2f}")

    return predefined_tasks


class ExecutionTier(Enum):
    """
    Defines where a task can be executed:
      - DEVICE: on the mobile device,
      - EDGE: on an edge node,
      - CLOUD: on the cloud.
    """
    DEVICE = 0
    EDGE = 1
    CLOUD = 2


class SchedulingState(Enum):
    """
    Represents the state of task scheduling:
      - UNSCHEDULED: not yet scheduled,
      - SCHEDULED: scheduled during initial minimal-delay scheduling,
      - KERNEL_SCHEDULED: rescheduled after energy optimization.
    """
    UNSCHEDULED = 0
    SCHEDULED = 1
    KERNEL_SCHEDULED = 2
