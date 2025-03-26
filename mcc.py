from copy import deepcopy
import bisect
from dataclasses import dataclass
from collections import deque
import numpy as np
from heapq import heappush, heappop
from enum import Enum
from data import SchedulingState
from data import generate_realistic_power_models, generate_realistic_network_conditions
from data import initialize_device_execution_times
from data import add_task_attributes


# Simplified ExecutionTier enum (two-tier model)
class ExecutionTier(Enum):
    DEVICE = 0
    CLOUD = 1

# Dictionary storing execution times for tasks on different cores
# This will be replaced with dynamically generated values
core_execution_times = {
    1: [9, 7, 5],
    2: [8, 6, 5],
    3: [6, 5, 4],
    4: [7, 5, 3],
    5: [5, 4, 2],
    6: [7, 6, 4],
    7: [8, 5, 3],
    8: [6, 4, 2],
    9: [5, 3, 2],
    10: [7, 4, 2],
    11: [12, 3, 3],
    12: [12, 8, 4],
    13: [11, 3, 2],
    14: [12, 11, 4],
    15: [13, 4, 2],
    16: [9, 7, 3],
    17: [9, 3, 3],
    18: [13, 9, 2],
    19: [10, 5, 3],
    20: [12, 5, 4]
}

# Cloud execution parameters
# These will be replaced with dynamically generated values
cloud_execution_times = [3, 1, 1]


@dataclass
class TaskMigrationState:
    # Class to track task migration decisions
    time: float  # T_total: Total completion time after migration
    energy: float  # E_total: Total energy consumption after migration
    efficiency: float  # Energy reduction per unit time (used for migration decisions)
    task_index: int  # v_tar: Task selected for migration
    target_execution_unit: int  # k_tar: Target execution unit (core or cloud)


class Task(object):
    def __init__(self, id, pred_tasks=None, succ_task=None):
        # Basic task graph structure
        self.id = id
        self.pred_tasks = pred_tasks or []
        self.succ_task = succ_task or []

        # Task execution timing parameters
        self.core_execution_times = core_execution_times.get(id, [5, 4, 3])
        self.cloud_execution_times = cloud_execution_times

        # Task completion timing parameters
        self.FT_l = 0  # Local core finish time
        self.FT_ws = 0  # Wireless sending finish time
        self.FT_c = 0  # Cloud computation finish time
        self.FT_wr = 0  # Wireless receiving finish time

        # Ready Times
        self.RT_l = -1  # Ready time for local execution
        self.RT_ws = -1  # Ready time for wireless sending
        self.RT_c = -1  # Ready time for cloud execution
        self.RT_wr = -1  # Ready time for receiving results

        # Task scheduling parameters
        self.priority_score = None
        self.assignment = -2
        self.is_core_task = False
        self.execution_unit_task_start_times = [-1, -1, -1, -1]
        self.execution_finish_time = -1
        self.is_scheduled = SchedulingState.UNSCHEDULED


def total_time(tasks):
    # Implementation of total completion time calculation
    return max(
        max(task.FT_l, task.FT_wr)
        for task in tasks
        if not task.succ_task
    )


def calculate_energy_consumption(task, core_powers, cloud_sending_power):
    # Calculate energy consumption for a single task
    if task.is_core_task:
        return core_powers[task.assignment] * task.core_execution_times[task.assignment]
    else:
        return cloud_sending_power * task.cloud_execution_times[0]


def total_energy(tasks, core_powers, cloud_sending_power):
    # Calculate total energy consumption
    return sum(
        calculate_energy_consumption(task, core_powers, cloud_sending_power)
        for task in tasks
    )


# NEW FUNCTIONS FOR ENHANCED MCC

def enhance_tasks(tasks):
    """
    Enhance tasks with attributes from data.py
    """
    complexity_range = (1.0, 5.0)
    data_intensity_range = (1.0, 5.0)

    # Set num_edge_nodes=0 since we're not using edge computing
    return add_task_attributes(tasks, num_edge_nodes=0,
                               complexity_range=complexity_range,
                               data_intensity_range=data_intensity_range)


def initialize_device_times(tasks, num_device_cores=3):
    """
    Replace hardcoded core_execution_times with dynamically generated values
    """
    # Generate device execution times
    device_times = initialize_device_execution_times(tasks, num_device_cores)

    # Update task objects with execution time models
    for task in tasks:
        if task.id in device_times:
            task.core_execution_times = device_times[task.id]

    return tasks


def initialize_power_models(device_type='mobile', battery_level=75, num_cores=3):
    """
    Generate realistic power models for device and cloud
    """

    # Generate power models (set num_edge_nodes=0 for two-tier model)
    power_models = generate_realistic_power_models(
        device_type, battery_level, num_edge_nodes=0, num_cores=num_cores)

    # Extract core powers for the original algorithm format
    core_powers = [power_models['device'][i]['dynamic_power'](1.0)
                   for i in range(num_cores)]

    # Extract RF power for cloud sending
    cloud_sending_power = power_models['rf']['device_to_cloud'](5.0)

    return power_models, core_powers, cloud_sending_power


def apply_network_conditions(tasks):
    """
    Apply realistic network conditions to cloud execution times
    """
    # Generate realistic network conditions
    upload_rates, download_rates = generate_realistic_network_conditions()

    for task in tasks:
        # If task has data_sizes attribute
        if hasattr(task, 'data_sizes'):
            # Calculate send time based on data size and network rate
            send_time = task.data_sizes['device_to_cloud'] / upload_rates['device_to_cloud']

            # Get or set cloud computation time
            if hasattr(task, 'cloud_execution_times') and len(task.cloud_execution_times) >= 2:
                cloud_time = task.cloud_execution_times[1]
            else:
                cloud_time = 1.0  # Default cloud computation time

            # Calculate receive time
            receive_time = task.data_sizes['cloud_to_device'] / download_rates['cloud_to_device']

            # Update cloud execution times
            task.cloud_execution_times = [send_time, cloud_time, receive_time]

    return tasks


# ORIGINAL MCC ALGORITHM FUNCTIONS

def primary_assignment(tasks):
    """
    Implements the "Primary Assignment" phase described in Section III.A.1.
    """
    for task in tasks:
        # Calculate T_i^l_min (minimum local execution time)
        t_l_min = min(task.core_execution_times)

        # Calculate T_i^re (remote execution time)
        t_re = (task.cloud_execution_times[0] +  # T_i^s (send)
                task.cloud_execution_times[1] +  # T_i^c (cloud)
                task.cloud_execution_times[2])  # T_i^r (receive)

        # Task assignment decision
        if t_re < t_l_min:
            task.is_core_task = False  # Mark as cloud task
        else:
            task.is_core_task = True  # Mark for local execution


def task_prioritizing(tasks):
    """
    Implements the "Task Prioritizing" phase described in Section III.A.2.
    """
    w = [0] * len(tasks)
    # Step 1: Calculate computation costs (wi) for each task
    for i, task in enumerate(tasks):
        if not task.is_core_task:
            # For cloud tasks
            w[i] = (task.cloud_execution_times[0] +  # Ti^s
                    task.cloud_execution_times[1] +  # Ti^c
                    task.cloud_execution_times[2])  # Ti^r
        else:
            # For local tasks
            w[i] = sum(task.core_execution_times) / len(task.core_execution_times)

    # Cache for memoization of priority calculations
    computed_priority_scores = {}

    def calculate_priority(task):
        """
        Recursive implementation of priority calculation.
        """
        # Memoization check
        if task.id in computed_priority_scores:
            return computed_priority_scores[task.id]

        # Base case: Exit tasks
        if task.succ_task == []:
            computed_priority_scores[task.id] = w[task.id - 1]
            return w[task.id - 1]

        # Recursive case: Non-exit tasks
        max_successor_priority = max(calculate_priority(successor)
                                     for successor in task.succ_task)
        task_priority = w[task.id - 1] + max_successor_priority
        computed_priority_scores[task.id] = task_priority
        return task_priority

    # Calculate priorities for all tasks using recursive algorithm
    for task in tasks:
        calculate_priority(task)

    # Update priority scores in task objects
    for task in tasks:
        task.priority_score = computed_priority_scores[task.id]


class InitialTaskScheduler:
    """
    Implements the initial scheduling algorithm described in Section III.A.
    """

    def __init__(self, tasks, num_cores=3):
        """
        Initialize scheduler with tasks and resources.
        """
        self.tasks = tasks
        self.k = num_cores  # K cores from paper

        # Resource timing tracking
        self.core_earliest_ready = [0] * self.k  # When each core becomes available
        self.ws_ready = 0  # Next available time for RF sending channel
        self.wr_ready = 0  # Next available time for RF receiving channel

        # Sk sequence sets
        self.sequences = [[] for _ in range(self.k + 1)]

    def get_priority_ordered_tasks(self):
        """
        Orders tasks by priority scores calculated in task_prioritizing().
        """
        task_priority_list = [(task.priority_score, task.id) for task in self.tasks]
        task_priority_list.sort(reverse=True)  # Higher priority first
        return [item[1] for item in task_priority_list]

    def classify_entry_tasks(self, priority_order):
        """
        Separates tasks into entry and non-entry tasks while maintaining priority order.
        """
        entry_tasks = []
        non_entry_tasks = []

        # Process tasks in priority order
        for task_id in priority_order:
            task = self.tasks[task_id - 1]

            # Check if task has predecessors
            if not task.pred_tasks:
                # Entry tasks have no predecessors
                entry_tasks.append(task)
            else:
                # Non-entry tasks must wait for predecessors
                non_entry_tasks.append(task)

        return entry_tasks, non_entry_tasks

    def identify_optimal_local_core(self, task, ready_time=0):
        """
        Finds optimal local core assignment for a task to minimize finish time.
        """
        # Initialize with worst-case values
        best_finish_time = float('inf')
        best_core = -1
        best_start_time = float('inf')

        # Try each available core k
        for core in range(self.k):
            # Calculate earliest possible start time on this core
            start_time = max(ready_time, self.core_earliest_ready[core])

            # Calculate finish time
            finish_time = start_time + task.core_execution_times[core]

            # Keep track of core that gives earliest finish time
            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_core = core
                best_start_time = start_time

        return best_core, best_start_time, best_finish_time

    def schedule_on_local_core(self, task, core, start_time, finish_time):
        """
        Assigns a task to a local core and updates all relevant timing information.
        """
        # Set task finish time on local core
        task.FT_l = finish_time
        # Set overall execution finish time
        task.execution_finish_time = finish_time
        # Initialize execution start times array
        task.execution_unit_task_start_times = [-1] * (self.k + 1)
        # Record actual start time on assigned core
        task.execution_unit_task_start_times[core] = start_time
        # Update core availability for next task
        self.core_earliest_ready[core] = finish_time
        # Set task assignment
        task.assignment = core
        # Mark task as scheduled
        task.is_scheduled = SchedulingState.SCHEDULED
        # Add task to execution sequence for this core
        self.sequences[core].append(task.id)

    def calculate_cloud_phases_timing(self, task):
        # Phase 1: RF Sending Phase (unchanged)
        send_ready = task.RT_ws
        send_finish = send_ready + task.cloud_execution_times[0]

        # Phase 2: Cloud Computing Phase (FIXED)
        # Check ALL predecessors, not just cloud predecessors
        pred_cloud_finish_times = [
            pred_task.FT_c for pred_task in task.pred_tasks
            if not pred_task.is_core_task and pred_task.FT_c > 0
        ]

        cloud_ready = max(
            send_finish,
            max(pred_cloud_finish_times, default=0)
        )
        cloud_finish = cloud_ready + task.cloud_execution_times[1]

        # Phase 3: RF Receiving Phase (FIXED)
        # Must use channel availability
        receive_ready = max(self.wr_ready, cloud_finish)
        receive_finish = receive_ready + task.cloud_execution_times[2]

        return send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish

    def schedule_on_cloud(self, task, send_ready, send_finish, cloud_ready, cloud_finish, receive_ready,
                          receive_finish):
        """
        Schedules a task for cloud execution, updating all timing parameters.
        """
        # Set timing parameters for three-phase cloud execution
        # Phase 1: RF Sending Phase
        task.RT_ws = send_ready  # When we can start sending
        task.FT_ws = send_finish  # When sending completes

        # Phase 2: Cloud Computing Phase
        task.RT_c = cloud_ready  # When cloud can start
        task.FT_c = cloud_finish  # When cloud computation ends

        # Phase 3: RF Receiving Phase
        task.RT_wr = receive_ready  # When results are ready
        task.FT_wr = receive_finish  # When results are received

        # Set overall execution finish time
        task.execution_finish_time = receive_finish

        # Clear local core finish time
        task.FT_l = 0

        # Initialize execution unit timing array
        task.execution_unit_task_start_times = [-1] * (self.k + 1)

        # Record cloud execution start time
        task.execution_unit_task_start_times[self.k] = send_ready

        # Set task assignment
        task.assignment = self.k

        # Mark task as scheduled
        task.is_scheduled = SchedulingState.SCHEDULED

        # Update wireless channel availability
        self.ws_ready = send_finish
        self.wr_ready = receive_finish

        # Add to cloud execution sequence
        self.sequences[self.k].append(task.id)

    def schedule_entry_tasks(self, entry_tasks):
        """
        Schedules tasks with no predecessors.
        """
        # Track tasks marked for cloud execution
        cloud_entry_tasks = []

        # First Phase: Schedule tasks assigned to local cores
        for task in entry_tasks:
            if task.is_core_task:
                # Find optimal core assignment
                core, start_time, finish_time = self.identify_optimal_local_core(task)

                # Schedule on chosen core
                self.schedule_on_local_core(task, core, start_time, finish_time)
            else:
                # Collect cloud tasks for second phase
                cloud_entry_tasks.append(task)

        # Second Phase: Schedule cloud tasks
        for task in cloud_entry_tasks:
            # Set wireless send ready time
            task.RT_ws = self.ws_ready

            # Calculate timing for three-phase cloud execution
            timing = self.calculate_cloud_phases_timing(task)

            # Schedule cloud execution
            self.schedule_on_cloud(task, *timing)

    def calculate_non_entry_task_ready_times(self, task):
        """
        Calculates ready times for tasks that have predecessors.
        """
        # Calculate local core ready time
        task.RT_l = max(
            max(max(pred_task.FT_l, pred_task.FT_wr)
                for pred_task in task.pred_tasks),
            0  # Ensure non-negative ready time
        )

        # Calculate cloud sending ready time
        task.RT_ws = max(
            max(max(pred_task.FT_l, pred_task.FT_ws)
                for pred_task in task.pred_tasks),
            self.ws_ready  # Channel availability
        )

    def schedule_non_entry_tasks(self, non_entry_tasks):
        """
        Schedules tasks that have predecessors.
        """
        # Process tasks in priority order
        for task in non_entry_tasks:
            # Calculate ready times based on predecessor finish times
            self.calculate_non_entry_task_ready_times(task)

            # If task was marked for cloud in primary assignment
            if not task.is_core_task:
                # Calculate three-phase cloud execution timing
                timing = self.calculate_cloud_phases_timing(task)
                # Schedule task on cloud
                self.schedule_on_cloud(task, *timing)
            else:
                # For tasks marked for local execution:
                # 1. Find best local core option
                core, start_time, finish_time = self.identify_optimal_local_core(
                    task, task.RT_l  # Consider ready time
                )

                # 2. Calculate cloud execution option for comparison
                timing = self.calculate_cloud_phases_timing(task)
                cloud_finish_time = timing[-1]  # FTi^wr

                # 3. Choose execution path with earlier finish time
                if finish_time <= cloud_finish_time:
                    # Local execution is faster
                    self.schedule_on_local_core(task, core, start_time, finish_time)
                else:
                    # Cloud execution is faster
                    task.is_core_task = False
                    self.schedule_on_cloud(task, *timing)


def execution_unit_selection(tasks):
    """
    Implements execution unit selection phase.
    """
    # Initialize scheduler with tasks and K=3 cores
    scheduler = InitialTaskScheduler(tasks, 3)

    # Order tasks by priority score
    priority_orderered_tasks = scheduler.get_priority_ordered_tasks()

    # Classify tasks based on dependencies
    entry_tasks, non_entry_tasks = scheduler.classify_entry_tasks(priority_orderered_tasks)

    # Schedule entry tasks
    scheduler.schedule_entry_tasks(entry_tasks)

    # Schedule non-entry tasks
    scheduler.schedule_non_entry_tasks(non_entry_tasks)

    # Return task sequences for each execution unit
    return scheduler.sequences


def construct_sequence(tasks, task_id, execution_unit, original_sequence):
    """
    Constructs new sequence after task migration while preserving task precedence.
    """
    # Step 1: Create task lookup dictionary for O(1) access
    task_id_to_task = {task.id: task for task in tasks}

    # Step 2: Get the target task v_tar for migration
    target_task = task_id_to_task.get(task_id)

    # Step 3: Get ready time for insertion
    target_task_rt = target_task.RT_l if target_task.is_core_task else target_task.RT_ws

    # Step 4: Remove task from original sequence
    original_assignment = target_task.assignment
    original_sequence[original_assignment].remove(target_task.id)

    # Step 5: Get sequence for new execution unit
    new_sequence_task_list = original_sequence[execution_unit]

    # Get start times for tasks in new sequence
    start_times = [
        task_id_to_task[task_id].execution_unit_task_start_times[execution_unit]
        for task_id in new_sequence_task_list
    ]

    # Step 6: Find insertion point using binary search
    insertion_index = bisect.bisect_left(start_times, target_task_rt)

    # Step 7: Insert task at correct position
    new_sequence_task_list.insert(insertion_index, target_task.id)

    # Step 8: Update task execution information
    target_task.assignment = execution_unit
    target_task.is_core_task = (execution_unit != 3)  # 3 indicates cloud

    return original_sequence


class KernelScheduler:
    """
    Implements the kernel (rescheduling) algorithm for task migration.
    """

    def __init__(self, tasks, sequences):
        """
        Initialize kernel scheduler for task migration rescheduling.
        """
        self.tasks = tasks
        self.sequences = sequences

        # Resource timing trackers
        self.RT_ls = [0] * 3  # Three cores
        self.cloud_phases_ready_times = [0] * 3

        # Initialize task readiness tracking vectors
        self.dependency_ready, self.sequence_ready = self.initialize_task_state()

    def initialize_task_state(self):
        """
        Initializes task readiness tracking vectors.
        """
        # Initialize ready1 vector (dependency tracking)
        dependency_ready = [len(task.pred_tasks) for task in self.tasks]

        # Initialize ready2 vector (sequence position tracking)
        sequence_ready = [-1] * len(self.tasks)

        # Process each execution sequence Sk
        for sequence in self.sequences:
            if sequence:  # Non-empty sequence
                # Mark first task in sequence as ready
                sequence_ready[sequence[0] - 1] = 0

        return dependency_ready, sequence_ready

    def update_task_state(self, task):
        """
        Updates readiness vectors for a task after scheduling changes.
        """
        # Only update state for unscheduled tasks
        if task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
            # Update ready1 vector (dependency tracking)
            self.dependency_ready[task.id - 1] = sum(
                1 for pred_task in task.pred_tasks
                if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
            )

            # Update ready2 vector (sequence position)
            for sequence in self.sequences:
                if task.id in sequence:
                    idx = sequence.index(task.id)
                    if idx > 0:
                        # Task has predecessor in sequence
                        # Check if predecessor has been scheduled
                        prev_task = self.tasks[sequence[idx - 1] - 1]
                        self.sequence_ready[task.id - 1] = (
                            # 1: Waiting for predecessor
                            # 0: Predecessor completed
                            1 if prev_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
                            else 0
                        )
                    else:
                        # First task in sequence
                        self.sequence_ready[task.id - 1] = 0
                    break

    def schedule_local_task(self, task):
        """
        Schedules a task for local core execution.
        """
        # Calculate ready time RTi^l for local execution
        if not task.pred_tasks:
            # Entry tasks can start immediately
            task.RT_l = 0
        else:
            # Find latest completion time among predecessors
            pred_task_completion_times = (
                max(pred_task.FT_l, pred_task.FT_wr)
                for pred_task in task.pred_tasks
            )
            task.RT_l = max(pred_task_completion_times, default=0)

        # Schedule on assigned core k
        core_index = task.assignment
        # Initialize execution timing array
        task.execution_unit_task_start_times = [-1] * 4

        # Calculate actual start time considering:
        # 1. Task ready time
        # 2. Core availability
        task.execution_unit_task_start_times[core_index] = max(
            self.RT_ls[core_index],  # Core availability
            task.RT_l  # Task ready time
        )

        # Calculate finish time
        task.FT_l = (
                task.execution_unit_task_start_times[core_index] +
                task.core_execution_times[core_index]
        )

        # Update core's next available time
        self.RT_ls[core_index] = task.FT_l

        # Clear cloud execution timings
        task.FT_ws = -1
        task.FT_c = -1
        task.FT_wr = -1

    def schedule_cloud_task(self, task):
        """
        Schedules three-phase cloud execution with proper dependency management.
        Fixes issues with cloud dependencies and wireless channel conflicts.
        """
        # Calculate wireless sending ready time
        if not task.pred_tasks:
            # Entry tasks can start sending immediately
            task.RT_ws = 0
        else:
            # Find latest completion time among predecessors
            pred_task_completion_times = (
                max(pred_task.FT_l, pred_task.FT_ws)
                for pred_task in task.pred_tasks
            )
            task.RT_ws = max(pred_task_completion_times)

        # Initialize timing array for execution units
        task.execution_unit_task_start_times = [-1] * 4

        # Set cloud start time considering channel availability
        task.execution_unit_task_start_times[3] = max(
            self.cloud_phases_ready_times[0],  # Channel availability
            task.RT_ws  # Task ready time
        )

        # Phase 1: RF Sending Phase
        task.FT_ws = (
                task.execution_unit_task_start_times[3] +
                task.cloud_execution_times[0]  # Ti^s
        )
        # Update sending channel availability
        self.cloud_phases_ready_times[0] = task.FT_ws

        # Phase 2: Cloud Computing Phase - FIXED
        # Collect all cloud predecessors' finish times
        cloud_pred_finish_times = [
            pred_task.FT_c for pred_task in task.pred_tasks
            if not pred_task.is_core_task and pred_task.FT_c > 0
        ]

        # Ready time is the maximum of sending completion and predecessors' cloud finish times
        task.RT_c = max(
            task.FT_ws,  # Must finish sending
            max(cloud_pred_finish_times, default=0)
        )

        # Calculate cloud finish time
        task.FT_c = task.RT_c + task.cloud_execution_times[1]  # Ti^c

        # Update cloud availability
        self.cloud_phases_ready_times[1] = task.FT_c

        # Phase 3: RF Receiving Phase - FIXED
        # Must consider wireless receiving channel availability
        task.RT_wr = max(self.cloud_phases_ready_times[2], task.FT_c)

        # Calculate receiving finish time
        task.FT_wr = task.RT_wr + task.cloud_execution_times[2]  # Ti^r

        # Update receiving channel availability
        self.cloud_phases_ready_times[2] = task.FT_wr

        # Clear local execution timing
        task.FT_l = -1

    def initialize_queue(self):
        """
        Initializes LIFO stack for linear-time scheduling.
        """
        # Create LIFO stack (implemented as deque)
        return deque(
            task for task in self.tasks
            if (
                # Check sequence readiness (ready2[i] = 0)
                    self.sequence_ready[task.id - 1] == 0
                    and
                    # Check dependency readiness (ready1[i] = 0)
                    all(pred_task.is_scheduled == SchedulingState.KERNEL_SCHEDULED
                        for pred_task in task.pred_tasks)
            )
        )


def kernel_algorithm(tasks, sequences):
    """
   Implements the kernel (rescheduling) algorithm for task migration.
   """
    # Initialize kernel scheduler
    scheduler = KernelScheduler(tasks, sequences)

    # Initialize LIFO stack with ready tasks
    queue = scheduler.initialize_queue()

    # Main scheduling loop
    while queue:
        # Pop next ready task from stack
        current_task = queue.popleft()
        # Mark as scheduled in kernel phase
        current_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED

        # Schedule based on execution type
        if current_task.is_core_task:
            # Schedule on assigned local core k
            scheduler.schedule_local_task(current_task)
        else:
            # Schedule three-phase cloud execution
            scheduler.schedule_cloud_task(current_task)

        # Update ready1 and ready2 vectors
        for task in tasks:
            scheduler.update_task_state(task)

            # Add newly ready tasks to stack
            if (scheduler.dependency_ready[task.id - 1] == 0 and  # ready1[j] = 0
                    scheduler.sequence_ready[task.id - 1] == 0 and  # ready2[j] = 0
                    task.is_scheduled != SchedulingState.KERNEL_SCHEDULED and
                    task not in queue):
                queue.append(task)

    # Reset scheduling state for next iteration
    for task in tasks:
        task.is_scheduled = SchedulingState.UNSCHEDULED

    return tasks


def generate_cache_key(tasks, task_idx, target_execution_unit):
    """
    Generates cache key for memoizing migration evaluations.
    """
    # Create cache key
    return (task_idx, target_execution_unit, tuple(task.assignment for task in tasks))


def evaluate_migration(tasks, seqs, task_idx, target_execution_unit, migration_cache, core_powers, cloud_sending_power):
    """
    Evaluates potential task migration scenario.
    """
    # Generate cache key for this migration scenario
    if core_powers is None:
        core_powers = [1, 2, 4]
    cache_key = generate_cache_key(tasks, task_idx, target_execution_unit)

    # Check cache for previously evaluated scenario
    if cache_key in migration_cache:
        return migration_cache[cache_key]

    # Create copies to avoid modifying original state
    sequence_copy = [seq.copy() for seq in seqs]
    tasks_copy = deepcopy(tasks)

    # Apply migration and recalculate schedule
    sequence_copy = construct_sequence(
        tasks_copy,
        task_idx + 1,  # Convert to 1-based task ID
        target_execution_unit,
        sequence_copy
    )
    kernel_algorithm(tasks_copy, sequence_copy)

    # Calculate new metrics
    migration_T = total_time(tasks_copy)
    migration_E = total_energy(tasks_copy, core_powers, cloud_sending_power)

    # Cache results
    migration_cache[cache_key] = (migration_T, migration_E)
    return migration_T, migration_E


def initialize_migration_choices(tasks):
    # Create matrix of migration possibilities
    migration_choices = np.zeros((len(tasks), 4), dtype=bool)

    # Set valid migration targets for each task
    for i, task in enumerate(tasks):
        if task.assignment == 3:  # Cloud task
            # Can migrate to any local core (not back to cloud)
            migration_choices[i, :3] = True
        else:  # Local core task
            # Can migrate to other cores or cloud (not current core)
            for j in range(4):
                if j != task.assignment:
                    migration_choices[i, j] = True

    return migration_choices


def identify_optimal_migration(migration_trials_results, T_final, E_total, T_max):
    """
    Identifies optimal task migration.
    """
    # Step 1: Find migrations that reduce energy without increasing time
    best_energy_reduction = 0
    best_migration = None

    for task_idx, resource_idx, time, energy in migration_trials_results:
        # Skip migrations violating T_max constraint
        if time > T_max:
            continue

        # Calculate potential energy reduction
        energy_reduction = E_total - energy

        # Check if migration reduces energy without increasing time
        if time <= T_final and energy_reduction > 0:
            if energy_reduction > best_energy_reduction:
                best_energy_reduction = energy_reduction
                best_migration = (task_idx, resource_idx, time, energy)

    # Return best energy-reducing migration if found
    if best_migration:
        task_idx, resource_idx, time, energy = best_migration
        return TaskMigrationState(
            time=time,
            energy=energy,
            efficiency=best_energy_reduction,
            task_index=task_idx + 1,
            target_execution_unit=resource_idx
        )

    # Step 2: If no direct energy reduction found, look for best energy/time tradeoff
    migration_candidates = []
    for task_idx, resource_idx, time, energy in migration_trials_results:
        # Maintain T_max constraint
        if time > T_max:
            continue

        # Calculate energy reduction
        energy_reduction = E_total - energy
        if energy_reduction > 0:
            # Calculate efficiency ratio
            time_increase = max(0, time - T_final)
            if time_increase == 0:
                efficiency = float('inf')  # Prioritize no time increase
            else:
                efficiency = energy_reduction / time_increase

            heappush(migration_candidates,
                     (-efficiency, task_idx, resource_idx, time, energy))

    if not migration_candidates:
        return None

    # Return migration with best efficiency ratio
    neg_ratio, n_best, k_best, T_best, E_best = heappop(migration_candidates)
    return TaskMigrationState(
        time=T_best,
        energy=E_best,
        efficiency=-neg_ratio,
        task_index=n_best + 1,
        target_execution_unit=k_best
    )


def optimize_task_scheduling(tasks, sequence, T_final, core_powers, cloud_sending_power):
    """
    Implements the task migration algorithm for energy optimization.
    """
    # Convert core powers to numpy array for efficient operations
    core_powers = np.array(core_powers)

    # Cache for memoizing migration evaluations
    migration_cache = {}

    # Calculate initial energy consumption
    current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power)

    # Iterative improvement loop
    energy_improved = True
    while energy_improved:
        # Store current energy for comparison
        previous_iteration_energy = current_iteration_energy

        # Get current schedule metrics
        current_time = total_time(tasks)  # T_total
        T_max = T_final * 1.5  # Allow some scheduling flexibility

        # Initialize migration possibilities matrix
        migration_choices = initialize_migration_choices(tasks)

        # Evaluate all valid migration options
        migration_trials_results = []
        for task_idx in range(len(tasks)):
            for possible_execution_unit in range(4):
                if not migration_choices[task_idx, possible_execution_unit]:
                    continue

                # Calculate T_total and E_total after migration
                migration_trial_time, migration_trial_energy = evaluate_migration(
                    tasks, sequence, task_idx, possible_execution_unit, migration_cache, core_powers,
                    cloud_sending_power
                )
                migration_trials_results.append(
                    (task_idx, possible_execution_unit,
                     migration_trial_time, migration_trial_energy)
                )

        # Select best migration using two-step criteria
        best_migration = identify_optimal_migration(
            migration_trials_results=migration_trials_results,
            T_final=current_time,
            E_total=previous_iteration_energy,
            T_max=T_max
        )

        # Exit if no valid migrations remain
        if best_migration is None:
            energy_improved = False
            break

        # Apply selected migration
        sequence = construct_sequence(
            tasks,
            best_migration.task_index,
            best_migration.target_execution_unit,
            sequence
        )

        # Apply kernel algorithm for O(N) rescheduling
        kernel_algorithm(tasks, sequence)

        # Calculate new energy consumption
        current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power)
        energy_improved = current_iteration_energy < previous_iteration_energy

        # Manage cache size for memory efficiency
        if len(migration_cache) > 1000:
            migration_cache.clear()

    return tasks, sequence


def print_task_schedule(tasks):
    """
    Prints formatted task scheduling information.
    """
    # Pre-define constant mappings
    ASSIGNMENT_MAPPING = {
        0: "Core 1",
        1: "Core 2",
        2: "Core 3",
        3: "Cloud",
        -2: "Not Scheduled"
    }

    # Use list comprehension with conditional formatting
    schedule_data = []
    for task in tasks:
        base_info = {
            "Task ID": task.id,
            "Assignment": ASSIGNMENT_MAPPING.get(task.assignment, "Unknown")
        }

        if task.is_core_task:
            # Local core execution timing
            start_time = task.execution_unit_task_start_times[task.assignment]
            schedule_data.append({
                **base_info,
                "Execution Window": f"{start_time:.2f} → "
                                    f"{start_time + task.core_execution_times[task.assignment]:.2f}"
            })
        else:
            # Cloud execution phases timing
            send_start = task.execution_unit_task_start_times[3]
            send_end = send_start + task.cloud_execution_times[0]
            cloud_end = task.RT_c + task.cloud_execution_times[1]
            receive_end = task.RT_wr + task.cloud_execution_times[2]

            schedule_data.append({
                **base_info,
                "Send Phase": f"{send_start:.2f} → {send_end:.2f}",
                "Cloud Phase": f"{task.RT_c:.2f} → {cloud_end:.2f}",
                "Receive Phase": f"{task.RT_wr:.2f} → {receive_end:.2f}"
            })

    # Print formatted output
    print("\nTask Scheduling Details:")
    print("-" * 80)

    for entry in schedule_data:
        print("\n", end="")
        for key, value in entry.items():
            print(f"{key:15}: {value}")
        print("-" * 40)


def print_task_graph(tasks):
    """
    Prints the structure of the task graph.
    """
    for task in tasks:
        succ_task_ids = [child.id for child in task.succ_task]
        pred_task_ids = [pred_task.id for pred_task in task.pred_tasks]
        print(f"Task {task.id}:")
        print(f"  Parents: {pred_task_ids}")
        print(f"  Children: {succ_task_ids}")
        print()

def check_schedule_constraints(tasks, num_cores=3):
    """
    Validates schedule constraints considering cloud task pipelining

    Args:
        tasks: List of Task objects with scheduling information
        num_cores: Number of device cores

    Returns:
        tuple: (is_valid, violations)
    """
    violations = []

    def check_sending_channel():
        """Verify wireless sending channel is used sequentially"""
        cloud_tasks = [n for n in tasks if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.execution_unit_task_start_times[num_cores])

        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]

            if current.FT_ws > next_task.execution_unit_task_start_times[num_cores]:
                violations.append({
                    'type': 'Wireless Sending Channel Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} sending ends at {current.FT_ws} but Task {next_task.id} starts at {next_task.execution_unit_task_start_times[num_cores]}'
                })

    def check_computing_channel():
        """Verify cloud computing respects dependencies only (cloud has parallel processing)"""
        cloud_tasks = [n for n in tasks if not n.is_core_task]

        for task in cloud_tasks:
            # For cloud tasks, only check that predecessors' cloud computations finished
            cloud_predecessors = [p for p in task.pred_tasks if not p.is_core_task]

            for pred in cloud_predecessors:
                if pred.FT_c > task.RT_c:
                    violations.append({
                        'type': 'Cloud Dependency Violation',
                        'pred_task': pred.id,
                        'task': task.id,
                        'detail': f'Predecessor {pred.id} finishes cloud computing at {pred.FT_c} but Task {task.id} starts at {task.RT_c}'
                    })

    def check_receiving_channel():
        """Verify wireless receiving channel is sequential"""
        cloud_tasks = [n for n in tasks if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.RT_wr)

        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]

            if current.FT_wr > next_task.RT_wr:
                violations.append({
                    'type': 'Wireless Receiving Channel Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} receiving ends at {current.FT_wr} but Task {next_task.id} starts at {next_task.RT_wr}'
                })

    def check_pipelined_dependencies():
        """Verify dependencies considering pipelined execution"""
        for task in tasks:
            if not task.is_core_task:  # For cloud tasks
                # Check if all pred_tasks have completed necessary phases
                for pred_task in task.pred_tasks:
                    if pred_task.is_core_task:
                        # Core pred_task must complete before child starts sending
                        if pred_task.FT_l > task.execution_unit_task_start_times[num_cores]:
                            violations.append({
                                'type': 'Core-Cloud Dependency Violation',
                                'pred_task': pred_task.id,
                                'child': task.id,
                                'detail': f'Core Task {pred_task.id} finishes at {pred_task.FT_l} but Cloud Task {task.id} starts sending at {task.execution_unit_task_start_times[num_cores]}'
                            })
                    else:
                        # Cloud pred_task must complete sending before child starts sending
                        if pred_task.FT_ws > task.execution_unit_task_start_times[num_cores]:
                            violations.append({
                                'type': 'Cloud Pipeline Dependency Violation',
                                'pred_task': pred_task.id,
                                'child': task.id,
                                'detail': f'Parent Task {pred_task.id} sending phase ends at {pred_task.FT_ws} but Task {task.id} starts sending at {task.execution_unit_task_start_times[num_cores]}'
                            })
            else:  # For core tasks
                # All pred_tasks must complete fully before core task starts
                for pred_task in task.pred_tasks:
                    pred_task_finish = (pred_task.FT_wr
                                        if not pred_task.is_core_task else pred_task.FT_l)
                    if pred_task_finish > task.execution_unit_task_start_times[task.assignment]:
                        violations.append({
                            'type': 'Core Task Dependency Violation',
                            'pred_task': pred_task.id,
                            'child': task.id,
                            'detail': f'Parent Task {pred_task.id} finishes at {pred_task_finish} but Core Task {task.id} starts at {task.execution_unit_task_start_times[task.assignment]}'
                        })

    def check_core_execution():
        """Verify core tasks don't overlap"""
        core_tasks = [n for n in tasks if n.is_core_task]
        for core_id in range(num_cores):  # Dynamic core count
            core_specific_tasks = [t for t in core_tasks if t.assignment == core_id]
            sorted_tasks = sorted(core_specific_tasks, key=lambda x: x.execution_unit_task_start_times[core_id])

            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]

                if current.FT_l > next_task.execution_unit_task_start_times[core_id]:
                    violations.append({
                        'type': f'Core {core_id} Execution Conflict',
                        'task1': current.id,
                        'task2': next_task.id,
                        'detail': f'Task {current.id} finishes at {current.FT_l} but Task {next_task.id} starts at {next_task.execution_unit_task_start_times[core_id]}'
                    })

    # Run all checks
    check_sending_channel()
    check_computing_channel()
    check_receiving_channel()
    check_pipelined_dependencies()
    check_core_execution()

    return len(violations) == 0, violations


def analyze_task_placement(tasks):
    """
    Analyze task placement decisions in relation to task attributes.

    Args:
        tasks: List of enhanced Task objects with type, complexity, and data_intensity

    Returns:
        None (prints analysis to console)
    """
    # Define headers and formatting
    headers = ["ID", "Type", "Complexity", "Data Int.", "Placement", "Location", "Exec. Time"]
    format_str = "{:<4} {:<10} {:<10.2f} {:<10.2f} {:<12} {:<8} {:<10.2f}"

    # Print header row
    print("\nTASK ATTRIBUTE AND PLACEMENT ANALYSIS")
    print("=" * 80)
    print("{:<4} {:<10} {:<10} {:<10} {:<12} {:<8} {:<10}".format(*headers))
    print("-" * 80)

    # Track statistics for analysis
    compute_tasks_cloud = 0
    compute_tasks_local = 0
    data_tasks_cloud = 0
    data_tasks_local = 0
    balanced_tasks_cloud = 0
    balanced_tasks_local = 0

    # Expected vs actual placement
    expected_mismatches = 0

    # Process each task
    for task in sorted(tasks, key=lambda t: t.id):
        # Determine placement and location
        if task.is_core_task:
            placement = "Local Core"
            location = f"Core {task.assignment + 1}"
            exec_time = task.core_execution_times[task.assignment]

            # Update statistics
            if task.task_type == "compute":
                compute_tasks_local += 1
            elif task.task_type == "data":
                data_tasks_local += 1
            else:  # balanced
                balanced_tasks_local += 1
        else:
            placement = "Cloud"
            location = "Cloud"
            exec_time = sum(task.cloud_execution_times)

            # Update statistics
            if task.task_type == "compute":
                compute_tasks_cloud += 1
            elif task.task_type == "data":
                data_tasks_cloud += 1
            else:  # balanced
                balanced_tasks_cloud += 1

        # Determine if placement aligns with expectations
        expected_placement = ""
        if task.task_type == "compute" and not task.is_core_task:
            expected_mismatches += 1
            expected_placement = " (!)"
        elif task.task_type == "data" and task.is_core_task:
            expected_mismatches += 1
            expected_placement = " (!)"

        # Print task info
        print(format_str.format(
            task.id,
            task.task_type,
            task.complexity,
            task.data_intensity,
            placement + expected_placement,
            location,
            exec_time
        ))

    # Print summary statistics
    print("\nPLACEMENT SUMMARY:")
    print("-" * 80)
    print(f"Compute tasks: {compute_tasks_local} on local cores, {compute_tasks_cloud} on cloud")
    print(f"Data tasks: {data_tasks_local} on local cores, {data_tasks_cloud} on cloud")
    print(f"Balanced tasks: {balanced_tasks_local} on local cores, {balanced_tasks_cloud} on cloud")


def print_environment_resources(tasks):
    """
    Print detailed information about environment resources generated in data.py

    Args:
        tasks: List of Task objects (after enhance_tasks has been called)

    Returns:
        None (prints resource information to console)
    """

    print("\n" + "=" * 80)
    print("ENVIRONMENT RESOURCES GENERATED BY DATA.PY")
    print("=" * 80)

    # 1. TASK ATTRIBUTES
    print("\n1. TASK ATTRIBUTES")
    print("-" * 80)

    # Task type distribution
    task_types = [task.task_type for task in tasks]
    compute_count = task_types.count("compute")
    data_count = task_types.count("data")
    balanced_count = task_types.count("balanced")

    print(f"Task Type Distribution: {compute_count} compute, {data_count} data, {balanced_count} balanced")

    # Task complexity and data intensity stats
    complexity_values = [task.complexity for task in tasks]
    data_intensity_values = [task.data_intensity for task in tasks]

    print(f"Complexity Range: {min(complexity_values):.2f} to {max(complexity_values):.2f}")
    print(f"Complexity Average: {sum(complexity_values) / len(complexity_values):.2f}")
    print(f"Data Intensity Range: {min(data_intensity_values):.2f} to {max(data_intensity_values):.2f}")
    print(f"Data Intensity Average: {sum(data_intensity_values) / len(data_intensity_values):.2f}")

    # 2. DEVICE EXECUTION TIMES
    print("\n2. DEVICE EXECUTION TIMES")
    print("-" * 80)
    print("Core execution times for each task (in time units):")
    print(f"{'Task ID':<8} {'Task Type':<10} {'Complexity':<10} {'Core 1':<8} {'Core 2':<8} {'Core 3':<8}")
    print("-" * 62)

    for task in sorted(tasks, key=lambda t: t.id):
        if hasattr(task, 'core_execution_times') and len(task.core_execution_times) >= 3:
            print(f"{task.id:<8} {task.task_type:<10} {task.complexity:<10.2f} "
                  f"{task.core_execution_times[0]:<8.2f} {task.core_execution_times[1]:<8.2f} "
                  f"{task.core_execution_times[2]:<8.2f}")

    # Core speed ratios
    if all(hasattr(task, 'core_execution_times') for task in tasks):
        core_ratios = []
        for task in tasks:
            if len(task.core_execution_times) >= 3 and task.core_execution_times[2] > 0:
                core_ratio = task.core_execution_times[0] / task.core_execution_times[2]
                core_ratios.append(core_ratio)

        if core_ratios:
            avg_ratio = sum(core_ratios) / len(core_ratios)
            print(f"\nAverage Core 1 to Core 3 Speed Ratio: {avg_ratio:.2f}x")
            print(f"(Core 1 is approximately {avg_ratio:.2f} times slower than Core 3)")

    # 3. CLOUD EXECUTION TIMES
    print("\n3. CLOUD EXECUTION TIMES")
    print("-" * 80)
    print("Cloud phase times for each task (in time units):")
    print(f"{'Task ID':<8} {'Task Type':<10} {'Data Int.':<10} {'Send':<8} {'Compute':<8} {'Receive':<8} {'Total':<8}")
    print("-" * 70)

    for task in sorted(tasks, key=lambda t: t.id):
        if hasattr(task, 'cloud_execution_times') and len(task.cloud_execution_times) >= 3:
            send = task.cloud_execution_times[0]
            compute = task.cloud_execution_times[1]
            receive = task.cloud_execution_times[2]
            total = sum(task.cloud_execution_times)
            print(f"{task.id:<8} {task.task_type:<10} {task.data_intensity:<10.2f} "
                  f"{send:<8.2f} {compute:<8.2f} {receive:<8.2f} {total:<8.2f}")

    # 4. POWER MODELS
    print("\n4. POWER MODELS")
    print("-" * 80)
    device_type = 'mobile'
    battery_level = 75
    num_cores = 3

    # Direct call to generate_realistic_power_models instead of using initialize_power_models
    power_models = generate_realistic_power_models(
        device_type, battery_level, num_edge_nodes=0, num_cores=num_cores)

    # Extract core powers manually (same logic as in initialize_power_models)
    core_powers = [power_models['device'][i]['dynamic_power'](1.0)
                   for i in range(num_cores)]

    # Calculate cloud sending power manually
    cloud_sending_power = power_models['rf']['device_to_cloud'](5.0)

    print(f"Device Type: {device_type}")
    print(f"Battery Level: {battery_level}%")
    print(f"Number of Cores: {num_cores}")

    print(f"\nCore Power Consumption (at 100% load):")
    for i, power in enumerate(core_powers):
        print(f"Core {i + 1}: {power:.2f} power units")

    print(f"\nRF Sending Power: {cloud_sending_power:.2f} power units (at 5 Mbps)")

    # 5. NETWORK CONDITIONS
    print("\n5. NETWORK CONDITIONS")
    print("-" * 80)
    upload_rates, download_rates = generate_realistic_network_conditions()

    print("Data Transfer Rates (in Mbps):")
    print(f"{'Link':<20} {'Upload':<10} {'Download':<10}")
    print("-" * 40)

    # Display network conditions for device-cloud link
    print(
        f"{'Device to Cloud':<20} {upload_rates['device_to_cloud']:<10.2f} {download_rates['cloud_to_device']:<10.2f}")

    # Rest of the function remains the same...
    # 6. COMPARATIVE ANALYSIS
    print("\n6. COMPARATIVE ANALYSIS")
    print("-" * 80)

    # Compare cloud vs. local execution times
    cloud_times = []
    local_times = []

    for task in tasks:
        # Get fastest local core time
        if hasattr(task, 'core_execution_times') and len(task.core_execution_times) > 0:
            local_times.append(min(task.core_execution_times))

        # Get total cloud time
        if hasattr(task, 'cloud_execution_times') and len(task.cloud_execution_times) >= 3:
            cloud_times.append(sum(task.cloud_execution_times))

    if cloud_times and local_times:
        avg_cloud = sum(cloud_times) / len(cloud_times)
        avg_local = sum(local_times) / len(local_times)

        print(f"Average Fastest Local Execution Time: {avg_local:.2f} time units")
        print(f"Average Cloud Execution Time: {avg_cloud:.2f} time units")
        print(f"Cloud/Local Time Ratio: {avg_cloud / avg_local:.2f}x")

        # Type-specific comparisons
        compute_cloud = []
        compute_local = []
        data_cloud = []
        data_local = []

        for task in tasks:
            if task.task_type == "compute":
                if hasattr(task, 'cloud_execution_times'):
                    compute_cloud.append(sum(task.cloud_execution_times))
                if hasattr(task, 'core_execution_times'):
                    compute_local.append(min(task.core_execution_times))
            elif task.task_type == "data":
                if hasattr(task, 'cloud_execution_times'):
                    data_cloud.append(sum(task.cloud_execution_times))
                if hasattr(task, 'core_execution_times'):
                    data_local.append(min(task.core_execution_times))

        if compute_cloud and compute_local:
            avg_compute_cloud = sum(compute_cloud) / len(compute_cloud)
            avg_compute_local = sum(compute_local) / len(compute_local)
            print(f"\nCompute Tasks - Cloud/Local Time Ratio: {avg_compute_cloud / avg_compute_local:.2f}x")

        if data_cloud and data_local:
            avg_data_cloud = sum(data_cloud) / len(data_cloud)
            avg_data_local = sum(data_local) / len(data_local)
            print(f"Data Tasks - Cloud/Local Time Ratio: {avg_data_cloud / avg_data_local:.2f}x")

    # 7. ENERGY ANALYSIS
    print("\n7. ENERGY ANALYSIS")
    print("-" * 80)

    # Compare energy consumption for local vs cloud execution
    print("Estimated Energy Consumption:")
    print(f"{'Task Type':<10} {'Local Energy':<15} {'Cloud Energy':<15} {'Ratio Local/Cloud':<20}")
    print("-" * 60)

    # For compute tasks
    if compute_local and compute_cloud:
        # Average local computation energy
        compute_local_energy = min(core_powers) * avg_compute_local
        # Cloud sending energy (ignoring receiving energy)
        compute_cloud_energy = cloud_sending_power * sum(task.cloud_execution_times[0]
                                                         for task in tasks if task.task_type == "compute") / len(
            compute_cloud)
        ratio = compute_local_energy / compute_cloud_energy if compute_cloud_energy > 0 else float('inf')
        print(f"{'Compute':<10} {compute_local_energy:<15.2f} {compute_cloud_energy:<15.2f} {ratio:<20.2f}")

    # For data tasks
    if data_local and data_cloud:
        # Average local computation energy
        data_local_energy = min(core_powers) * avg_data_local
        # Cloud sending energy
        data_cloud_energy = cloud_sending_power * sum(task.cloud_execution_times[0]
                                                      for task in tasks if task.task_type == "data") / len(data_cloud)
        ratio = data_local_energy / data_cloud_energy if data_cloud_energy > 0 else float('inf')
        print(f"{'Data':<10} {data_local_energy:<15.2f} {data_cloud_energy:<15.2f} {ratio:<20.2f}")


def enhanced_mcc_scheduling():
    # Step 1: Create task graph
    task20 = Task(id=20, succ_task=[])
    task19 = Task(id=19, succ_task=[])
    task18 = Task(id=18, succ_task=[])
    task17 = Task(id=17, succ_task=[])
    task16 = Task(id=16, succ_task=[task19])
    task15 = Task(id=15, succ_task=[task19])
    task14 = Task(id=14, succ_task=[task18, task19])
    task13 = Task(id=13, succ_task=[task17, task18])
    task12 = Task(id=12, succ_task=[task17])
    task11 = Task(id=11, succ_task=[task15, task16])
    task10 = Task(id=10, succ_task=[task11, task15])
    task9 = Task(id=9, succ_task=[task13, task14])
    task8 = Task(id=8, succ_task=[task12, task13])
    task7 = Task(id=7, succ_task=[task12])
    task6 = Task(id=6, succ_task=[task10, task11])
    task5 = Task(id=5, succ_task=[task9, task10])
    task4 = Task(id=4, succ_task=[task8, task9])
    task3 = Task(id=3, succ_task=[task7, task8])
    task2 = Task(id=2, succ_task=[task7, task8])
    task1 = Task(id=1, succ_task=[task7])

    # Set predecessors
    task1.pred_tasks = []
    task2.pred_tasks = []
    task3.pred_tasks = []
    task4.pred_tasks = []
    task5.pred_tasks = []
    task6.pred_tasks = []
    task7.pred_tasks = [task1, task2, task3]
    task8.pred_tasks = [task3, task4]
    task9.pred_tasks = [task4, task5]
    task10.pred_tasks = [task5, task6]
    task11.pred_tasks = [task6, task10]
    task12.pred_tasks = [task7, task8]
    task13.pred_tasks = [task8, task9]
    task14.pred_tasks = [task9, task10]
    task15.pred_tasks = [task10, task11]
    task16.pred_tasks = [task11]
    task17.pred_tasks = [task12, task13]
    task18.pred_tasks = [task13, task14]
    task19.pred_tasks = [task14, task15, task16]
    task20.pred_tasks = [task12]

    tasks = [
        task1, task2, task3, task4, task5, task6, task7, task8,
        task9, task10, task11, task12, task13, task14, task15,
        task16, task17, task18, task19, task20
    ]

    # Step 2: Enhance tasks with attributes
    tasks = enhance_tasks(tasks)

    print_environment_resources(tasks)

    # Step 3: Initialize device execution time models
    tasks = initialize_device_times(tasks, num_device_cores=3)

    # Step 4: Apply network conditions
    tasks = apply_network_conditions(tasks)

    # Step 5: Initialize power models
    power_models, core_powers, cloud_sending_power = initialize_power_models()

    # Print task graph
    print_task_graph(tasks)

    # Step 6: Run initial scheduling
    primary_assignment(tasks)
    task_prioritizing(tasks)
    sequence = execution_unit_selection(tasks)

    # Step 7: Calculate initial metrics
    T_initial = total_time(tasks)
    E_initial = total_energy(tasks, core_powers, cloud_sending_power)
    print("INITIAL SCHEDULING APPLICATION COMPLETION TIME: ", T_initial)
    print("INITIAL APPLICATION ENERGY CONSUMPTION:", E_initial)
    print("INITIAL TASK SCHEDULE: ")

    print("\n----- INITIAL TASK PLACEMENT ANALYSIS -----")
    analyze_task_placement(tasks)

    print_task_schedule(tasks)

    # VALIDATION: Check if initial schedule is valid
    is_valid_initial, violations_initial = check_schedule_constraints(tasks)
    print("\nINITIAL SCHEDULE VALIDATION:")
    if is_valid_initial:
        print("✓ Initial schedule is valid - All constraints satisfied")
    else:
        print("✗ Initial schedule has constraint violations:")
        for i, violation in enumerate(violations_initial):
            print(f"  Violation {i + 1}: {violation['type']}")
            print(f"    {violation['detail']}")

    # Step 8: Run task migration for energy optimization
    tasks, sequence = optimize_task_scheduling(tasks, sequence, T_initial, core_powers, cloud_sending_power)

    # Step 9: Calculate final metrics
    T_final = total_time(tasks)
    E_final = total_energy(tasks, core_powers, cloud_sending_power)
    print("\nFINAL SCHEDULING APPLICATION COMPLETION TIME: ", T_final)
    print("FINAL APPLICATION ENERGY CONSUMPTION:", E_final)

    print("\n----- FINAL TASK PLACEMENT ANALYSIS -----")
    analyze_task_placement(tasks)

    print("ENERGY REDUCTION FACTOR:", E_initial / E_final)

    # VALIDATION: Check if optimized schedule is valid
    is_valid_final, violations_final = check_schedule_constraints(tasks)
    print("\nFINAL SCHEDULE VALIDATION:")
    if is_valid_final:
        print("✓ Final schedule is valid - All constraints satisfied")
    else:
        print("✗ Final schedule has constraint violations:")
        for i, violation in enumerate(violations_final):
            print(f"  Violation {i + 1}: {violation['type']}")
            print(f"    {violation['detail']}")

    return tasks, sequence, (is_valid_initial, violations_initial), (is_valid_final, violations_final)


if __name__ == '__main__':
    tasks, sequence, initial_validation, final_validation = enhanced_mcc_scheduling()

    print("\nVALIDATION SUMMARY:")
    print(
        f"Initial Schedule: {'Valid' if initial_validation[0] else 'Invalid'} ({len(initial_validation[1])} violations)")
    print(f"Final Schedule: {'Valid' if final_validation[0] else 'Invalid'} ({len(final_validation[1])} violations)")
