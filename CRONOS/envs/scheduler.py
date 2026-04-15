import random
import re

class TaskScheduler:
    """Manages task sequencing for multi-task robotic manipulation."""
    
    def __init__(self, task_pool, mode="sequential", num_envs=64):
        self.task_pool = task_pool  # List of task strings
        self.mode = mode
        self.num_envs = num_envs
        self.current_task_idx = 0
        
    @staticmethod
    def _extract_obj_recep(task_str):
        """Extracts object and receptacle from a task string."""
        pattern = r"put (.*?) on (.*)"
        match = re.search(pattern, task_str)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def get_next_tasks(self, num_groups=None):
        """Returns the next batch of (objects, receptacles) for all environments."""
        objects, receptacles = [], []
        if num_groups is None:
            num_groups = len(self.task_pool)
        
        num_groups = min(num_groups, self.num_envs)
        group_size = self.num_envs // num_groups
        
        if self.mode == "sequential":
            for i in range(num_groups):
                task = self.task_pool[(self.current_task_idx + i) % len(self.task_pool)]
                obj, recep = self._extract_obj_recep(task)
                objects.extend([obj] * group_size)
                receptacles.extend([recep] * group_size)
            # Handle remainder envs if any
            remainder = self.num_envs % num_groups
            if remainder > 0:
                objects.extend([objects[-1]] * remainder)
                receptacles.extend([receptacles[-1]] * remainder)
                
        elif self.mode == "random":
            for _ in range(self.num_envs):
                rand_idx = random.randint(0, min(num_groups - 1, len(self.task_pool) - 1))
                task = self.task_pool[(self.current_task_idx + rand_idx) % len(self.task_pool)]
                obj, recep = self._extract_obj_recep(task)
                objects.append(obj)
                receptacles.append(recep)
                
        return objects, receptacles

    def update_index(self):
        """Moves to the next task sequence."""
        self.current_task_idx = (self.current_task_idx + 1) % len(self.task_pool)
