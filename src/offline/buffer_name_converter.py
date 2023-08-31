from pathlib import Path
import sys

PROJ_DIR = Path.cwd().parent.parent
sys.path.append(str(PROJ_DIR))

import h5py
import numpy as np
np.set_printoptions(precision=3, linewidth=120)
import torch

from src.defaults import ROOT_DIR
from src.checkpoint import CheckpointableData, Checkpointer
from src.config import BaseConfig, Require
from src.torch_util import device, Module, random_indices, torchify
from src.smbpo import SMBPO
from src.sampling import SampleBuffer, ConstraintSafetySampleBuffer


# load basic config: path_to_buffer
path_to_buffers = Path(ROOT_DIR) / "logs" / "point-robot" / \
    "08-29-23_14.13.59_DRPO_49283" / "test-2023-08-31-20-22-50-offline" / \
    "point-robot-expert-50k.h5py"

# define a new class has same components' name with d4rl dataset
class SafeBuffer_D4RL(Module):
    COMPONENT_NAMES = ('state', 'action', 'next_state', 'reward', 'done', 'cost', 'h')
    
    def __init__(
        self,
        state_dim,
        action_dim,
        capacity,
        discrete_actions=False,
        device="cpu"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.discrete_actions = discrete_actions
        self.device = device

        self._bufs = {}
        self.register_buffer('_pointer', torch.tensor(0, dtype=torch.long))

        if discrete_actions:
            assert action_dim == 1
            action_dtype = torch.int
            action_shape = []
        else:
            action_dtype = torch.float
            action_shape = [action_dim]

        components = (
            ('state', torch.float, [state_dim]),
            ('action', action_dtype, action_shape),
            ('next_state', torch.float, [state_dim]),
            ('reward', torch.float, []),
            ('done', torch.bool, []),
            ('cost', torch.float, []),
            ('h', torch.float, []),
        )
        for name, dtype, shape in components:
            self._create_buffer(name, dtype, shape)
    
    @classmethod
    def from_state_dict(cls, state_dict, device=device):
        # Must have same keys
        assert set(state_dict.keys()) == {*(f'_{name}' for name in cls.COMPONENT_NAMES), '_pointer'}
        states, actions = state_dict['_states'], state_dict['_actions']

        # Check that length (size of first dimension) matches
        l = len(states)
        for name in cls.COMPONENT_NAMES:
            tensor = state_dict[f'_{name}']
            assert torch.is_tensor(tensor)
            assert len(tensor) == l

        # Capacity, dimensions, and type of action inferred from state_dict
        buffer = cls(state_dim=states.shape[1], action_dim=actions.shape[1], capacity=l,
                     discrete_actions=(not actions.dtype.is_floating_point),
                     device=device)
        buffer.load_state_dict(state_dict)
        return buffer

    @classmethod
    def from_h5py(cls, path, device="cpu"):
        with h5py.File(path, 'r') as f:
            data = {name: torchify(np.array(f[name]), to_device=False) for name in f.keys()}
        n_steps = len(data['reward'])
        if 'next_state' not in data:
            all_states = data['state']
            assert len(all_states) == n_steps + 1
            data['state'] = all_states[:-1]
            data['next_state'] = all_states[1:]
        for v in data.values():
            assert len(v) == n_steps

        # Capacity, dimensions, and type of action inferred from h5py file
        states, actions = data['state'], data['action']
        buffer = cls(state_dim=states.shape[1], action_dim=actions.shape[1], capacity=n_steps,
                     discrete_actions=(not actions.dtype.is_floating_point),
                     device=device)
        buffer.extend(**data)
        return buffer

    def __len__(self):
        return min(self._pointer, self.capacity)

    def _create_buffer(self, name, dtype, shape):
        assert name not in self._bufs
        _name = f'_{name}'
        buffer_shape = [self.capacity, *shape]
        buffer = torch.empty(*buffer_shape, dtype=dtype, device=self.device)
        self.register_buffer(_name, buffer)
        self._bufs[name] = buffer

    def _get1(self, name):
        buf = self._bufs[name]
        if self._pointer <= self.capacity:
            return buf[:self._pointer]
        else:
            i = self._pointer % self.capacity
            return torch.cat([buf[i:], buf[:i]])

    def get(self, *names, device=device, as_dict=False):
        """
        Retrieves data from the buffer. Pass a vararg list of names.
        What is returned depends on how many names are given:
            * a list of all components if no names are given
            * a single component if one name is given
            * a list with one component for each name otherwise
        """
        if len(names) == 0:
            names = self.COMPONENT_NAMES
        bufs = [self._get1(name).to(device) for name in names]
        if as_dict:
            return dict(zip(names, bufs))
        else:
            return bufs if len(bufs) > 1 else bufs[0]

    def append(self, **kwargs):
        assert set(kwargs.keys()) == set(self.COMPONENT_NAMES)
        i = self._pointer % self.capacity
        for name in self.COMPONENT_NAMES:
            self._bufs[name][i] = kwargs[name]
        self._pointer += 1

    def extend(self, **kwargs):
        assert set(kwargs.keys()) == set(self.COMPONENT_NAMES)
        batch_size = len(list(kwargs.values())[0])
        assert batch_size <= self.capacity, 'We do not support extending by more than buffer capacity'
        i = self._pointer % self.capacity
        end = i + batch_size
        if end <= self.capacity:
            for name in self.COMPONENT_NAMES:
                self._bufs[name][i:end] = kwargs[name]
        else:
            fit = self.capacity - i
            overflow = end - self.capacity
            # Note: fit + overflow = batch_size
            for name in self.COMPONENT_NAMES:
                buf, arg = self._bufs[name], kwargs[name]
                buf[-fit:] = arg[:fit]
                buf[:overflow] = arg[-overflow:]
        self._pointer += batch_size

    def sample(self, batch_size, replace=True, device=device, include_indices=False):
        indices = torch.randint(len(self), [batch_size], device=device) if replace else \
            random_indices(len(self), size=batch_size, replace=False)
        bufs = [self._bufs[name][indices].to(device) for name in self.COMPONENT_NAMES]
        return (bufs, indices) if include_indices else bufs

    def split_episodes(self, max_length):
        """Use to split a single buffer into episodes that make it up.
        Note: this method computes the episode structure assuming the samples in the dataset are ordered sequentially.
        If this is not the case, the returned "episodes" are meaningless."""
        assert self._pointer <= self.capacity, 'split_episodes will give bad results on a circular buffer'
        states, actions, next_states, rewards, dones = self.get()
        n = len(self)
        done_indices = list(map(int, dones.nonzero()))
        episodes = []
        offset = 0
        used_indices = set()
        while offset < n:
            max_end = min(offset + max_length, n)
            actual_end = max_end
            if len(done_indices) > 0:
                next_done_index = done_indices[0]
                if next_done_index <= max_end:
                    actual_end = next_done_index + 1
                    done_indices.pop(0)

            episode_indices = set(range(offset, actual_end))
            assert len(episode_indices) > 0, 'Cannot have empty episode!'
            assert len(used_indices & episode_indices) == 0, 'Indices should not overlap!'
            traj_buffer = SampleBuffer(self.state_dim, self.action_dim, len(episode_indices),
                                       discrete_actions=self.discrete_actions)
            traj_buffer.extend(
                states[offset:actual_end],
                actions[offset:actual_end],
                next_states[offset:actual_end],
                rewards[offset:actual_end],
                dones[offset:actual_end]
            )
            episodes.append(traj_buffer)

            offset = actual_end
            used_indices |= episode_indices

        # Sanity checks
        assert len(done_indices) == 0
        assert sum(len(traj) for traj in episodes) == n
        assert used_indices == set(range(n))
        return episodes

    def trimmed_copy(self):
        new_buffer = self.__class__(self.state_dim, self.action_dim, len(self),
                                    discrete_actions=self.discrete_actions)
        new_buffer.extend(*self.get())
        return new_buffer

    def save_h5py(self, path, remove_duplicate_states=False):
        data = self.get(as_dict=True, device='cpu')
        if remove_duplicate_states:
            next_states = data.pop('next_state')
            data['state'] = torch.cat((
                data['state'], next_states[-1].unsqueeze(0)
            ))

        with h5py.File(path, 'w') as f:
            for k, v in data.items():
                f.create_dataset(k, data=v.numpy())


# main function
if __name__ == '__main__':
    ## 1: load old buffer
    old_buffer = ConstraintSafetySampleBuffer.from_h5py(path_to_buffers)
    data = old_buffer.get(device='cpu', as_dict=True)
    new_data = {}
    new_data["state"] = data["states"]
    new_data["action"] = data["actions"]
    new_data["next_state"] = data["next_states"]
    new_data["reward"] = data["rewards"]
    new_data["done"] = data["dones"]
    ### 2.1 mild convertion on cost (from bool to float {0, 1})
    new_data["cost"] = data["violations"].float()
    ### 2.2 mild convertion on h (from size(4,) to size()))
    new_data["h"] = data["constraint_values"]
    print(new_data["h"].shape)
    
    ## 2: create new buffer with different names
    new_buffer = SafeBuffer_D4RL(
        state_dim=old_buffer.state_dim,
        action_dim=old_buffer.action_dim,
        capacity=old_buffer.capacity,
        discrete_actions=old_buffer.discrete_actions,
        device=old_buffer.device
    )
    
    new_buffer.extend(**new_data)
    
    ## 3: save new buffer
    new_buffer.save_h5py(path_to_buffers.parent / "point-robot-expert-normal-50k.h5py")