"""Pinned-CPU optimizer-state parking for one-GPU Muon training."""

import time


class OptimizerStateOffloader:
    """Keep optimizer state in reusable CPU buffers outside ``step()``.

    Parameters and gradients stay on their original device. Immediately before
    an optimizer step, every tensor in ``optimizer.state`` is restored to its
    owning parameter's device. After the update, the tensors are copied back to
    persistent host buffers and the CUDA copies are released. Optimizer
    checkpoints therefore naturally serialize CPU tensors with unchanged keys,
    shapes, dtypes, and values.

    This manager is deliberately scoped to ordinary (non-DTensor) one-device
    optimizers. Distributed state needs a sharding-aware implementation rather
    than silently materializing full tensors on every rank.
    """

    def __init__(
        self,
        torch_module,
        optimizer,
        *,
        pin_memory=True,
        require_cuda=True,
        optimizer_label="optimizer",
        pin_memory_option="--no-optimizer-state-offload-pin-memory",
    ):
        self.torch = torch_module
        self.optimizer = optimizer
        self.pin_memory = bool(pin_memory)
        self.require_cuda = bool(require_cuda)
        self.optimizer_label = str(optimizer_label)
        self.pin_memory_option = str(pin_memory_option)
        self.host_buffers = {}
        self.restore_devices = {}
        self.park_calls = 0
        self.restore_calls = 0
        self.park_seconds = 0.0
        self.restore_seconds = 0.0
        self.optimizer_step_calls = 0
        self.optimizer_step_seconds = 0.0
        self.device_to_host_bytes = 0
        self.host_to_device_bytes = 0

    @staticmethod
    def _tensor_bytes(tensor):
        return int(tensor.numel()) * int(tensor.element_size())

    def _parameter_device(self, parameter):
        if hasattr(parameter, "to_local"):
            raise RuntimeError(
                f"{self.optimizer_label} state offload currently supports "
                "ordinary one-GPU parameters only; DTensor/FSDP2 state "
                "requires sharding-aware "
                "offload"
            )
        return parameter.device

    def _new_host_buffer(self, value):
        try:
            return self.torch.empty_like(
                value,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        except RuntimeError as error:
            if self.pin_memory:
                raise RuntimeError(
                    "Could not allocate pinned CPU memory for "
                    f"{self.optimizer_label} state. Use {self.pin_memory_option} "
                    "to allow "
                    "slower pageable CPU buffers."
                ) from error
            raise

    def _host_buffer(self, key, value):
        buffer = self.host_buffers.get(key)
        if (
            buffer is None
            or tuple(buffer.shape) != tuple(value.shape)
            or buffer.dtype != value.dtype
            or buffer.layout != value.layout
        ):
            buffer = self._new_host_buffer(value)
            self.host_buffers[key] = buffer
        return buffer

    def park(self):
        """Copy live state to host buffers and release its device tensors."""
        started_at = time.perf_counter()
        copied_bytes = 0
        cuda_devices = set()
        live_keys = set()

        for parameter, state in self.optimizer.state.items():
            self._parameter_device(parameter)
            for name, value in list(state.items()):
                if not self.torch.is_tensor(value):
                    continue
                key = (parameter, name)
                live_keys.add(key)
                buffer = self._host_buffer(key, value)
                if value is buffer:
                    continue
                if value.device.type not in ("cpu", "cuda"):
                    raise RuntimeError(
                        f"{self.optimizer_label} state offload supports "
                        "CPU/CUDA tensors only; "
                        f"state {name!r} is on {value.device}"
                    )
                # Some torch optimizers deliberately keep scalar bookkeeping
                # (notably AdamW's step counter) on CPU even for CUDA
                # parameters. Restore every entry to the device chosen by the
                # child optimizer instead of blindly moving all state to the
                # owning parameter's device.
                self.restore_devices[key] = value.device
                if value.device.type == "cuda":
                    cuda_devices.add(value.device)
                    copied_bytes += self._tensor_bytes(value)
                buffer.copy_(
                    value,
                    non_blocking=(self.pin_memory and value.device.type == "cuda"),
                )
                state[name] = buffer

        for device in cuda_devices:
            self.torch.cuda.synchronize(device)
        stale_keys = set(self.host_buffers).difference(live_keys)
        for key in stale_keys:
            del self.host_buffers[key]
            self.restore_devices.pop(key, None)

        self.park_calls += 1
        self.park_seconds += time.perf_counter() - started_at
        self.device_to_host_bytes += copied_bytes
        return copied_bytes

    def restore(self):
        """Restore parked state to each owning parameter's device."""
        started_at = time.perf_counter()
        copied_bytes = 0
        cuda_devices = set()

        for parameter, state in self.optimizer.state.items():
            parameter_device = self._parameter_device(parameter)
            if self.require_cuda and parameter_device.type != "cuda":
                raise RuntimeError(
                    f"Production {self.optimizer_label} state offload requires "
                    "CUDA model "
                    f"parameters; got {parameter_device}"
                )
            for name, value in list(state.items()):
                if not self.torch.is_tensor(value):
                    continue
                key = (parameter, name)
                buffer = self.host_buffers.get(key)
                if buffer is None or value is not buffer:
                    raise RuntimeError(
                        f"{self.optimizer_label} state was modified while parked: "
                        f"missing host buffer for {name!r}"
                    )
                target_device = self.restore_devices.get(key)
                if target_device is None:
                    raise RuntimeError(
                        f"{self.optimizer_label} state was modified while parked: "
                        f"missing restore device for {name!r}"
                    )
                restored = self.torch.empty_like(value, device=target_device)
                restored.copy_(
                    value,
                    non_blocking=(self.pin_memory and target_device.type == "cuda"),
                )
                state[name] = restored
                copied_bytes += self._tensor_bytes(value)
                if target_device.type == "cuda":
                    cuda_devices.add(target_device)

        for device in cuda_devices:
            self.torch.cuda.synchronize(device)
        self.restore_calls += 1
        self.restore_seconds += time.perf_counter() - started_at
        self.host_to_device_bytes += copied_bytes
        return copied_bytes

    def is_parked(self):
        for parameter, state in self.optimizer.state.items():
            for name, value in state.items():
                if not self.torch.is_tensor(value):
                    continue
                if value is not self.host_buffers.get((parameter, name)):
                    return False
        return True

    def stats(self):
        return {
            "enabled": True,
            "optimizer_label": self.optimizer_label,
            "pin_memory": self.pin_memory,
            "state_tensors": len(self.host_buffers),
            "state_bytes": sum(
                self._tensor_bytes(buffer) for buffer in self.host_buffers.values()
            ),
            "park_calls": self.park_calls,
            "restore_calls": self.restore_calls,
            "park_seconds": self.park_seconds,
            "restore_seconds": self.restore_seconds,
            "optimizer_step_calls": self.optimizer_step_calls,
            "optimizer_step_seconds": self.optimizer_step_seconds,
            "device_to_host_bytes": self.device_to_host_bytes,
            "host_to_device_bytes": self.host_to_device_bytes,
        }


def make_state_offloaded_optimizer_class(
    torch_module,
    optimizer_class,
    *,
    pin_memory=True,
    require_cuda=True,
    manager_attribute="_arc_optimizer_state_offloader",
    optimizer_label="optimizer",
    pin_memory_option="--no-optimizer-state-offload-pin-memory",
):
    """Return an optimizer subclass whose state is parked around every step."""

    class StateOffloadedOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            manager = OptimizerStateOffloader(
                torch_module,
                self,
                pin_memory=pin_memory,
                require_cuda=require_cuda,
                optimizer_label=optimizer_label,
                pin_memory_option=pin_memory_option,
            )
            setattr(self, manager_attribute, manager)
            manager.park()

        def step(self, closure=None):
            manager = getattr(self, manager_attribute)
            try:
                manager.restore()
                optimizer_started_at = time.perf_counter()
                try:
                    return super().step(closure=closure)
                finally:
                    manager.optimizer_step_calls += 1
                    manager.optimizer_step_seconds += (
                        time.perf_counter() - optimizer_started_at
                    )
            finally:
                # A failed update must not leave the full optimizer state
                # resident during the next attempted forward/backward pass.
                # Keep this guard around restore too: a partial H2D restore can
                # itself OOM after replacing only some state entries.
                manager.park()

        def state_dict(self):
            manager = getattr(self, manager_attribute)
            if not manager.is_parked():
                manager.park()
            return super().state_dict()

        def load_state_dict(self, state_dict):
            result = super().load_state_dict(state_dict)
            # torch.optim restores state to the parameter device. Reuse the
            # existing host buffers and immediately re-park it before the next
            # forward pass.
            getattr(self, manager_attribute).park()
            return result

    StateOffloadedOptimizer.__name__ = f"StateOffloaded{optimizer_class.__name__}"
    StateOffloadedOptimizer.__qualname__ = StateOffloadedOptimizer.__name__
    return StateOffloadedOptimizer


def make_state_offloaded_normuon_class(
    torch_module,
    normuon_class,
    *,
    pin_memory=True,
    require_cuda=True,
):
    """Return a NorMuon subclass whose state is parked around every step."""
    return make_state_offloaded_optimizer_class(
        torch_module,
        normuon_class,
        pin_memory=pin_memory,
        require_cuda=require_cuda,
        manager_attribute="_arc_normuon_state_offloader",
        optimizer_label="NorMuon optimizer",
        pin_memory_option="--no-normuon-state-offload-pin-memory",
    )
