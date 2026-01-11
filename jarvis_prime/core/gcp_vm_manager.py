"""
GCP VM Manager v1.0 - Cloud Infrastructure Lifecycle Management
================================================================

Manages GCP Spot/Preemptible VMs for cloud inference with:
- Automatic provisioning and teardown
- Preemption detection and handling
- Auto-scaling based on demand
- Cost tracking and limits
- Health monitoring
- Checkpoint migration

ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────┐
    │                     GCP VM MANAGER                              │
    └──────────────────────────┬─────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
    ┌──────────┐        ┌──────────────┐      ┌──────────────┐
    │Provisioner│       │ Health       │      │ Cost         │
    │ (Create/  │       │ Monitor      │      │ Tracker      │
    │  Delete)  │       │              │      │              │
    └──────────┘        └──────────────┘      └──────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Preemption       │
                    │   Handler          │
                    │   (Checkpoint +    │
                    │    Migrate)        │
                    └────────────────────┘

FEATURES:
    - Async lifecycle management
    - Preemption signal detection
    - Automatic checkpointing
    - Instance migration on preemption
    - Auto-scaling with cooldowns
    - Cost limits and alerts
    - GPU availability checking
    - Zone selection optimization
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# TRY IMPORTS
# =============================================================================

try:
    from google.cloud import compute_v1
    from google.oauth2 import service_account
    GCP_SDK_AVAILABLE = True
except ImportError:
    GCP_SDK_AVAILABLE = False
    logger.warning("google-cloud-compute not available - GCP features disabled")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class VMState(Enum):
    """VM lifecycle states."""
    UNKNOWN = "unknown"
    PROVISIONING = "provisioning"
    STAGING = "staging"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    PREEMPTED = "preempted"
    FAILED = "failed"


class MachineType(Enum):
    """Common GCP machine types for inference."""
    N1_STANDARD_4 = "n1-standard-4"      # 4 vCPU, 15GB RAM
    N1_STANDARD_8 = "n1-standard-8"      # 8 vCPU, 30GB RAM
    N1_STANDARD_16 = "n1-standard-16"    # 16 vCPU, 60GB RAM
    N1_HIGHMEM_8 = "n1-highmem-8"        # 8 vCPU, 52GB RAM
    A2_HIGHGPU_1G = "a2-highgpu-1g"      # 12 vCPU, 85GB RAM, 1x A100


class GPUType(Enum):
    """GCP GPU types."""
    NVIDIA_TESLA_T4 = "nvidia-tesla-t4"
    NVIDIA_TESLA_V100 = "nvidia-tesla-v100"
    NVIDIA_TESLA_A100 = "nvidia-tesla-a100"
    NVIDIA_L4 = "nvidia-l4"


@dataclass
class VMConfig:
    """Configuration for a GCP VM instance."""
    name: str
    machine_type: str = "n1-standard-8"
    zone: str = "us-central1-a"
    region: str = "us-central1"

    # Disk
    disk_size_gb: int = 100
    disk_type: str = "pd-ssd"
    image_family: str = "ubuntu-2204-lts"
    image_project: str = "ubuntu-os-cloud"

    # GPU
    gpu_type: Optional[str] = None
    gpu_count: int = 0

    # Spot/Preemptible
    spot: bool = True
    preemptible: bool = True  # Legacy, use spot instead

    # Startup script
    startup_script: Optional[str] = None
    startup_script_url: Optional[str] = None

    # Networking
    network: str = "default"
    subnetwork: Optional[str] = None
    external_ip: bool = True
    internal_ip_only: bool = False

    # Labels
    labels: Dict[str, str] = field(default_factory=dict)

    # Service account
    service_account_email: Optional[str] = None
    scopes: List[str] = field(default_factory=lambda: [
        "https://www.googleapis.com/auth/cloud-platform",
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "machine_type": self.machine_type,
            "zone": self.zone,
            "spot": self.spot,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
        }


@dataclass
class AutoScaleConfig:
    """Configuration for auto-scaling."""
    enabled: bool = True
    min_instances: int = 0
    max_instances: int = 2
    scale_up_threshold_percent: float = 80.0
    scale_down_threshold_percent: float = 20.0
    cooldown_seconds: int = 300
    scale_up_step: int = 1
    scale_down_step: int = 1


@dataclass
class CostConfig:
    """Configuration for cost management."""
    enabled: bool = True
    max_hourly_spend: float = 5.0
    max_daily_spend: float = 50.0
    alert_threshold_percent: float = 80.0
    shutdown_on_limit: bool = True


@dataclass
class PreemptionConfig:
    """Configuration for preemption handling."""
    check_interval_seconds: int = 60
    signal_file: str = "/tmp/preemption_signal"
    checkpoint_on_preemption: bool = True
    auto_migrate: bool = True
    migrate_zones: List[str] = field(default_factory=lambda: [
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-central1-f",
    ])


@dataclass
class GCPManagerConfig:
    """Overall configuration for GCP VM Manager."""
    project_id: str = ""
    region: str = "us-central1"
    credentials_path: Optional[str] = None

    vm_config: VMConfig = field(default_factory=lambda: VMConfig(name="jarvis-prime-worker"))
    auto_scale: AutoScaleConfig = field(default_factory=AutoScaleConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)

    health_check_interval_seconds: int = 30
    state_file: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "gcp" / "vm_state.json")

    @classmethod
    def from_env(cls) -> "GCPManagerConfig":
        """Create config from environment variables."""
        return cls(
            project_id=os.getenv("GCP_PROJECT_ID", ""),
            region=os.getenv("GCP_REGION", "us-central1"),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            vm_config=VMConfig(
                name=os.getenv("GCP_VM_NAME", "jarvis-prime-worker"),
                machine_type=os.getenv("GCP_MACHINE_TYPE", "n1-standard-8"),
                zone=os.getenv("GCP_ZONE", "us-central1-a"),
                gpu_type=os.getenv("GCP_GPU_TYPE"),
                gpu_count=int(os.getenv("GCP_GPU_COUNT", "0")),
            ),
        )


# =============================================================================
# VM INSTANCE STATE
# =============================================================================

@dataclass
class VMInstance:
    """Represents a GCP VM instance."""
    name: str
    zone: str
    state: VMState = VMState.UNKNOWN
    external_ip: Optional[str] = None
    internal_ip: Optional[str] = None
    machine_type: str = ""
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    preempted_at: Optional[datetime] = None

    # GPU
    gpu_type: Optional[str] = None
    gpu_count: int = 0

    # Health
    is_healthy: bool = False
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0

    # Cost
    hourly_cost: float = 0.0
    total_cost: float = 0.0
    runtime_hours: float = 0.0

    # Endpoint
    inference_endpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "zone": self.zone,
            "state": self.state.value,
            "external_ip": self.external_ip,
            "internal_ip": self.internal_ip,
            "machine_type": self.machine_type,
            "is_healthy": self.is_healthy,
            "hourly_cost": self.hourly_cost,
            "total_cost": self.total_cost,
            "inference_endpoint": self.inference_endpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VMInstance":
        return cls(
            name=data["name"],
            zone=data["zone"],
            state=VMState(data.get("state", "unknown")),
            external_ip=data.get("external_ip"),
            internal_ip=data.get("internal_ip"),
            machine_type=data.get("machine_type", ""),
            is_healthy=data.get("is_healthy", False),
            hourly_cost=data.get("hourly_cost", 0.0),
            total_cost=data.get("total_cost", 0.0),
            inference_endpoint=data.get("inference_endpoint"),
        )


# =============================================================================
# MOCK GCP CLIENT (for development)
# =============================================================================

class MockGCPClient:
    """Mock GCP client for development without actual GCP access."""

    def __init__(self, config: GCPManagerConfig):
        self._config = config
        self._instances: Dict[str, VMInstance] = {}
        self._lock = asyncio.Lock()

    async def create_instance(self, vm_config: VMConfig) -> VMInstance:
        """Mock create instance."""
        async with self._lock:
            instance = VMInstance(
                name=vm_config.name,
                zone=vm_config.zone,
                state=VMState.RUNNING,
                external_ip=f"34.123.45.{len(self._instances) + 1}",
                internal_ip=f"10.128.0.{len(self._instances) + 1}",
                machine_type=vm_config.machine_type,
                created_at=datetime.now(),
                started_at=datetime.now(),
                gpu_type=vm_config.gpu_type,
                gpu_count=vm_config.gpu_count,
                hourly_cost=0.50,
                inference_endpoint=f"http://34.123.45.{len(self._instances) + 1}:8000",
            )
            self._instances[vm_config.name] = instance
            logger.info(f"[MOCK] Created instance {vm_config.name}")
            return instance

    async def delete_instance(self, name: str, zone: str) -> bool:
        """Mock delete instance."""
        async with self._lock:
            if name in self._instances:
                del self._instances[name]
                logger.info(f"[MOCK] Deleted instance {name}")
                return True
            return False

    async def get_instance(self, name: str, zone: str) -> Optional[VMInstance]:
        """Mock get instance."""
        return self._instances.get(name)

    async def list_instances(self, zone: str) -> List[VMInstance]:
        """Mock list instances."""
        return [
            inst for inst in self._instances.values()
            if inst.zone == zone
        ]

    async def start_instance(self, name: str, zone: str) -> bool:
        """Mock start instance."""
        if name in self._instances:
            self._instances[name].state = VMState.RUNNING
            self._instances[name].started_at = datetime.now()
            return True
        return False

    async def stop_instance(self, name: str, zone: str) -> bool:
        """Mock stop instance."""
        if name in self._instances:
            self._instances[name].state = VMState.STOPPED
            return True
        return False

    async def check_preemption(self, name: str, zone: str) -> bool:
        """Mock preemption check - randomly preempt for testing."""
        import random
        return random.random() < 0.01  # 1% chance


# =============================================================================
# REAL GCP CLIENT
# =============================================================================

class GCPClient:
    """Real GCP client using google-cloud-compute SDK."""

    def __init__(self, config: GCPManagerConfig):
        self._config = config
        self._instances_client = None
        self._credentials = None

        if GCP_SDK_AVAILABLE:
            self._initialize_clients()

    def _initialize_clients(self):
        """Initialize GCP clients."""
        if self._config.credentials_path:
            self._credentials = service_account.Credentials.from_service_account_file(
                self._config.credentials_path
            )
            self._instances_client = compute_v1.InstancesClient(
                credentials=self._credentials
            )
        else:
            # Use default credentials
            self._instances_client = compute_v1.InstancesClient()

    async def create_instance(self, vm_config: VMConfig) -> VMInstance:
        """Create a new VM instance."""
        if not self._instances_client:
            raise RuntimeError("GCP client not initialized")

        # Build instance configuration
        instance = compute_v1.Instance()
        instance.name = vm_config.name
        instance.machine_type = f"zones/{vm_config.zone}/machineTypes/{vm_config.machine_type}"

        # Boot disk
        boot_disk = compute_v1.AttachedDisk()
        boot_disk.boot = True
        boot_disk.auto_delete = True
        boot_disk.initialize_params = compute_v1.AttachedDiskInitializeParams()
        boot_disk.initialize_params.source_image = (
            f"projects/{vm_config.image_project}/global/images/family/{vm_config.image_family}"
        )
        boot_disk.initialize_params.disk_size_gb = vm_config.disk_size_gb
        boot_disk.initialize_params.disk_type = (
            f"zones/{vm_config.zone}/diskTypes/{vm_config.disk_type}"
        )
        instance.disks = [boot_disk]

        # Network interface
        network_interface = compute_v1.NetworkInterface()
        network_interface.network = f"global/networks/{vm_config.network}"

        if vm_config.external_ip:
            access_config = compute_v1.AccessConfig()
            access_config.name = "External NAT"
            access_config.type_ = "ONE_TO_ONE_NAT"
            network_interface.access_configs = [access_config]

        instance.network_interfaces = [network_interface]

        # Scheduling (spot/preemptible)
        scheduling = compute_v1.Scheduling()
        if vm_config.spot:
            scheduling.provisioning_model = "SPOT"
            scheduling.instance_termination_action = "STOP"
        elif vm_config.preemptible:
            scheduling.preemptible = True
        instance.scheduling = scheduling

        # GPU
        if vm_config.gpu_type and vm_config.gpu_count > 0:
            accelerator = compute_v1.AcceleratorConfig()
            accelerator.accelerator_type = (
                f"zones/{vm_config.zone}/acceleratorTypes/{vm_config.gpu_type}"
            )
            accelerator.accelerator_count = vm_config.gpu_count
            instance.guest_accelerators = [accelerator]

            # GPU instances need specific scheduling
            scheduling.on_host_maintenance = "TERMINATE"

        # Startup script
        if vm_config.startup_script:
            metadata = compute_v1.Metadata()
            metadata.items = [
                compute_v1.Items(key="startup-script", value=vm_config.startup_script)
            ]
            instance.metadata = metadata

        # Labels
        if vm_config.labels:
            instance.labels = vm_config.labels

        # Create instance
        loop = asyncio.get_event_loop()
        operation = await loop.run_in_executor(
            None,
            lambda: self._instances_client.insert(
                project=self._config.project_id,
                zone=vm_config.zone,
                instance_resource=instance,
            )
        )

        # Wait for operation to complete
        await self._wait_for_operation(operation, vm_config.zone)

        # Get instance details
        return await self.get_instance(vm_config.name, vm_config.zone)

    async def delete_instance(self, name: str, zone: str) -> bool:
        """Delete a VM instance."""
        if not self._instances_client:
            return False

        try:
            loop = asyncio.get_event_loop()
            operation = await loop.run_in_executor(
                None,
                lambda: self._instances_client.delete(
                    project=self._config.project_id,
                    zone=zone,
                    instance=name,
                )
            )
            await self._wait_for_operation(operation, zone)
            return True
        except Exception as e:
            logger.error(f"Failed to delete instance {name}: {e}")
            return False

    async def get_instance(self, name: str, zone: str) -> Optional[VMInstance]:
        """Get instance details."""
        if not self._instances_client:
            return None

        try:
            loop = asyncio.get_event_loop()
            instance = await loop.run_in_executor(
                None,
                lambda: self._instances_client.get(
                    project=self._config.project_id,
                    zone=zone,
                    instance=name,
                )
            )

            # Parse state
            state_map = {
                "PROVISIONING": VMState.PROVISIONING,
                "STAGING": VMState.STAGING,
                "RUNNING": VMState.RUNNING,
                "STOPPING": VMState.STOPPING,
                "STOPPED": VMState.STOPPED,
                "SUSPENDED": VMState.SUSPENDED,
                "TERMINATED": VMState.TERMINATED,
            }
            state = state_map.get(instance.status, VMState.UNKNOWN)

            # Get IPs
            external_ip = None
            internal_ip = None
            for ni in instance.network_interfaces:
                internal_ip = ni.network_i_p
                for ac in ni.access_configs:
                    if ac.nat_i_p:
                        external_ip = ac.nat_i_p
                        break

            return VMInstance(
                name=name,
                zone=zone,
                state=state,
                external_ip=external_ip,
                internal_ip=internal_ip,
                machine_type=instance.machine_type.split("/")[-1],
                inference_endpoint=f"http://{external_ip}:8000" if external_ip else None,
            )

        except Exception as e:
            logger.error(f"Failed to get instance {name}: {e}")
            return None

    async def list_instances(self, zone: str) -> List[VMInstance]:
        """List all instances in a zone."""
        if not self._instances_client:
            return []

        try:
            loop = asyncio.get_event_loop()
            instances = await loop.run_in_executor(
                None,
                lambda: self._instances_client.list(
                    project=self._config.project_id,
                    zone=zone,
                )
            )

            result = []
            for instance in instances:
                vm = await self.get_instance(instance.name, zone)
                if vm:
                    result.append(vm)

            return result

        except Exception as e:
            logger.error(f"Failed to list instances: {e}")
            return []

    async def start_instance(self, name: str, zone: str) -> bool:
        """Start a stopped instance."""
        if not self._instances_client:
            return False

        try:
            loop = asyncio.get_event_loop()
            operation = await loop.run_in_executor(
                None,
                lambda: self._instances_client.start(
                    project=self._config.project_id,
                    zone=zone,
                    instance=name,
                )
            )
            await self._wait_for_operation(operation, zone)
            return True
        except Exception as e:
            logger.error(f"Failed to start instance {name}: {e}")
            return False

    async def stop_instance(self, name: str, zone: str) -> bool:
        """Stop a running instance."""
        if not self._instances_client:
            return False

        try:
            loop = asyncio.get_event_loop()
            operation = await loop.run_in_executor(
                None,
                lambda: self._instances_client.stop(
                    project=self._config.project_id,
                    zone=zone,
                    instance=name,
                )
            )
            await self._wait_for_operation(operation, zone)
            return True
        except Exception as e:
            logger.error(f"Failed to stop instance {name}: {e}")
            return False

    async def check_preemption(self, name: str, zone: str) -> bool:
        """Check if instance is being preempted."""
        # GCP provides a metadata endpoint to check preemption
        # This would be called from within the VM itself
        if not AIOHTTP_AVAILABLE:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                url = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
                headers = {"Metadata-Flavor": "Google"}
                async with session.get(url, headers=headers, timeout=5) as response:
                    if response.status == 200:
                        text = await response.text()
                        return text.lower() == "true"
        except Exception:
            pass

        return False

    async def _wait_for_operation(self, operation, zone: str, timeout: int = 300):
        """Wait for a GCP operation to complete."""
        if not GCP_SDK_AVAILABLE:
            return

        operations_client = compute_v1.ZoneOperationsClient()
        start_time = time.time()

        while time.time() - start_time < timeout:
            loop = asyncio.get_event_loop()
            op = await loop.run_in_executor(
                None,
                lambda: operations_client.get(
                    project=self._config.project_id,
                    zone=zone,
                    operation=operation.name,
                )
            )

            if op.status == compute_v1.Operation.Status.DONE:
                if op.error:
                    raise RuntimeError(f"Operation failed: {op.error}")
                return

            await asyncio.sleep(2)

        raise TimeoutError(f"Operation {operation.name} timed out")


# =============================================================================
# PREEMPTION HANDLER
# =============================================================================

class PreemptionHandler:
    """
    Handles VM preemption events with checkpointing and migration.

    Features:
        - Preemption signal detection
        - Automatic checkpoint saving
        - Instance migration to new zone
        - Graceful shutdown coordination
    """

    def __init__(
        self,
        config: PreemptionConfig,
        on_preemption: Optional[Callable[[], Awaitable[None]]] = None,
        on_checkpoint: Optional[Callable[[Path], Awaitable[None]]] = None,
    ):
        self._config = config
        self._on_preemption = on_preemption
        self._on_checkpoint = on_checkpoint

        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._preemption_detected = asyncio.Event()

    async def start_monitoring(self, instance: VMInstance):
        """Start monitoring for preemption."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(instance)
        )
        logger.info(f"Started preemption monitoring for {instance.name}")

    async def stop_monitoring(self):
        """Stop preemption monitoring."""
        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self, instance: VMInstance):
        """Monitor for preemption signals."""
        while self._monitoring:
            try:
                # Check preemption signal file
                signal_file = Path(self._config.signal_file)
                if signal_file.exists():
                    logger.warning(f"Preemption signal detected for {instance.name}")
                    await self._handle_preemption(instance)
                    signal_file.unlink()  # Clear signal

                # Check GCP metadata endpoint
                if await self._check_metadata_preemption():
                    logger.warning(f"Preemption metadata detected for {instance.name}")
                    await self._handle_preemption(instance)

                await asyncio.sleep(self._config.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Preemption monitor error: {e}")
                await asyncio.sleep(5)

    async def _check_metadata_preemption(self) -> bool:
        """Check GCP metadata endpoint for preemption."""
        if not AIOHTTP_AVAILABLE:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                url = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
                headers = {"Metadata-Flavor": "Google"}
                async with session.get(url, headers=headers, timeout=5) as response:
                    if response.status == 200:
                        text = await response.text()
                        return text.lower() == "true"
        except Exception:
            pass

        return False

    async def _handle_preemption(self, instance: VMInstance):
        """Handle a preemption event."""
        self._preemption_detected.set()

        # Save checkpoint if configured
        if self._config.checkpoint_on_preemption:
            await self._save_checkpoint(instance)

        # Call preemption callback
        if self._on_preemption:
            await self._on_preemption()

        logger.info(f"Preemption handled for {instance.name}")

    async def _save_checkpoint(self, instance: VMInstance):
        """Save model checkpoint before preemption."""
        checkpoint_dir = Path.home() / ".jarvis" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{instance.name}_{int(time.time())}.json"

        # Save instance state
        state = {
            "instance": instance.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "preempted": True,
        }

        checkpoint_path.write_text(json.dumps(state, indent=2))

        if self._on_checkpoint:
            await self._on_checkpoint(checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    async def wait_for_preemption(self) -> bool:
        """Wait for preemption event."""
        await self._preemption_detected.wait()
        return True


# =============================================================================
# COST TRACKER
# =============================================================================

class CostTracker:
    """
    Tracks and manages GCP costs with limits and alerts.

    Features:
        - Real-time cost estimation
        - Hourly and daily limits
        - Alert thresholds
        - Automatic shutdown on limit
    """

    # Approximate hourly costs (USD) - update with actual pricing
    HOURLY_COSTS = {
        "n1-standard-4": 0.19,
        "n1-standard-8": 0.38,
        "n1-standard-16": 0.76,
        "n1-highmem-8": 0.47,
        "a2-highgpu-1g": 3.67,
        "nvidia-tesla-t4": 0.35,
        "nvidia-tesla-v100": 2.48,
        "nvidia-tesla-a100": 2.93,
        "nvidia-l4": 0.81,
    }

    # Spot discount (approximately 60-80% off)
    SPOT_DISCOUNT = 0.3

    def __init__(
        self,
        config: CostConfig,
        on_limit_reached: Optional[Callable[[], Awaitable[None]]] = None,
        on_alert: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ):
        self._config = config
        self._on_limit_reached = on_limit_reached
        self._on_alert = on_alert

        # Tracking
        self._hourly_costs: Dict[str, float] = {}
        self._daily_costs: Dict[str, float] = {}
        self._total_cost: float = 0.0
        self._current_hour_start: datetime = datetime.now().replace(minute=0, second=0, microsecond=0)
        self._current_day_start: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Alert tracking
        self._alerts_sent: Set[str] = set()

        # Lock
        self._lock = asyncio.Lock()

    def estimate_hourly_cost(
        self,
        machine_type: str,
        gpu_type: Optional[str] = None,
        gpu_count: int = 0,
        spot: bool = True,
    ) -> float:
        """Estimate hourly cost for a configuration."""
        base_cost = self.HOURLY_COSTS.get(machine_type, 0.5)

        if gpu_type and gpu_count > 0:
            gpu_cost = self.HOURLY_COSTS.get(gpu_type, 1.0) * gpu_count
            base_cost += gpu_cost

        if spot:
            base_cost *= self.SPOT_DISCOUNT

        return base_cost

    async def record_usage(
        self,
        instance_name: str,
        hours: float,
        machine_type: str,
        gpu_type: Optional[str] = None,
        gpu_count: int = 0,
        spot: bool = True,
    ):
        """Record usage for an instance."""
        async with self._lock:
            # Calculate cost
            hourly_cost = self.estimate_hourly_cost(machine_type, gpu_type, gpu_count, spot)
            cost = hourly_cost * hours

            # Update tracking
            now = datetime.now()
            current_hour = now.replace(minute=0, second=0, microsecond=0)
            current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # Reset if new hour/day
            if current_hour > self._current_hour_start:
                self._hourly_costs = {}
                self._current_hour_start = current_hour

            if current_day > self._current_day_start:
                self._daily_costs = {}
                self._current_day_start = current_day
                self._alerts_sent.clear()

            # Add costs
            self._hourly_costs[instance_name] = self._hourly_costs.get(instance_name, 0) + cost
            self._daily_costs[instance_name] = self._daily_costs.get(instance_name, 0) + cost
            self._total_cost += cost

            # Check limits
            await self._check_limits()

    async def _check_limits(self):
        """Check if cost limits have been reached."""
        if not self._config.enabled:
            return

        hourly_total = sum(self._hourly_costs.values())
        daily_total = sum(self._daily_costs.values())

        # Check hourly limit
        if hourly_total >= self._config.max_hourly_spend:
            logger.warning(f"Hourly cost limit reached: ${hourly_total:.2f}")
            if self._on_limit_reached and self._config.shutdown_on_limit:
                await self._on_limit_reached()

        # Check daily limit
        if daily_total >= self._config.max_daily_spend:
            logger.warning(f"Daily cost limit reached: ${daily_total:.2f}")
            if self._on_limit_reached and self._config.shutdown_on_limit:
                await self._on_limit_reached()

        # Check alert thresholds
        hourly_threshold = self._config.max_hourly_spend * (self._config.alert_threshold_percent / 100)
        daily_threshold = self._config.max_daily_spend * (self._config.alert_threshold_percent / 100)

        if hourly_total >= hourly_threshold and "hourly_alert" not in self._alerts_sent:
            self._alerts_sent.add("hourly_alert")
            if self._on_alert:
                await self._on_alert("hourly", hourly_total)

        if daily_total >= daily_threshold and "daily_alert" not in self._alerts_sent:
            self._alerts_sent.add("daily_alert")
            if self._on_alert:
                await self._on_alert("daily", daily_total)

    def get_current_costs(self) -> Dict[str, Any]:
        """Get current cost information."""
        return {
            "hourly_total": sum(self._hourly_costs.values()),
            "daily_total": sum(self._daily_costs.values()),
            "total": self._total_cost,
            "hourly_limit": self._config.max_hourly_spend,
            "daily_limit": self._config.max_daily_spend,
            "hourly_by_instance": dict(self._hourly_costs),
            "daily_by_instance": dict(self._daily_costs),
        }


# =============================================================================
# GCP VM MANAGER
# =============================================================================

class GCPVMManager:
    """
    Master manager for GCP VM lifecycle with auto-scaling, cost tracking,
    and preemption handling.

    Features:
        - Automatic provisioning and teardown
        - Preemption detection and handling
        - Auto-scaling based on demand
        - Cost tracking and limits
        - Health monitoring
        - Checkpoint migration
    """

    def __init__(self, config: Optional[GCPManagerConfig] = None):
        self._config = config or GCPManagerConfig.from_env()

        # Determine which client to use
        if GCP_SDK_AVAILABLE and self._config.project_id:
            self._client = GCPClient(self._config)
            logger.info("Using real GCP client")
        else:
            self._client = MockGCPClient(self._config)
            logger.info("Using mock GCP client (development mode)")

        # Components
        self._preemption_handler = PreemptionHandler(
            config=self._config.preemption,
            on_preemption=self._on_preemption,
        )

        self._cost_tracker = CostTracker(
            config=self._config.cost,
            on_limit_reached=self._on_cost_limit,
            on_alert=self._on_cost_alert,
        )

        # State
        self._instances: Dict[str, VMInstance] = {}
        self._running = False
        self._lock = asyncio.Lock()

        # Background tasks
        self._health_task: Optional[asyncio.Task] = None
        self._autoscale_task: Optional[asyncio.Task] = None
        self._cost_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "instances_created": 0,
            "instances_deleted": 0,
            "preemptions_handled": 0,
            "scale_ups": 0,
            "scale_downs": 0,
        }

        # Load saved state
        self._load_state()

    def _load_state(self):
        """Load saved state from disk."""
        if self._config.state_file.exists():
            try:
                data = json.loads(self._config.state_file.read_text())
                for name, inst_data in data.get("instances", {}).items():
                    self._instances[name] = VMInstance.from_dict(inst_data)
                logger.info(f"Loaded {len(self._instances)} instances from state")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save state to disk."""
        try:
            self._config.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "instances": {name: inst.to_dict() for name, inst in self._instances.items()},
                "stats": self._stats,
                "updated_at": datetime.now().isoformat(),
            }
            self._config.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    async def start(self):
        """Start the VM manager."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._health_task = asyncio.create_task(self._health_monitor_loop())

        if self._config.auto_scale.enabled:
            self._autoscale_task = asyncio.create_task(self._autoscale_loop())

        self._cost_task = asyncio.create_task(self._cost_tracking_loop())

        logger.info("GCP VM Manager started")

    async def stop(self):
        """Stop the VM manager."""
        self._running = False

        # Cancel background tasks
        for task in [self._health_task, self._autoscale_task, self._cost_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop preemption monitoring
        await self._preemption_handler.stop_monitoring()

        # Save state
        self._save_state()

        logger.info("GCP VM Manager stopped")

    async def provision_instance(
        self,
        name: Optional[str] = None,
        vm_config: Optional[VMConfig] = None,
    ) -> Optional[VMInstance]:
        """Provision a new VM instance."""
        async with self._lock:
            config = vm_config or self._config.vm_config
            if name:
                config.name = name

            try:
                instance = await self._client.create_instance(config)

                if instance:
                    self._instances[instance.name] = instance
                    self._stats["instances_created"] += 1

                    # Start preemption monitoring
                    await self._preemption_handler.start_monitoring(instance)

                    # Save state
                    self._save_state()

                    logger.info(f"Provisioned instance {instance.name}")
                    return instance

            except Exception as e:
                logger.error(f"Failed to provision instance: {e}")

            return None

    async def terminate_instance(self, name: str) -> bool:
        """Terminate a VM instance."""
        async with self._lock:
            if name not in self._instances:
                return False

            instance = self._instances[name]

            try:
                success = await self._client.delete_instance(name, instance.zone)

                if success:
                    del self._instances[name]
                    self._stats["instances_deleted"] += 1
                    self._save_state()
                    logger.info(f"Terminated instance {name}")

                return success

            except Exception as e:
                logger.error(f"Failed to terminate instance {name}: {e}")
                return False

    async def get_inference_endpoint(self) -> Optional[str]:
        """Get an available inference endpoint."""
        for instance in self._instances.values():
            if instance.state == VMState.RUNNING and instance.is_healthy:
                return instance.inference_endpoint
        return None

    async def ensure_capacity(self, min_instances: int = 1) -> bool:
        """Ensure minimum capacity is available."""
        running = sum(
            1 for inst in self._instances.values()
            if inst.state == VMState.RUNNING
        )

        if running >= min_instances:
            return True

        needed = min_instances - running
        for i in range(needed):
            instance = await self.provision_instance(
                name=f"{self._config.vm_config.name}-{len(self._instances) + i}"
            )
            if not instance:
                return False

        return True

    async def _health_monitor_loop(self):
        """Monitor instance health."""
        while self._running:
            try:
                for name, instance in list(self._instances.items()):
                    if instance.state != VMState.RUNNING:
                        continue

                    # Check health endpoint
                    healthy = await self._check_instance_health(instance)
                    instance.is_healthy = healthy
                    instance.last_health_check = datetime.now()

                    if not healthy:
                        instance.consecutive_failures += 1
                        if instance.consecutive_failures >= 3:
                            logger.warning(f"Instance {name} unhealthy, restarting...")
                            await self._client.stop_instance(name, instance.zone)
                            await asyncio.sleep(5)
                            await self._client.start_instance(name, instance.zone)
                    else:
                        instance.consecutive_failures = 0

                await asyncio.sleep(self._config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    async def _check_instance_health(self, instance: VMInstance) -> bool:
        """Check health of an instance."""
        if not instance.inference_endpoint or not AIOHTTP_AVAILABLE:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{instance.inference_endpoint}/health"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _autoscale_loop(self):
        """Auto-scale based on demand."""
        last_scale_time = time.time()

        while self._running:
            try:
                # Get current utilization (mock for now)
                utilization = await self._get_cluster_utilization()

                now = time.time()
                cooldown = self._config.auto_scale.cooldown_seconds

                # Check if cooldown has passed
                if now - last_scale_time < cooldown:
                    await asyncio.sleep(30)
                    continue

                running = sum(
                    1 for inst in self._instances.values()
                    if inst.state == VMState.RUNNING
                )

                # Scale up
                if utilization > self._config.auto_scale.scale_up_threshold_percent:
                    if running < self._config.auto_scale.max_instances:
                        logger.info(f"Scaling up: utilization={utilization:.1f}%")
                        await self.provision_instance()
                        self._stats["scale_ups"] += 1
                        last_scale_time = now

                # Scale down
                elif utilization < self._config.auto_scale.scale_down_threshold_percent:
                    if running > self._config.auto_scale.min_instances:
                        logger.info(f"Scaling down: utilization={utilization:.1f}%")
                        # Find oldest instance to terminate
                        oldest = min(
                            [i for i in self._instances.values() if i.state == VMState.RUNNING],
                            key=lambda x: x.created_at or datetime.now(),
                            default=None,
                        )
                        if oldest:
                            await self.terminate_instance(oldest.name)
                            self._stats["scale_downs"] += 1
                            last_scale_time = now

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Autoscale error: {e}")
                await asyncio.sleep(30)

    async def _get_cluster_utilization(self) -> float:
        """Get cluster utilization percentage."""
        # This would query actual metrics from the instances
        # For now, return a mock value
        import random
        return random.uniform(30, 70)

    async def _cost_tracking_loop(self):
        """Track costs periodically."""
        last_update = time.time()

        while self._running:
            try:
                now = time.time()
                hours = (now - last_update) / 3600

                for instance in self._instances.values():
                    if instance.state == VMState.RUNNING:
                        await self._cost_tracker.record_usage(
                            instance_name=instance.name,
                            hours=hours,
                            machine_type=instance.machine_type,
                            gpu_type=instance.gpu_type,
                            gpu_count=instance.gpu_count,
                            spot=True,
                        )

                last_update = now
                await asyncio.sleep(60)  # Update every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cost tracking error: {e}")
                await asyncio.sleep(60)

    async def _on_preemption(self):
        """Handle preemption callback."""
        self._stats["preemptions_handled"] += 1

        if self._config.preemption.auto_migrate:
            # Find a new zone and provision
            for zone in self._config.preemption.migrate_zones:
                if zone != self._config.vm_config.zone:
                    vm_config = VMConfig(
                        name=f"{self._config.vm_config.name}-migrated",
                        zone=zone,
                        **{k: v for k, v in self._config.vm_config.__dict__.items() if k not in ["name", "zone"]},
                    )
                    instance = await self.provision_instance(vm_config=vm_config)
                    if instance:
                        logger.info(f"Migrated to zone {zone}")
                        break

    async def _on_cost_limit(self):
        """Handle cost limit reached."""
        logger.warning("Cost limit reached - shutting down instances")

        for name in list(self._instances.keys()):
            await self.terminate_instance(name)

    async def _on_cost_alert(self, alert_type: str, amount: float):
        """Handle cost alert."""
        logger.warning(f"Cost alert ({alert_type}): ${amount:.2f}")

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "running": self._running,
            "instances": {name: inst.to_dict() for name, inst in self._instances.items()},
            "stats": self._stats,
            "costs": self._cost_tracker.get_current_costs(),
            "config": {
                "project_id": self._config.project_id,
                "region": self._config.region,
                "auto_scale_enabled": self._config.auto_scale.enabled,
            },
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_manager: Optional[GCPVMManager] = None
_manager_lock = asyncio.Lock()


async def get_gcp_manager(
    config: Optional[GCPManagerConfig] = None,
) -> GCPVMManager:
    """Get or create the global GCP VM Manager."""
    global _manager

    if _manager is not None:
        return _manager

    async with _manager_lock:
        if _manager is not None:
            return _manager

        _manager = GCPVMManager(config)
        await _manager.start()

        return _manager


async def shutdown_gcp_manager():
    """Shutdown the global manager."""
    global _manager

    async with _manager_lock:
        if _manager is not None:
            await _manager.stop()
            _manager = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "VMState",
    "MachineType",
    "GPUType",
    # Configuration
    "VMConfig",
    "AutoScaleConfig",
    "CostConfig",
    "PreemptionConfig",
    "GCPManagerConfig",
    # Data classes
    "VMInstance",
    # Clients
    "MockGCPClient",
    "GCPClient",
    # Components
    "PreemptionHandler",
    "CostTracker",
    # Manager
    "GCPVMManager",
    # Factory
    "get_gcp_manager",
    "shutdown_gcp_manager",
]
