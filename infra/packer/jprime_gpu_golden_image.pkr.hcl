# =============================================================================
# jprime_gpu_golden_image.pkr.hcl
#
# HashiCorp Packer template to BAKE the `jarvis-prime-coder-32b` GPU golden image
# for the Sovereign Failover Mesh's QUALITY tier (Adaptive Workload Provisioner).
#
# WHY PRE-BAKED (not cloud-init): the failover node is a TEMPORARY survival tier.
# Installing NVIDIA drivers + CUDA + pulling a 32B model at boot costs 5-10 min of
# cold-boot — unacceptable.
#
# BASE = Google Deep Learning VM (DLVM): the NVIDIA driver + CUDA toolkit are
# PRE-INSTALLED, kernel-matched, and officially maintained — this image only adds
# Ollama + the pre-pulled 32B weights on top, so the quality node boots READY. No
# DKMS compile (that failure class is eliminated). The runtime cloud-init only
# forces the Ollama bind (failover_deadman.build_inference_bind_block) and the
# TTFT armor (prime_client) absorbs the model-load latency.
#
# The companion survival image (`jarvis-prime-coder`, 7B/CPU) is a separate, much
# simpler bake (no driver/CUDA) and is NOT in scope here.
#
# Build:
#   packer init  jprime_gpu_golden_image.pkr.hcl
#   packer build -var "project_id=jarvis-473803" jprime_gpu_golden_image.pkr.hcl
#
# Notes
#   * Everything is a `variable` — NO hardcoded project / zone / model / GPU.
#   * The BUILD instance carries a GPU so `ollama pull` warms + verifies on-device.
#   * Image family `jarvis-prime-coder-32b` is what failover_tier.py's
#     JARVIS_FAILOVER_QUALITY_IMAGE default resolves to — the provisioner POSTs
#     sourceImage=.../family/jarvis-prime-coder-32b.
# =============================================================================

packer {
  required_plugins {
    googlecompute = {
      source  = "github.com/hashicorp/googlecompute"
      version = ">= 1.1.0"
    }
  }
}

# ----------------------------------------------------------------------------
# Variables — override at build time; defaults match failover_tier.py.
# ----------------------------------------------------------------------------
variable "project_id" {
  type        = string
  description = "GCP project to build + publish the image in."
}

variable "zone" {
  type        = string
  default     = "us-central1-b"
  description = "Build zone (CPU instance — no accelerator constraint; any zone works)."
}

variable "image_family" {
  type        = string
  default     = "jarvis-prime-coder-32b"
  description = "Published family — must equal JARVIS_FAILOVER_QUALITY_IMAGE."
}

variable "source_image" {
  type        = string
  default     = "common-cu124-v20250325-debian-11"
  description = "Google Deep Learning VM (DLVM) image — NVIDIA driver + CUDA PRE-INSTALLED, kernel-matched, Google-maintained (eliminates the DKMS compile failure class). Pinned by EXACT NAME (not family): DLVM marks every image `DEPRECATED` the moment a newer one ships, which 404s the family pointer — but a DEPRECATED image is still launchable by name. Bump this when a newer common-cu* image ships."
}

variable "source_image_project" {
  type        = string
  default     = "deeplearning-platform-release"
  description = "Project hosting the DLVM source image."
}

variable "build_machine_type" {
  type        = string
  default     = "e2-standard-4"
  description = "Build VM — a cheap CPU box. BAKING needs NO GPU (only Ollama install + 32B pull, both CPU work). The GPU is a RUNTIME concern; on-device hardware validation moved to the runtime cloud-init nvidia-smi gate. CPU nodes never stock out -> deterministic bake."
}

variable "model_label" {
  type        = string
  default     = "qwen2.5-coder:32b"
  description = "Ollama model to PRE-PULL into the image. Must equal JARVIS_FAILOVER_QUALITY_MODEL."
}

variable "disk_size_gb" {
  type        = number
  default     = 150 # DLVM base (~50GB: driver+CUDA+conda) + 32B Q4 weights ~20GB + headroom
}

# ----------------------------------------------------------------------------
# Source — a cheap CPU build instance (NO accelerator). Baking only downloads the
# model + installs Ollama; the GPU is a runtime concern (validated by the cloud-
# init nvidia-smi gate). CPU nodes never stock out -> deterministic bake.
# ----------------------------------------------------------------------------
source "googlecompute" "jprime_gpu" {
  project_id              = var.project_id
  zone                    = var.zone
  source_image            = var.source_image # pinned by name (DLVM family pointers 404)
  source_image_project_id = [var.source_image_project] # plugin expects a list
  image_name              = "${var.image_family}-{{timestamp}}"
  image_family            = var.image_family
  image_description       = "JARVIS J-Prime QUALITY tier: ${var.model_label}, DLVM base (driver+CUDA) + pre-pulled Ollama 32B. Baked on CPU; GPU validated at runtime."
  machine_type            = var.build_machine_type
  disk_size               = var.disk_size_gb
  ssh_username            = "packer"

  image_labels = {
    role  = "jprime-failover-quality"
    model = replace(replace(var.model_label, ":", "_"), ".", "-")
    tier  = "gpu-32b" # runtime tier (the image runs on an L4 at runtime)
  }
}

# ----------------------------------------------------------------------------
# Build (CPU node) — Ollama install → pre-pull the 32B weights. NO GPU smoke test
# here (the build node has no GPU); the NVIDIA driver + CUDA are already in the
# DLVM base and are validated at RUNTIME by the cloud-init nvidia-smi gate.
# ----------------------------------------------------------------------------
build {
  sources = ["source.googlecompute.jprime_gpu"]

  # 1) Ollama (the DLVM base already carries the NVIDIA driver + CUDA for runtime).
  #    The DLVM base lacks `zstd`, which the Ollama installer REQUIRES to extract
  #    its binary -- install it first (also `jq`/`curl` for robustness). Wait for
  #    the DLVM first-boot apt/dpkg lock to release before apt.
  provisioner "shell" {
    inline = [
      "set -euxo pipefail",
      "for i in $(seq 1 60); do sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 && sleep 5 || break; done",
      # The DLVM (debian-11/bullseye) ships a `bullseye-backports` apt source that
      # has been RETIRED from deb.debian.org -> `apt-get update` errors out. Strip
      # every backports source (zstd lives in main, not backports).
      "sudo sed -i '/backports/d' /etc/apt/sources.list 2>/dev/null || true",
      "sudo rm -f /etc/apt/sources.list.d/*backports*.list 2>/dev/null || true",
      # -o DPkg::Lock::Timeout: wait out the DLVM's first-boot apt instead of failing.
      "sudo apt-get -o DPkg::Lock::Timeout=300 update -y",
      "sudo apt-get -o DPkg::Lock::Timeout=300 install -y zstd jq curl",
      "curl -fsSL https://ollama.com/install.sh | sudo sh",
      "sudo systemctl enable ollama",
    ]
  }

  # 2) PRE-PULL the 32B weights INTO the image (the whole point — no boot download).
  #    On the CPU build node the pull is a plain download; no GPU load is attempted.
  provisioner "shell" {
    inline = [
      "set -euxo pipefail",
      "sudo systemctl start ollama",
      "for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/tags && break || sleep 2; done",
      # Pull via the RUNNING systemd daemon (which has HOME set + owns the model
      # store) -- NOT `sudo -u ollama` (HOME unset -> Ollama/Go panics; the known
      # bake lesson). The daemon downloads + stores; the CLI just tells it to.
      "OLLAMA_HOST=127.0.0.1:11434 ollama pull ${var.model_label}",
      # Verify the weights landed in the image's model store.
      "OLLAMA_HOST=127.0.0.1:11434 ollama list | grep -q \"$(echo ${var.model_label} | cut -d: -f1)\"",
      "sudo systemctl stop ollama",
    ]
  }

  # 4) Bake the systemd drop-in so the runtime cloud-init only needs to (re)bind.
  #    Mirrors failover_deadman.build_inference_bind_block — bind 0.0.0.0 so the
  #    Reachability Racer's external natIP probe can reach it.
  provisioner "shell" {
    inline = [
      "set -euxo pipefail",
      "sudo mkdir -p /etc/systemd/system/ollama.service.d",
      "printf '[Service]\\nEnvironment=\"OLLAMA_HOST=0.0.0.0:11434\"\\nEnvironment=\"OLLAMA_KEEP_ALIVE=-1\"\\n' | sudo tee /etc/systemd/system/ollama.service.d/10-jarvis-bind.conf",
      "sudo systemctl daemon-reload",
      # Clean cloud-init state so the published image boots fresh.
      "sudo cloud-init clean --logs || true",
    ]
  }
}
