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

# -- Resilience knobs (NOT magic disk sizes). The required disk FOOTPRINT is
#    never hardcoded/parameterized -- it is DERIVED at runtime from the model's
#    registry manifest (see the dynamic disk preflight in the pull provisioner).
variable "pull_max_attempts" {
  type        = number
  default     = 3
  description = "Bounded retry count for the resumable `ollama pull` AND the registry manifest probe. `ollama pull` keeps already-fetched blobs, so a retry RESUMES -- a single transient network reset must never waste the whole bake."
}

variable "pull_backoff_base_s" {
  type        = number
  default     = 20
  description = "Linear backoff base (seconds): attempt N sleeps N*base before the next try."
}

variable "ollama_registry" {
  type        = string
  default     = "registry.ollama.ai"
  description = "Ollama model registry host. Used to resolve the model's EXACT byte footprint for the dynamic disk preflight -- so the disk requirement adapts to whatever model_label actually is, with zero hardcoded GB."
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
    # Run under bash, not Packer's default /bin/sh (= dash on Debian DLVM), which
    # rejects `set -o pipefail` ("Illegal option") and silently swallows pipe
    # failures. Our pull provisioner relies on pipefail (curl|jq, df|awk, list|grep).
    inline_shebang = "/bin/bash -e"
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

  # 2) PRE-PULL the weights INTO the image (the whole point — no boot download).
  #    On the CPU build node the pull is a plain download; no GPU load is attempted.
  #    Hardened against the "passes-bake-fails-live" class:
  #      (a) DYNAMIC disk preflight — derive the model's EXACT footprint from its
  #          registry manifest (no hardcoded/parameterized GB) and assert df.
  #      (b) RESUMABLE retry+backoff on the pull (a transient reset != dead bake).
  #      (c) INTEGRITY proof via `ollama show` (dereferences manifest + config
  #          blob) — strictly stronger than `list` (manifest-presence only).
  provisioner "shell" {
    inline_shebang = "/bin/bash -e"  # bash (not dash) -> `set -o pipefail` is honored
    inline = [
      "set -euxo pipefail",
      "MODEL='${var.model_label}'",
      # -- Parse <ns>/<name>:<tag> (Ollama default ns=library, default tag=latest).
      "case \"$MODEL\" in *:*) REPO=\"$${MODEL%:*}\"; TAG=\"$${MODEL##*:}\";; *) REPO=\"$MODEL\"; TAG=latest;; esac",
      "case \"$REPO\" in */*) NS=\"$${REPO%%/*}\"; NAME=\"$${REPO##*/}\";; *) NS=library; NAME=\"$REPO\";; esac",
      "MANIFEST_URL=\"https://${var.ollama_registry}/v2/$${NS}/$${NAME}/manifests/$${TAG}\"",
      "echo \"[disk-preflight] resolving footprint via $MANIFEST_URL\"",
      # -- (a) DYNAMIC disk preflight. Sum the manifest layer+config byte sizes =
      #    the EXACT on-disk blob footprint; x1.5 unpack/headroom heuristic; assert
      #    live df has it. The required size adapts to whatever MODEL is -- no GB
      #    is ever hardcoded or parameterized. The manifest probe shares the same
      #    bounded retry so a registry blip doesn't false-fail the preflight.
      "MANIFEST=''; for a in $(seq 1 ${var.pull_max_attempts}); do MANIFEST=$(curl -fsSL -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \"$MANIFEST_URL\") && break; echo \"[disk-preflight] manifest probe $a failed -> retry\"; sleep $((a * ${var.pull_backoff_base_s})); done",
      "[ -n \"$MANIFEST\" ] || { echo \"FATAL: registry manifest unreachable for $MODEL\"; exit 1; }",
      "NEED=$(echo \"$MANIFEST\" | jq -e '[.layers[].size] + [.config.size] | add') || { echo 'FATAL: could not parse model byte size from manifest'; exit 1; }",
      "[ \"$NEED\" -gt 0 ] 2>/dev/null || { echo \"FATAL: non-positive model size ($NEED)\"; exit 1; }",
      # x1.5 unpack/headroom heuristic (integer byte math).
      "REQ=$(( NEED * 3 / 2 )); AVAIL=$(df -PB1 / | awk 'NR==2{print $4}')",
      "echo \"[disk-preflight] model=$MODEL need_bytes=$NEED required_x1.5=$REQ avail_bytes=$AVAIL\"",
      "[ \"$AVAIL\" -ge \"$REQ\" ] || { echo \"FATAL INSUFFICIENT DISK: avail=$AVAIL < required=$REQ (model $NEED x1.5) for $MODEL\"; exit 1; }",
      "echo '[disk-preflight] OK'",
      # -- Start the daemon and wait for its REST socket.
      "sudo systemctl start ollama",
      "for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/tags && break || sleep 2; done",
      # -- (b) RESUMABLE pull with bounded retry+backoff. The CLI reads $HOME/.ollama
      #    on startup and PANICS (Go) if HOME is unset -- `sudo HOME=/root` fixes it
      #    (documented bake lesson). The daemon does the download/store; the CLI
      #    instructs it. `ollama pull` keeps fetched blobs -> a retry RESUMES.
      "for a in $(seq 1 ${var.pull_max_attempts}); do echo \"[pull] attempt $a/${var.pull_max_attempts} model=$MODEL\"; sudo HOME=/root OLLAMA_HOST=127.0.0.1:11434 ollama pull \"$MODEL\" && { echo '[pull] OK'; break; }; rc=$?; if [ \"$a\" -ge ${var.pull_max_attempts} ]; then echo \"FATAL: pull exhausted (rc=$rc)\"; exit $rc; fi; bk=$((a * ${var.pull_backoff_base_s})); echo \"[pull] failed rc=$rc -> resumable retry in $${bk}s\"; sleep $bk; done",
      # -- (c) INTEGRITY proof: `ollama show` dereferences the manifest + config
      #    blob; an incomplete/corrupt model fails HERE (unlike `list`). Then the
      #    presence proof confirms the exact label is registered in the store.
      "sudo HOME=/root OLLAMA_HOST=127.0.0.1:11434 ollama show \"$MODEL\" >/dev/null || { echo \"FATAL INTEGRITY: ollama show failed for $MODEL (manifest/blob incomplete)\"; exit 1; }",
      "sudo HOME=/root OLLAMA_HOST=127.0.0.1:11434 ollama list | grep -q \"$(echo \"$MODEL\" | cut -d: -f1)\" || { echo \"FATAL: $MODEL not present in ollama list\"; exit 1; }",
      "echo '[pull] integrity + presence proven'",
      "sudo systemctl stop ollama",
    ]
  }

  # 3) Bake the systemd drop-in so the runtime cloud-init only needs to (re)bind.
  #    Mirrors failover_deadman.build_inference_bind_block — bind 0.0.0.0 so the
  #    Reachability Racer's external natIP probe can reach it.
  provisioner "shell" {
    inline_shebang = "/bin/bash -e"  # bash (not dash) -> `set -o pipefail` is honored
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
