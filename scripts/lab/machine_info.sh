#!/usr/bin/env bash
# machine_info.sh — hardware/software summary of the machine it runs on (no root needed).
#   bash scripts/lab/machine_info.sh
#   bash scripts/lab/machine_info.sh > ~/results/info_$(hostname).txt   # save it
sec() { printf '\n== %s ==\n' "$1"; }
has() { command -v "$1" >/dev/null 2>&1; }

sec "machine"
echo "host:    $(hostname)"
echo "os:      $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") / kernel $(uname -r)"
echo "uptime: $(uptime)"

sec "CPU"
if has lscpu; then
  lscpu | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)|CPU max MHz|L3 cache)' | sed 's/  */ /g'
else
  grep -m1 'model name' /proc/cpuinfo
fi
echo "usable cores for you: $(nproc)"

sec "memory"
free -h | head -2
echo "/dev/shm: $(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2" total, "$4" free"}')"

sec "GPU"
if has lspci; then
  lspci | grep -i -E 'vga|3d|display' || echo "(no display/3D controllers found)"
fi
if has nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
  echo "CUDA (driver max): $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*')"
  echo "GPU processes:"; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>/dev/null || true
elif lspci 2>/dev/null | grep -qi nvidia; then
  echo "NVIDIA hardware present but nvidia-smi missing -> driver not installed (ask the admin)"
else
  echo "=> no NVIDIA GPU: not usable for training"
fi
has rocm-smi && rocm-smi --showproductname 2>/dev/null

sec "disks (local vs network)"
df -hT -x tmpfs -x devtmpfs -x squashfs -x overlay 2>/dev/null | awk 'NR==1 || $7 ~ /^\/(|tmp|scratch|local|data|home|cta.*)$/'
echo "home: $HOME -> $(df -P "$HOME" | awk 'NR==2{print $1}')"

sec "software"
for c in python3 gcc nvcc git curl tmux screen uv; do
  if has "$c"; then printf '%-8s %s\n' "$c" "$("$c" --version 2>&1 | head -1)"; else printf '%-8s (not installed)\n' "$c"; fi
done
python3 -c "import venv, ensurepip" 2>/dev/null && echo "python3-venv: yes" || echo "python3-venv: no (use uv)"
has module && echo "environment modules: yes (try 'module avail')"
has sbatch && echo "SLURM: yes (jobs must go through sbatch)"

sec "network"
if curl -s -m 5 -o /dev/null -w '%{http_code}' https://huggingface.co | grep -q '^[23]'; then
  echo "internet: yes"; else echo "internet: no"; fi

sec "who else is using it"
echo "logged-in users: $(who | awk '{print $1}' | sort -u | tr '\n' ' ')"
echo "load average:    $(cut -d' ' -f1-3 /proc/loadavg)  (cores: $(nproc))"
echo "top CPU processes:"
ps -eo user,pcpu,pmem,etime,comm --sort=-pcpu | head -6
