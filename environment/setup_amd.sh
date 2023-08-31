#!/bin/bash
sudo tuned-adm profile throughput-performance
sudo cpupower -c all frequency-set -g performance
echo 10000 | sudo tee /proc/sys/kernel/sched_cfs_bandwidth_slice_us
echo 0 | sudo tee /proc/sys/kernel/sched_child_runs_first
echo 100 | sudo tee /proc/sys/kernel/sched_rr_timeslice_ms
echo 1000000 | sudo tee /proc/sys/kernel/sched_rt_period_us
echo 990000 | sudo tee /proc/sys/kernel/sched_rt_runtime_us
echo 0 | sudo tee /proc/sys/kernel/sched_schedstats
echo 3000 | sudo tee /proc/sys/vm/dirty_expire_centisecs
echo 500 | sudo tee /proc/sys/vm/dirty_writeback_centisecs
echo 40 | sudo tee /proc/sys/vm/dirty_ratio
echo 10 | sudo tee /proc/sys/vm/dirty_background_ratio
echo 10 | sudo tee /proc/sys/vm/swappiness
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

ulimit -n 1024000

if ! grep -q "\[always\]" /sys/kernel/mm/transparent_hugepage/defrag; then
  echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
fi
if ! grep -q "\[always\]" /sys/kernel/mm/transparent_hugepage/enabled; then
  echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
fi

if ! grep -q "UserTasksMax=970000" /etc/systemd/logind.conf; then
  echo UserTasksMax=970000 | sudo tee -a /etc/systemd/logind.conf
fi

if ! grep -q "DefaultTasksMax=970000" /etc/systemd/system.conf; then
  echo DefaultTasksMax=970000 | sudo tee -a /etc/systemd/system.conf
fi

if ! grep -q "vm.max_map_count=471859" /etc/sysctl.conf; then
  echo vm.max_map_count=471859 | sudo tee -a /etc/sysctl.conf
  sudo sysctl -p
fi
