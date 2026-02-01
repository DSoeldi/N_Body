# main_concurrent_popen.py
import subprocess
import sys

# Determine the correct Python executable (python, python3, py)
python_executable = sys.executable # Usually the safest way to get the current interpreter

print(f"Using Python executable: {python_executable}")

# List of scripts to run
scripts = [r'./pm_python_experiments/brute_num_int_anim.py', 
           r'./pm_python_experiments/brute_num_int.py']
# # List of scripts to run
# scripts = [r'./pm_python_experiments/pm_num_int_anim.py', 
#            r'./pm_python_experiments/pm_num_int.py', 
#            r'./pm_python_experiments/brute_num_int_anim.py', 
#            r'./pm_python_experiments/brute_num_int.py']
processes = []

print("Starting processes concurrently...")
# Create and start each process
for script in scripts:
    # Popen starts the process and returns immediately
    proc = subprocess.Popen([python_executable, script])
    processes.append(proc)
    print(f"  Started {script} with PID: {proc.pid}")

print("Waiting for processes to finish...")
# Optionally, wait for all processes to complete
for proc in processes:
    proc.wait() # Wait for this specific process to terminate
    print(f"  Process {proc.pid} finished with code: {proc.returncode}")

print("All concurrent processes finished.")