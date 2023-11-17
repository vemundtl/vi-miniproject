import os
import signal
import subprocess

def kill_tensorboard_processes():
    # Get the list of running processes
    try:
        output = subprocess.check_output(["pgrep", "-fl", "tensorboard"])
    except subprocess.CalledProcessError:
        print("No TensorBoard processes found.")
        return

    # Parse the output and kill each process
    for line in output.decode("utf-8").splitlines():
        pid = int(line.split()[0])
        os.kill(pid, signal.SIGTERM)
        print(f"Killed TensorBoard process with PID: {pid}")

if __name__ == "__main__":
    kill_tensorboard_processes()