from __future__ import annotations
import sys
import subprocess
from typing import Literal
from pathlib import Path
import os
import getpass


# Configuration
env_file = "syntax"  # Name of the virtual environment or directory
requirement_file = Path(__file__).parent.parent.absolute().joinpath('requirements.txt').__str__()
# Path to the requirements.txt file

cmd = None  # Placeholder for command input

# TODO: 
# Remove environment - rm -r ./syntax
def pip_available(path : str) -> str | None:

    for pip_version in ['pip', 'pip3']:
        try:
            pip_path : str = os.path.join(path, "bin", pip_version) 
            subprocess.run([pip_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"- {pip_available} Installed...")
            return pip_path
        except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"- {pip_available} Not-Installed...")
                continue
    return None

def create_environment(env_type : Literal["conda"] | Literal["local"]):
    try:
        path = Path(__file__).parent.parent.absolute().joinpath(env_file)
        if env_type == "local":   
            # Step 1: Create a virtual environment
            subprocess.check_call([sys.executable, "-m", "venv", "syntax"])

            # Step 2: Activate the virtual environment
            if sys.platform == "win32":
                activate_script = os.path.join(env_file, "Scripts", "activate")
            else:
                activate_script = os.path.join(path, "bin", "activate")
            subprocess.run(["source", activate_script], shell=True)

            while cmd :=input("[y/n]").lower() :
                if cmd in ['y', 'n']:
                    break

            # Step 3: Install requirements using pip
            if cmd == 'y':
                pip = pip_available(path) 

                if pip != None:
                    print(pip)
                    subprocess.run([pip, "install", "-r", requirement_file])
   
                else:
                    print("need to install pip module...")

            # Step 4: Deactivate the virtual environment
            subprocess.run(["deactivate"], shell=True)
        else:
            subprocess.run(['conda', 'create', '--name', env_file])

    except subprocess.CalledProcessError:
        print("An error occurred while creating the environment or installing requirements.")

# TODO: When migrating to docker implement this functions
# def docker_manager():
#     print(f"'sorry I can do that {getpass.getuser()}...' - HAL-9000")


# TODO: build unit tester function that preforms a set 
#       of unit test within the backend
def unit_tester():
    print(f"'sorry I can do that {getpass.getuser()}...' - HAL-9000")

# TODO: 
#   - Build Unit testing module
#   - Build Docker when we build the backend api
OPTIONS = {'l' : 'create_environment',
           'a' : 'create_environment',
           'u' : 'unit_tester', }

def function_commands(cmd: str):
    """
    Execute a command specified by the `cmd` argument using subprocess.run.

    Parameters:
        cmd (str): The command to be executed, which should be a key in the OPTIONS dictionary.

    Returns:
        None

    Prints the standard output of the executed command or an error message in case of failure.

    Raises:
        subprocess.CalledProcessError: If the command execution fails.
    """
    try:
        fn = getattr(sys.modules[__name__], OPTIONS[cmd])
        if cmd in ["l", 'a']:
            fn('local' if cmd == 'l' else 'conda')
        else:
            fn()
    except subprocess.CalledProcessError as e:
        print(f"Error executing the command: {e}")

def menu():
    print(f"What would you like to do?\n\
\t-{'(l)':^6}Local Build\n\
\t-{'(a)':^6}Anaconda Build\n\
\t-{'(u)':^6}Unit Testing\n\
\t-{'(q)':^6}Quit Program\n")    

if __name__ == "__main__":
    menu()
    try:
        cmd = input("Enter command: ").lower()
        while not cmd in list(OPTIONS.keys()):
            if cmd == "q":
                sys.exit()
            print("\nSorry that command does not exist... try one of the command that do exist\n")
            cmd = input("Enter command: ").lower()
    except KeyboardInterrupt as key:
        sys.exit()
        
    function_commands(cmd)