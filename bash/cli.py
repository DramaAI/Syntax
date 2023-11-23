from __future__ import annotations
import sys
import subprocess
from typing import Literal
from pathlib import Path
import os
import getpass


# Configuration
env_file = "syntax"  # Name of the virtual environment or directory
repo = Path(__file__).parent.parent.absolute()
requirement_file = repo.joinpath('requirements.txt').__str__()
# Path to the requirements.txt file

cmd = None  # Placeholder for command input

def pip_available(path : str) -> str | None:
    """
    Check if pip is available in the specified path.

    This function attempts to locate and verify the presence of pip (or pip3) in the given path. If found, it returns the
    full path to the pip executable; otherwise, it returns None.

    Args:
        path (str): The directory path to check for pip.

    Returns:
        str | None: The full path to the pip executable if found, or None if pip is not available in the path.

    Example:
        pip_path = pip_available('/path/to/python/')
        if pip_path:
            print(f'Found pip at {pip_path}')
        else:
            print('pip is not available in the specified path.')
    """
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
    """
    Create a virtual environment and install requirements.

    This function creates a virtual environment, activates it, installs requirements using pip (if requested),
    and deactivates the environment.

    Args:
        env_type (Literal["conda"] | Literal["local"]): The type of environment to create (either "conda" or "local").

    Example:
        create_environment("local")  # Create a local virtual environment and install requirements.
    """
    try:
        path = repo.joinpath(env_file)
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


def unit_tester():
    """
        Execute a python main.py within the test directory to run all the unit tests

        Parameter:
            None
        
        Return :
            None
        
        Raises:
            subprocess.CalledProcessError: If the command execution fails.
            FileNotFoundError: If the file is not found.
    """
    try:
        subprocess.run([sys.executable, os.path.join(repo, 'test', 'main.py')])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("An error occurred while running a unit tests.")

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
            print(f"\nSorry {getpass.getuser()} that command does not exist... try one of the command that do exist\n")
            cmd = input("Enter command: ").lower()
    except KeyboardInterrupt as key:
        sys.exit()
        
    function_commands(cmd)