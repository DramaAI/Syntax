from __future__ import annotations
import sys
import subprocess
from typing import Literal
from pathlib import Path
import os
import getpass

env_file = "syntax"
requirement_file = Path(__file__).parent.parent.absolute().joinpath('requirements.txt').__str__()
cmd = None


def create_and_install_environment(env_type : Literal["conda"] | Literal["local"]):
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
            print(activate_script)
            subprocess.run(["source", activate_script], shell=True)

            while cmd :=input("[y/n]").lower() :
                if cmd in ['y', 'n']:
                    break

            # Step 3: Install requirements using pip
            if cmd == 'y':
                subprocess.run([os.path.join(path, "bin", "pip"), "install", "-r", requirement_file])

            # Step 4: Deactivate the virtual environment
            subprocess.run(["deactivate"], shell=True)
        else:
            subprocess.run(['conda', 'create', '--name', env_file])

    except subprocess.CalledProcessError:
        print("An error occurred while creating the environment or installing requirements.")

# TODO: 
#   - Build Unit testing module
#   - Build Docker when we build the backend api
OPTIONS = {'rm': ["echo", f"'sorry I can do that {getpass.getuser()}...' - HAL-9000"],  
           'b' : ["echo", f"'sorry I can do that {getpass.getuser()}...' - HAL-9000"], 
           'r' : ["echo", f"'sorry I can do that {getpass.getuser()}...' - HAL-9000"], 
           'e' : ["echo", f"'sorry I can do that {getpass.getuser()}...' - HAL-9000"], 
           'l' : create_and_install_environment,
           'a' : create_and_install_environment,
           'u' : ["echo", f"'sorry I can do that {getpass.getuser()}...' - HAL-9000"], }

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
        if cmd in ["l", 'a']:
            OPTIONS[cmd]('local' if cmd == 'l' else 'conda')
        else:
            print(OPTIONS[cmd])
            result = subprocess.run(OPTIONS[cmd], text=True, capture_output=True, check=True, shell=True)
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the command: {e}")

def menu():
    print(f"What would you like to do?\n\
          \r\t-{'(rm)':^6}Remove Container\n\
          \r\t-{'(b)':^6}Build Container\n\
          \r\t-{'(r)':^6}Running Container\n\
          \r\t-{'(e)':^6}Enter Container\n\
          \r\t-{'(l)':^6}Local Build\n\
          \r\t-{'(a)':^6}Anaconda Build\n\
          \r\t-{'(u)':^6}Unit Testing\n\
          \r\t-{'(q)':^6}Quit Program\n")    

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