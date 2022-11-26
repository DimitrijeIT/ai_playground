# Setup
Create a folder in wsl (linux 20.03 recomended) and clone the repo

## Install Docker:
sudo apt get install docker

Start docker 
``` sudo dockerd ```

# Build container
cotnrol + shfit + p and then 'Dev Container: Rebuild Container'
It will take some time for first install 
This pull docker container with python and install evrything in requirements.txt

# Running
In WSL

Start docker 
``` sudo dockerd ```

Maybe a change of DISPLAY variable is need to run visual applications
Install xLaunch 
!!! export DISPLAY in VSCode terminal !!! 

https://www.evernote.com/shard/s35/sh/344c969e-e16b-0d0d-3793-38df77d06dee/e2b929271ff44be3a5a76ae322b811e4


Import could not be resolved 
Command Palette (Cmd/Ctrl+Shift+P) -> Python Select Interpreter
and changed it to one matching 'which python' on the command line.

