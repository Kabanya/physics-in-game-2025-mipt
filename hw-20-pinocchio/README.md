
### Physics in Game 2025 - Humanoid Robot Walking

#### Getting Started

### 1) You need to install Miniconda. 
**Download Miniconda**: https://www.anaconda.com/docs/getting-started/miniconda/main

### 2) Create a conda environment and install the required libraries:

2.1) Let's create a conda environment:
```bash
conda create -n pin-env
```
OR

```bash
conda create --name pin-env python=3.12.3   
```

---
2.2) Activate pin-env:

```bash
conda activate pin-env
```
OR acrivate pin-env via your IDE.

---

2.3) Let's install the necessary libraries:
```bash
conda install scipy 
conda install meshcat
conda install -c conda-forge matplotlib
conda install -c conda-forge pinocchio 
conda install -c conda-forge example-robot-data
```

**optional**:
``` bash
conda install decorator
```

2.4*) If something doesn't work, you can always try just with -c conda-forge (`conda install scipy`) or without it (`conda install -c conda-forge matplotlib`). If nothing works, then `pip install ...`.

### 3) Start main.py and wait.

Running the Project.
Activate the environment:

```bash
conda activate pin-env
```
```bash
cd /path/to/physics-in-game-2025-mipt/hw-20-pinocchio/
```
```bash
python main.py
```

Wait for the simulation to complete - the robot will perform a walking sequence.

### 4) Also you have option to run _talos_CoM_Zmp.ipynb 
with a description of what's going on (CoP, Zmp, CoM). 
Choose pin-env in you IDE and run the cells.


### 5) To remove pin-env:

Let's take a look at existing environments:

```bash
conda env list
```

We can delete pin-env:

```bash 
conda env remove -n pin-env
```

### Troubleshooting: Modules not installed

https://github.com/microsoft/vscode-jupyter/wiki/Failure-to-start-Kernel-due-to-Modules-not-installed