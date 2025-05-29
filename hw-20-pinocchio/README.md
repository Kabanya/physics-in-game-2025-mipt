
### Physics in Game 2025 - Humanoid Robot Walking

#### Getting Started


**First**, you need to install Miniconda. **Download Miniconda**: https://www.anaconda.com/docs/getting-started/miniconda/main

**Second**, create a dedicated conda environment and install the required libraries:

```bash
conda create -n pin-env
conda install -c anaconda scipy 
conda install -c conda-forge matplotlib
conda install -c conda-forge meshcat
conda install pinocchio -c conda-forge 
```
3) Start main.py and wait.


Running the Project
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