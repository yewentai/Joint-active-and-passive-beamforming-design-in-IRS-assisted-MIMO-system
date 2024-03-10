# Intelligent Reflecting Surface (IRS) Optimization and Simulation

This project explores the application of Intelligent Reflecting Surfaces (IRS) in improving communication systems' performance. It covers two main approaches: Deep Deterministic Policy Gradient (DDPG) based optimization and Projected Gradient Method (PGM) for optimizing the IRS's phase shifts to maximize the communication channel's capacity.

## Project Structure

The project is organized into two main directories, each targeting a specific approach for IRS optimization and simulation:

- `IRS_DDPG`: Contains Python scripts and a Jupyter notebook (`main.ipynb`) for the implementation of the DDPG algorithm for optimizing IRS-assisted communication systems.
- `IRS_Optimization`: Features Python scripts for generating channel matrices, optimizing phase shifts using DDPG and Projected Gradient Descent (PGD), and simulating the system's performance.
- `IRS_PGM`: MATLAB scripts for running simulations using the Projected Gradient Method, including channel matrix generation and optimization routines.

Additionally, utility scripts and data files are included for generating channel models, locations, and path loss calculations, facilitating comprehensive simulations and optimizations.

## Key Components

### IRS_DDPG

- `agent.py`: Defines the DDPG agent, including the actor and critic networks for learning the policy of phase shift adjustments.
- `channel.py`: Implements channel models and calculations necessary for IRS system simulations.
- `main.py`: The main script for running DDPG-based optimization and simulation.
- `outcome`: A directory containing output figures and simulation results in various formats (e.g., PDF, PNG).

### IRS_Optimization

- `generate_channel.py`, `generate_location.py`: Scripts for generating channel matrices and locations of transmitters, receivers, and the IRS.
- `optimize_ddpg.py`, `optimize_pgd.py`: Optimization scripts using DDPG and PGD methods.
- `simulation.py`: Runs simulations to evaluate the system performance under different configurations and optimization techniques.

### IRS_PGM

- MATLAB scripts (`PGM_run.m`, `generate_channel.m`, `optimize_pgd.m`) for running simulations and optimizations using the Projected Gradient Method.

## Getting Started

To run the simulations and optimizations in this project, you will need:

- Python 3.x with libraries: `numpy`, `matplotlib`, `torch`, `pickle`, `scipy`.
- MATLAB for the scripts in the `IRS_PGM` directory.

### Running DDPG-based Optimization

Navigate to the `IRS_DDPG` directory and execute the main script:

```bash
python main.py
```

Or, open the `main.ipynb` Jupyter notebook and run the cells sequentially.

### Running Optimization and Simulations in MATLAB

Open MATLAB, navigate to the `IRS_PGM` directory, and run the `PGM_run.m` script to start the simulation and optimization process using the Projected Gradient Method.

## Results and Analysis

The project includes various output files and plots in the `outcome` directory, illustrating the system's performance under different optimization methods. By comparing these results, you can analyze the effectiveness of IRS in improving communication system performance.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
