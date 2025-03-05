# QBtoWEIS

**QBtoWEIS** is an extension of the WEIS (Wind Energy with Integrated Servo-control) framework that integrates the capabilities of QBlade to expand the design and optimization process for floating offshore wind turbines. By adding computationally highly optimized methods for wake aerodynamics and structural modeling, QBtoWEIS creates an even more versatile and powerful multi-fidelity toolchain, offering greater flexibility in co-design and optimization.

## WEIS

[WEIS](https://github.com/WISDEM/WEIS) is a comprehensive framework developed by the National Renewable Energy Laboratory (NREL) that integrates several NREL-developed tools for the design optimization of floating offshore wind turbines. It facilitates aero-servo-hydro-elastic modeling, structural analysis, and control system design, enabling a holistic approach to floating wind turbine development.

## QBlade

[QBlade](https://qblade.org/) is a versatile aero-servo-hydro-elastic simulation tool for wind turbines. It features a highly optimized lifting-line free vortex wake algorithm, which ensures efficient and accurate wake modeling. The underlying structural model implemented in QBlade supports Euler-Bernoulli, Timoshenko, and geometrically exact beam theory, all of which provide computationally efficient structural models. Notably, all three approaches can be associated with similar computational costs, allowing for flexibility in model selection without sacrificing performance.

[QBlade documentation](https://docs.qblade.org/)

## Additional Packages

QBtoWEIS integrates the following packages in addition to the stack of tools already available in WEIS:
* [QBlade](https://qblade.org/) - freely available wind turbine simulation tool

## Additional Packages (coming soon)

* [SONATA](https://github.com/ptrbortolotti/SONATA) - toolbox for Multidisciplinary Rotor Blade Design Environment for Structural Optimization and Aeroelastic Analysis

## Installation

The installation process is almost equivalent to the one of the main branch of WEIS:

On laptop and personal computers, installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis. WEIS requires [Anaconda 64-bit](https://www.anaconda.com/distribution/). However, the `conda` command has begun to show its age, and we now recommend the one-for-one replacement with the [Miniforge3 distribution](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3), which is much more lightweight and more easily solves for the package dependencies. Sometimes, using `mamba` in place of `conda` with this distribution speeds up the installation process. QBtoWEIS is currently supported on Linux, Windows Subsystem for Linux (WSL2) and Windows.

The installation instructions below use the environment name, "qbweis-env," but any name is acceptable. For those working behind company firewalls, you may have to change the conda authentication with `conda config --set ssl_verify no`. Proxy servers can also be set with `conda config --set proxy_servers.http http://id:pw@address:port` and `conda config --set proxy_servers.https https://id:pw@address:port`.

1. Clone the test-develop repository and create a virtual environment install the software:
   
        conda config --add channels conda-forge
        git clone -b test-develop https://github.com/gctroise/QBtoWEIS-CM.git qbweis-env
        cd QBtoWEIS-CM


2. Add in final packages and install the software

        conda install -y petsc4py mpi4py pyoptsparse     # (Mac / Linux only, sometimes Windows users may need to install mpi4py)
        pip install -e .

**NOTE:** To use QBtoWEIS-CM again after installation is complete, you will always need to activate the conda environment first with conda activate qbweis-env (or source activate qbweis-env).

## Download and configure QBladeCE

   Download QBladeCE from https://qblade.org/downloads/ and place it in your prefered directory # (Linux/WSL2 & Windows).
   
   install some libraries that are required to run qblade # (Linux/WSL2 only):
       	
        sudo apt-get update -y
        sudo apt-get install -y libqt5opengl5 libqt5xml5 libquadmath0 libglu1-mesa
        
   Make QBladeCE executable
   
        chmod +x run_qblade.sh
        chmod +x QBladeCE_x.y.z # x.y.z should be replaced by the actual version number, e.g. 2.0.8.1 

**NOTE:** QBtoWEIS-CM requires QBladeCE/QBladeEE version 2.0.8 or newer

## Instructions for Running Simulations/Optimizations with QBalde

   Before running simulations or optimizations using QBalde within WEIS, you must configure the paths to the necessary shared library files (.dll for Windows or .so for Linux/WSL2) within the `modeling_options.yaml` file of your WEIS problem. For example: `qb_examples\00_run_test\modeling_options.yaml`.

### **Linux/WSL2**
   Specify both `path2qb_libs` and `path2qb_dll` as shown below:

        path2qb_libs: /home/user/qblade/software/QBladeCE_2.0.8.3/Libraries
        path2qb_dll: /home/user/qblade/software/QBladeCE_2.0.8.3/libQBladeCE_2.0.8.3.so.1.0.0

### **Windows**
   in Windows only "path2qb_dll" has to be specified:

        path2qb_dll: C:\Users\User\QBladeCE_2.0.8.3\QBladeEE_2.0.8.3.dll

### **Default Path**
   To avoid modifying the path in every new WEIS problem you can specify your default path in the modeling_schema.yaml:
   
        QBtoWEIS-CM/weis/inputs/modeling_schema.yaml 
        
   "path2qb_libs" & "path2qb_dll" are found below the "qblade_configuration" object.


## Troubleshoot.
If you are having trouble creating the virtual environment try allocating more RAM to the WSL2 (e.g. https://learn.microsoft.com/en-us/answers/questions/1296124/how-to-increase-memory-and-cpu-limits-for-wsl2-win)

Verify if correct package versions were installed

        conda list openfast # (check if openfast version equals 3.5.2 otherwise uninstall and revert)
        pip uninstall openfast
        pip install wisdem==3.5.2
