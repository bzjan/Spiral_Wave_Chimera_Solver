
# Spiral Wave Chimera Solver

#### Contact information
Jan Totz,  <jantotz@itp.tu-berlin.de>


###  Requirements

* Linux Ubuntu    >= 16.04
* Boost           >= 1.58
* CUDA            >= 7.0
* python          >= 3.5



###  Installation  

#### 1) install boost
* download the latest version from: http://www.boost.org/users/download/  
* unzip tar.bz2 in $HOME/source:  
```
tar xvjf ~/Downloads/boost*.tar.bz2 -C ~/source
cd ~/source/boost*
./bootstrap.sh
sudo ./b2 install
```

#### 2) CUDA installation
(if you want to use GPU-accelerated code, otherwise skip to 3)  
* get .run file(s) (possible patches) from cuda repository https://developer.nvidia.com/cuda-toolkit  
* move both to ~/cuda  
```
chmod u+x ~/cuda/*
sudo service lightdm stop
sudo ./cuda_*.run --silent --override --toolkit --samples
```

#### 3) python installation
```
sudo apt-get install python3 python3-pip
sudo -H pip3 install --upgrade pip
sudo -H pip3 install jupyter numpy matplotlib
```

#### 4) program
for the CPU version:
```
cd SWC_CPU_solver
make
```


##  Usage

#### 1) prepare and run simulation
* modify <nameOfIni>.ini file in ini directory to your liking  
* run CPU program via
```
./SWC_CPU_solver.exe --ini ../ini/<nameOfIni>.ini
```

or run GPU-accelerated program via  
```
./SWC_GPU_solver.exe --ini ../ini/<nameOfIni>.ini
```

* for example:
```
./SWC_CPU_solver.exe --ini ../ini/chimera_spiral_zbke2k.ini
```

#### 2) to run the visulization, enter on the commandline:
```
jupyter notebook
```
and navigate to the directory with the .ipynb file  
* activate cells with shift+enter  
* enter name and continue
