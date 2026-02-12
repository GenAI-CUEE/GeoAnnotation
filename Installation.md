## Installation

1. Install pixi in the user directory

```
user@~$ curl -fsSL https://pixi.sh/install.sh | sh
```
or
```
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

2. Go to the working directory and set the pixi environment.

```
user@~$ cd workspace$
user@~/workspace$ pixi init geo
user@~/workspace$ cd geo
```

3. Configure pixi.toml by following the instruction from https://samgeo.gishub.org/installation/#3-configure-pixitoml 

For example, if your computer do not have a GPU, you may install using the following option for CPU:

```
[workspace]
channels = ["https://prefix.dev/conda-forge"]
name = "geo"
platforms = ["linux-64", "win-64"]

[dependencies]
python = "3.12.*"
pytorch-cpu = ">=2.7.1,<3"
segment-geospatial = ">=1.2.0"
sam3 = ">=0.1.0.20251211"
jupyterlab = "*"
ipykernel = "*"
libopenblas = ">=0.3.30"
```

4. Install the pixi environment. 

```
user@~/workspace/geo$ pixi install 
```

5. Clone the github repo inside the pixi environment: `geo`.

```
user@~/workspace/geo$ git clone https://github.com/GenAI-CUEE/GeoAnnotation.git
```

6. Open the cloned github folder in VSCode. Then, set the default python interpreter to the pixi environment.

- Clt+Shift+P. Under `Default interpreter path`, provide the python path from pixi

![Default interpreter path](./figs/setting_env_path.png)


7. Activate the pixi environment for python script and jupyter notebook kernel. 

![Activate environment](./figs/activate_env_path_as_kernel.png)

8. You may have to install additional packages under `geo`

```
user@~/workspace$ cd geo
user@~/workspace/geo$ pixi add segment-geospatial[samgeo2] --pypi
user@~/workspace/geo$ pixi add sam2 --pypi
user@~/workspace/geo$ pixi add fiona --pypi
user@~/workspace/geo$ pixi add plantcv --pypi  
```
 