# DeepDeWedge

This repository contains an implementation of the DeepDeWedge method as described in our manuscript ["A Deep Learning Method for Simultaneous Denoising and Missing Wedge Reconstruction in Cryogenic Electron Tomography"](https://arxiv.org/abs/2311.05539). Our implementation comes as a Python package with an accompanying command line interface.

## Installation
The first step is to clone this repository, e.g. via
```
git clone https://github.com/MLI-lab/DeepDeWedge
```
We recommend to continue the installation in a fresh `Python 3.10.13` environment. To create such an environment, you can for example use [Anaconda](https://www.anaconda.com/download):
```
conda create -n ddw_env python=3.10.13
conda activate ddw_env
```
Next, you have to install a version of `PyTorch` that is compatible with your `CUDA` version. DeepDeWedge was developed using `Pytorch 2.2.0` and `CUDA 11.8`, so we recommend this combination. The corresponding `conda` install command is
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
You can find a list of all `PyTorch` versions and the compatible `CUDA` versions [here](https://pytorch.org/get-started/previous-versions/). The remaining requirements can be istalled via
```
pip install -r requirements.txt
```
Finally, you can, install the DeepDeWedge package via
```
pip install -e .
``` 
The installation should not take more than a few minutes in total. Upon successful installation, running the command
```
ddw --help
```
should display a help message for the DeepDeWedge command line interface.

## Usage
The DeepDeWedge command line interface provides three commands which correspond to the three steps of the algorithm outlined in our paper:
- `ddw prepare-data`: Extracts cubic sub-tomograms used to generate model inputs and targets for model fitting.
- `ddw fit-model`: Fits a U-Net for denoising and missing wedge reconstruction on data generated based on the output of the `prepare-data` command.
- `ddw refine-tomogram`: Refines one or more tomograms using a fitted model.

Each command has its own set of options which can be displayed by running `ddw <command> --help`. Moreover, all commands are also available as Python functions inside the `ddw` package. 

We encourage developers, researchers and interested users to have a look at the `ddw.utils` package which contains most of the actual implementation of DeepDeWedge. 

## Tutorial
To get started with the DeepDeWedge command line interface, we strongly encourage you to have a look at our tutorial Jupyter notebooks in the `tutorial/` directory. There, we reconstruct the flagella of Chlamydomonas Reinhardtii based on data from the Tomo110 dataset, which was used in the tutorial for the related [CryoCARE](https://github.com/juglab/cryoCARE_T2T) denoising method.


## Contact

If you have any questions, or if you think you found a bug in the code, please do not hesitate to [contact us](mailto:simonw.wiedemann@tum.de).

## Citation

```
@article{wiedemann2023deep,
  title   = {A Deep Learning Method for Simultaneous Denoising and Missing Wedge Reconstruction in Cryogenic Electron Tomography},
  author  = {Wiedemann, Simon and Heckel, Reinhard},
  journal = {arXiv preprint arXiv:2311.05539},
  year    = {2023}
}
```

## License
All files are provided under the terms of the BSD 2-Clause license.