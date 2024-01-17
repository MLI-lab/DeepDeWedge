# DeepDeWedge

This repository contains an implementation of the DeepDeWedge method as described in our manuscript ["A Deep Learning Method for Simultaneous Denoising and Missing Wedge Reconstruction in Cryogenic Electron Tomography"](https://arxiv.org/abs/2311.05539). 

## Installation
The first step is to clone this repository, e.g. via
```
git clone https://github.com/MLI-lab/DeepDeWedge
```
We recommend to continue the installation in a fresh `Python 3.7.13` environment. To create such an environment, you can for example use [Anaconda](https://www.anaconda.com/download):
```
conda create -n DDW python=3.7.13
conda activate DDW
```
Next, you have to install a version of `PyTorch` that is compatible with your `CUDA` version. DeepDeWedge was developed using `Pytorch 1.12.1` and `CUDA 11.3`, so we recommend this combination. The corresponding `conda` install command is
```
conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
```
**Note:** You can find a list of all `PyTorch` versions and the compatible `CUDA` versions [here](https://pytorch.org/get-started/previous-versions/). 

The remaining requirements can be istalled via
```
pip install -r requirements.txt
```

## Tutorial
To get started with DeepDeWedge, we strongly encourage you to have a look at our tutorial in `tutorial.ipynb`. There, we reconstruct the flagella of Chlamydomonas Reinhardtii based on data from the Tomo110 dataset, which was used in the tutorial for the related [CryoCARE](https://github.com/juglab/cryoCARE_T2T) denoising method.


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