# DeepDeWedge

This repository contains an implementation of the DeepDeWedge method as described in our manuscript ["A Deep Learning Method for Simultaneous Denoising and Missing Wedge Reconstruction in Cryogenic Electron Tomography"](https://www.nature.com/articles/s41467-024-51438-y). Our implementation comes as a Python package with an accompanying command line interface.

## Updates

- **Multi-GPU model fitting** (2024-12-02): You can now use mutltiple GPUs for model fitting by setting the `gpu` argument in the `fit_model` section of your `yaml` config to a list of GPU indices. For more details, see the output of `ddw fit-model --help`.  **Note:** Currently, the `refine-tomogram` still only works with a single GPU. If `refine-tomogram` receives multiple GPU indices (e.g. through the `shared` field), it will print a warning and only use the first GPU.

## Installation
The first step is to clone this repository, e.g. via
```
git clone https://github.com/MLI-lab/DeepDeWedge
cd DeepDeWedge
```
We recommend to continue the installation in a fresh `Python 3.10.13` environment. To create such an environment, you can for example use [Anaconda](https://www.anaconda.com/download):
```
conda create -n ddw_env python=3.10.13 pip=23.2.1
conda activate ddw_env
```
Next, you have to install a version of `PyTorch` that is compatible with your `CUDA` version. DeepDeWedge was developed using `Pytorch 2.2.0` and `CUDA 11.8`, so we recommend this combination. The corresponding `conda` install command is
```
conda install pytorch==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
You can find a list of all `PyTorch` versions and the compatible `CUDA` versions [here](https://pytorch.org/get-started/previous-versions/). The remaining requirements can be istalled via
```
pip install -r requirements.txt
```
Finally, you can, install the DeepDeWedge package via
```
pip install .
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

Each command has its own set of arguments which can be displayed by running `ddw <command> --help`. All agruments can be either specified via the command line or by providing a `YAML` configuration file (see Tutorial). Moreover, all DeepDeWedge commands are available as Python functions inside the `ddw` package. 

We encourage developers, researchers and interested users to have a look at the `ddw.utils` package which contains most of the actual implementation of DeepDeWedge. 

## Tutorial
To get started with the DeepDeWedge command line interface, we strongly encourage you to have a look at our tutorial Jupyter notebooks in the `tutorial/` directory. There, we reconstruct the flagella of Chlamydomonas Reinhardtii based on data from the Tomo110 dataset, which was used in the tutorial for the related [CryoCARE](https://github.com/juglab/cryoCARE_T2T) denoising method.

## FAQ
If you have a question that is not answered here, please do not hesitate to [contact us](mailto:simonw.wiedemann@tum.de).

- **Q: When using my own data, the fitting and/or validation loss is very low or even close to zero. Is this a problem? What should I do about it?**\
  A: Low losses can be due to overall small voxel values in the tomogram and may cause instabilities during model fitting. If you observe very low losses (e.g. `1e-3` to `1e-9`) in the first epoch of model fitting, try standardizing your tomograms such that they have zero mean and unit variance before. You can do that manually, or by setting `standardize_tomograms: true` in the `shared` field of your config.

- **Q: How to speed up the model fitting process?**  
  A: There is a number of things you can try: 
    - **Smaller model:** You can try using a smaller U-Net. While this will reduce the expressiveness of the model, we have found that using a U-Net with 32 channels in the first layer provides similar results to a U-Net with the default 64 channels, but is signficantly faster to train. You can modify the number of channels by adjusting `chans` in the `unet_params_dict` argument.
    - **Manual early stopping:** While in general, you should fit the model until the fitting and/or validation losses converge or until the validation starts to increase, you can try to stop earlier. We found that DeepDeWedge often produces good results that do not change much anymore even if the fitting and/or validation losses are still decreasing. Therefore, we recommend to occasinally check the output of `ddw refine-tomogram` during fitting to see if the results are already satisfactory. However, be aware that reconstructions may still improve as long as the losses are decreasing.
    - **Faster dataloading:** If you notice that the GPU utilization fluctuates a lot during model fitting, you can increase the number of CPU workers for data loading by adjusting `num_workers`.
    - **Larger batches:** If you have a fast GPU with a lot of memory, you can try increasing the batch size by adjusting the `batch_size`.


- **Q: How large should the sub-tomograms for model fitting and tomogram refinement be?**\
  A: We have found that larger sub-tomograms give better results up to a point (see the Appendix of our paper).
  In most of our experiments, we used sub-tomograms of size 96x96x96 voxels, and we recommend not going below 64x64x64 voxels. \
  **Note**: The size of the sub-tomograms must be divisible by 2^`num_downsample_layers`, where `num_downsample_layers` is the number of downsample layers in the U-Net, e.g., for a U-Net with 3 downsample layers, the size of the sub-tomograms must be divisible by 8.

- **Q: How many sub-tomograms should I use for model fitting?**\
  A: So far, we have seen good results when fitting the default U-Net on at least 150 sub-tomograms of size 96x96x96 voxels. The smaller the sub-tomograms, the more sub-tomograms you should use, but we have not yet found a clear rule of thumb. You can increase/decrease the number of sub-tomograms by decreasing/increasing the three values in the `subtomo_extraction_strides` argument used in `ddw prepare-data`.
   

## Contact

If you have any questions or problems, or if you found a bug in the code, please do not hesitate to [contact us](mailto:simonw.wiedemann@tum.de) or to open an issue on GitHub.

## Citation

```
@article{wiedemann2024deep,
  title={A deep learning method for simultaneous denoising and missing wedge reconstruction in cryogenic electron tomography},
  author={Wiedemann, Simon and Heckel, Reinhard},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={8255},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## License
All files are provided under the terms of the BSD 2-Clause license.
