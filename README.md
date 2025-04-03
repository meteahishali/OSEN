=============================

This repository includes the implentation of the methods in [Operational Support Estimator Networks](https://ieeexplore.ieee.org/abstract/document/10542410). OSENs are generic networks and they can be used in different support estimation problems. There are three applications that we consider in this repository: i. Support Estimation from CS Measurements, ii. Representation-based Classification, and iii. Learning aided CS for MRI Reconstruction.

M. Ahishali, M. Yamac, S. Kiranyaz and M. Gabbouj, "Operational Support Estimator Networks," in _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol. 46, no. 12, pp. 8442-8458, Dec. 2024, doi: 10.1109/TPAMI.2024.3406473.

![Example Results](/images/osen_mnist.png)

Software environment:
```
# Python with the following version and libraries.

conda create -n osen python=3.7.9

conda activate osen

conda install tensorflow-gpu=2.4.1

pip install numpy==1.19.2 scipy==1.6.2 scikit-learn==1.0.2

pip install tensorflow-addons==0.13.0

conda install matplotlib==3.5

conda install -c simpleitk simpleitk
```
```
MATLAB -> MATLAB R2019a.
```

Content:
- [Citation](#citation)
- [Application I: Support Estimation from CS Measurements](#application-i-support-estimation-from-cs-measurements)
- [Application II: Representation-based Classification](#application-ii-representation-based-classification)
- [Application III: Learning-aided CS for MRI Reconstruction](#application-iii-learning-aided-cs-for-mri-reconstruction)
- [References](#references)

## Citation

If you use method(s) provided in this repository, please cite the following paper:

```
@ARTICLE{ahishali,
  author={Ahishali, Mete and Yamac, Mehmet and Kiranyaz, Serkan and Gabbouj, Moncef},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Operational Support Estimator Networks}, 
  year={2024},
  volume={46},
  number={12},
  pages={8442-8458},
  keywords={Task analysis;Estimation;Kernel;Neurons;Training;Dictionaries;Sparse approximation;Support estimation;sparse representation;operational layers;compressive sensing;machine learning},
  doi={10.1109/TPAMI.2024.3406473}}

```

## Application I: Support Estimation from CS Measurements

In this experiments, we use MNIST dataset. These samples are an example of a sparse signal due to the fact that foreground is more predominant than the digits. Averaged sparsity ratio is indeed computed as 0.2 which is the ratio of the number of non-zero pixels to the total number of pixels in the dataset.

Training can be started by choosing the method, polynomial order, and measurement rate:

```
cd se_mnist/
python main.py --method OSEN1 --q 3 --mr 0.05
```

In the script, supported methods are the following: proposed operational layers with generative super neurons: ```OSEN1```, ```OSEN2```, non-linear compressive learning module with proxy mapping layer: ```NCL_OSEN1``` and ```NCL_OSEN2```, and localized kernels: ```OSEN1_local``` and ```OSEN2_local```. Traditional convolutional layer versions are also included in ```CSEN1``` and ```CSEN2```.

The model weights are provided in [weights_mnist.zip (click here)](https://drive.google.com/file/d/1xfaU8iPxWTP7vwzlj9lIGYRX4XM3c4i2/view?usp=sharing) file, please unzip the file and place it under ```se_mnist/weights_mnist/``` folder to reproduce the reported results in the paper. The evaluation of the methods can be performed similarly using provided weights,

```
python main.py --method OSEN1 --q 3 --mr 0.05 --weights True
```

## Application II: Representation-based Classification

For the prepration of OSEN input-output pairs, we need to run the following MATLAB script:

```
cd classification_yale/
run prepareData.m
```

Note that the script of prepareData.m produces estimations using the CRC-light model that is discussed in the paper. Once the CRC approach is run, the input and output pairs of OSENs are prepared and they are stored under ```OSENdata/```. Similar to Application I, the proposed classifier can be run as follows,

```
python main.py --method OSEN_C --q 3 --mr 0.01
```

Supported classifier methods are proposed operational layers with generative super neurons: ```OSEN_C```, non-linear compressive learning module with proxy mapping layer: ```NCL_OSEN_C```, localized kernels: ```OSEN_C_local```, and traditional convolutional layers: ```CSEN_C```.

The model weights for classification are provided in [weights_yale_b.zip (click here)](https://drive.google.com/file/d/1jc6nKwjUCMpdryzHPzG16VGVP5Z0Uy14/view?usp=sharing) file, please unzip the file and place it under ```classification_yale/weights_yale_b/``` folder to reproduce the reported results in the paper:

```
python main.py --method OSEN_C --q 3 --mr 0.01 --weights True
```

Compared Sparse Representation-based Classification (SRC) approaches are provided under ```classification_yale/src/```. They can be run as follows,

```
cd classification_yale/src/
run main_src.m
```
In the script, ```l1method``` can be set to different methods. There are implemented 8 different SRC algorithms including ADMM [1], DALM [2], OMP [2], Homotopy [3], GPSR [4], ℓ1-LS [5], ℓ1-magic [6], and PALM [2]:

```
l1method={'solve_ADMM','solve_dalm','solve_OMP','solve_homotopy','solveGPSR_BCm', 'solve_L1LS','solve_l1magic','solve_PALM'}; %'solve_PALM' is very slow
```

Classification using Collaborative Representation-based Classification (CRC) method [7] can be run as follows:
```
cd classification_yale/crc/
run main_crc.m
```

## Application III: Learning-aided CS for MRI Reconstruction

![Example Results](/images/mri_samples.png)

Please download  ```MICCAI-2013-SATA-Challenge-Data.zip ``` file by registering at https://www.synapse.org/#!Synapse:syn3193805/wiki/217780. After unzipping the file, place ```diencephalon``` folder under ```learning_aided_cs/diencephalon/```. We manually remove empty MRI slices in the dataset. You can find the indices of the selected slices in ```train_slices.npy``` and ```test_slices.npy``` files. To construct the dataset in NumPy format, you can run the following:

```
cd learning_aided_cs/
python nii_to_npy.py
```

This will read the data, select the corresponding slices, and finally create ```x_train.npy``` and ```x_test.npy``` files that are ready for the training and testing. Next, the support estimation networks can be run as,

```
python main_mri.py --method OSEN1 --q 3 --mr 0.05 --nu 1
```

Supported methods are proposed operational layers with generative super neurons: ```OSEN1```, ```OSEN2```, and localized kernels: ```OSEN1_local``` and ```OSEN2_local```. Traditional convolutional layer versions are ```CSEN1``` and ```CSEN2```. Note that instead of repeating the experiments for 5 times in MRI reconstruction, we specify the current number of run with ```--nu``` parameter. Because, TV-based minimization will require a lot of computational time in the next reconstruction stage.

The model weights are provided in [weights_mri.zip (click here)](https://drive.google.com/file/d/1EvBsWikpHHPHQuHg8ggB9VIoNmlSDedP/view?usp=sharing) file, please unzip the file and place it under ```learning_aided_cs/weights_mri/``` folder to reproduce the reported results in the paper:

```
python main_mri.py --method OSEN1 --q 3 --mr 0.05 --nu 1 --weights True
```

In above stage, we estimated probability maps for given MRI test samples. Now, we use the probability maps that are produced by the networks in learning-aided CS MRI scheme in the following script:

```
python main_mri_tv.py --learn_aid True --model_aid OSEN1 --q 3 --mr 0.05 --nu 1
```

If ```learn_aid``` is set to ```False```, then the baseline model, i.e., traditional TV-based minimization is run.

Note that since this computation is slow, one can specify the sample number explicitly to perform computation in parallel (for example using different work stations). For example, passing ```--sample 9``` runs reconstruction only for the 10th sample in the test set. A sample computationally efficient run procedure using Slurm environment is provided in ```parallel_run.sh``` script where each sample is reconstructed on a node in parallel. Correspondingly, we provide also ```analyze_se.py``` and ```analyze_cs.py``` scripts to calculate overall averaged support estimation and CS performances, respectively.


## References
[1] S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein et al., "Distributed optimization and statistical learning via the alternating direction method of multipliers," *Found. Trends Mach. Learn.*, vol. 3, no. 1, 2011. \
[2] A. Y. Yang, Z. Zhou, A. G. Balasubramanian, S. S. Sastry, and Y. Ma, "Fast l1-minimization algorithms for robust face recognition," *IEEE Trans. Image Process.*, vol. 22, no. 8, pp. 3234–3246, 2013. \
[3] D. M. Malioutov, M. Cetin, and A. S. Willsky, "Homotopy continuation for sparse signal representation," *in Proc. IEEE Int. Conf. Acoust., Speech, and Signal Process. (ICASSP)*, vol. 5, 2005, pp. 733–736. \
[4] M. A. Figueiredo, R. D. Nowak, and S. J. Wright, "Gradient projection for sparse reconstruction: Application to compressed sensing and other inverse problems," *IEEE J. Sel. Topics Signal Process.*, vol. 1, no. 4, pp. 586–597, 2007. \
[5] K. Koh, S.-J. Kim, and S. Boyd, "An interior-point method for large-scale l1-regularized logistic regression," *J. Mach. Learn. Res.*, vol. 8, pp. 1519–1555, 2007. \
[6] E. Candes and J. Romberg, "l1-magic: Recovery of sparse signals via convex programming," *Caltech, Tech. Rep.*, 2005. [Online]. Available: https://candes.su.domains/software/l1magic/downloads/l1magic.pdf. \
[7] L. Zhang, M. Yang, and X. Feng, "Sparse representation or collaborative representation: Which helps face recognition?" *in Proc. IEEE Int. Conf. Comput. Vision (ICCV)*, 2011, pp. 471–478.
