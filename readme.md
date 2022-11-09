# PU learning for audio signal enhancement (PULSE)
<a href="https://github.com/nobutaka-ito/pulse/blob/master/LICENSE">
  <img alt="license" src="https://img.shields.io/github/license/nobutaka-ito/pulse">
</a>

This code is a Pytorch implementation of *PU learning for audio signal enhancement (PULSE)*, a method for audio signal enhancement (SE) proposed in ["Audio signal enhancement with learning from positive and unlabelled data"](https://arxiv.org/abs/2210.15143) [1]. PULSE is based on a weakly supervised learning framework called *learning from positive and unlabelled data (PU learning)* [2]. 

With this code, you can run speech enhancement experiments using PULSE and train and evaluate speech enhancement models from scratch. This code also includes implementations of ordinary supervised learning and mixture invariant training (MixIT) [3] as baseline SE methods. If you find this code useful, please cite [1]:
```
@misc{Ito2022arXiv10PULSE,
  doi = {10.48550/ARXIV.2210.15143},
  url = {https://arxiv.org/abs/2210.15143},
  author = {Ito, Nobutaka and Sugiyama, Masashi},
  title = {Audio Signal Enhancement with Learning from Positive and Unlabelled Data},
  howpublished = {arXiv},
  year = {2022},
}
```

## What's PULSE?
SE is the task of extracting a desired class of sounds (a "clean signal") while suppressing the other classes of sounds ("noise") from an observed mixture of them (a "noisy signal"). Applications include automatic speech recognition (ASR), music information retrieval, and sound event detection. Although the mainstream SE approach is supervised learning, it is physically impossible to record required parallel training data consisting of both noisy signals and the corresponding clean signals. These data are thus synthesised in practice, which can severely degrade real-world performance due to a data mismatch. 

In contrast, PULSE [1] is an SE method using non-parallel training data consisting of noisy signals and noise, which can be easily recorded in the real world. PULSE is based on PU learning [2], a framework for weakly supervised learning from positive and unlabelled data. 


## Set up
```
# Clone the repo
git clone https://github.com/nobutaka-ito/pulse.git
cd pulse

# Make a conda environment
conda create -n pulse python=3.8.5
conda activate pulse

# Install requirements
conda install pytorch==1.9.0 torchaudio==0.9.0 scipy==1.7.3 -c pytorch -c conda-forge
```

## Quick start
Here is how to quickly run a speech enhancement experiment using PULSE with publicly available speech [4] and noise datasets [5]. 

**NB:** This is intended to be a preliminary experiment to show the feasibility of SE using non-parallel training data consisting of noisy signals and noise through PULSE. As such, here we focus on synthetic data created from the speech [4] and the noise datasets [5] instead of real data. This facilitates evaluation because most evaluation metrics for speech enhancement, including scale-invariant SNR (SI-SNR), require parallel data, which cannot be recorded in the real world but can only be synthesised. Evaluation on real data without parallel data using ASR performance or the non-intrusive DNSMOS [6] will be covered in upcoming updates.

### Download speech and noise datasets
The following command will download the speech [4] and the noise datasets [5] from the web and put the former in `pulse/voicebank/` and the latter in `pulse/DEMAND/`.
```
bash download.sh
```

### Run an experiment
The following commands will generate a speech enhancement dataset, train a convolutional neural network (CNN) for speech enhancement using each method, evaluate the trained CNN in terms of the SI-SNR improvement (SI-SNRi) on the test set, and display the result. A single GPU is used. 

```
### PULSE
python pulse.py --method 'PU' --lr_per_GPU 0.000037 --blocks 2

### Supervised learning
python pulse.py --method 'PN' --lr_per_GPU 0.0002  --blocks 3

### MixIT
python pulse.py --method 'MixIT' --lr_per_GPU 0.000034 --blocks 2
```

Here, the option `--lr_per_GPU` specifies the learning rate per GPU. The option `--blocks` is related to the number of layers, where `--blocks 2` corresponds to a 7-layer CNN and `--blocks 3` to a 9-layer one. These hyperparameters have already been tuned using the SI-SNRi on the validation set.

## Using datasets of your own choice
Here we explain how to run an experiment using a clean speech dataset of your own choice using the TIMIT dataset [7] (as in [1]) as an example. You can also use a noise dataset of your choice in a similar way.

### Prepare TIMIT
Put TIMIT in `pulse/TIMIT/`, which you need to purchase from [LDC](https://catalog.ldc.upenn.edu/LDC93S1). If you wish to use a speech dataset other than TIMIT (or a noise dataset of your choice), put it in `pulse/<dataset name>/`, where `<dataset name>` should be replaced by the name of the dataset.

### Prepare configuration files
Configuration files include the information necessary to synthesise speech enhancement datasets from the speech and the noise datasets, including how to partition data into training, validation, and test sets. For the current example using TIMIT and DEMAND, we already prepared `pulse/TIMIT_train_set.txt`, `pulse/TIMIT_val_set.txt`, and `pulse/TIMIT_test_set.txt` for the training, the validation, and the test sets, respectively. However, if you wish to use a speech dataset other than TIMIT (or the default one [4]) or a noise dataset of your choice, you will need to create configuration files by yourself.

Each line of a configuration file for the training set (e.g., `pulse/TIMIT_train_set.txt`) includes the information necessary to synthesise a training example (i.e., a noisy speech example and a noise example in PULSE and MixIT and a noisy speech example and the corresponding clean speech example in supervised learning). Specifically, each line consists of the following six entries separated by a space: 
1. the file path for the clean speech for generating a noisy signal example,
1. the file path for the noise for generating a noisy signal example,
1. the starting time of the noise excerpt for generating a noisy signal example,
1. the signal-to-noise ratio (SNR) for generating a noisy signal example,
1. the file path for the noise for generating a noise example,
1. the starting time of the noise excerpt to be used as a noise example.

Here, the information in 5. and 6. is ignored in supervised learning. The configuration files for the validation and the test sets are the same except that 5. and 6. are not used and thus omitted.

**NB:** When creating configuration files, one should make sure that test examples are unseen in the training and the validation sets.

### How data are generated
Here is how speech enhancement datasets are generated using the speech and the noise datasets and the information in the configuration files. Each training example for PULSE and MixIT is generated according to the following procedure:
1. Generate a noisy signal example by excerpting a noise segment from the specified noise file starting from the specified starting time and mix it with the specified clean speech at the specified SNR.
1. Generate a noise example by excerpting a noise segment from the specified noise file starting from the specified starting time.

Each training example for supervised learning and each validation/test example for all methods is generated as in step 1 above without step 2.

### Run an experiment
The following commands will run an experiment with TIMIT. 

```
# PULSE
python pulse.py --method 'PU' --lr_per_GPU 0.000037 --train_fname 'TIMIT_train_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT'  --blocks 4 --fcblocks 1 --prior 0.7 

# Supervised learning
python pulse.py --method 'PN' --lr_per_GPU 0.0002 --train_fname 'TIMIT_train_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT'  --blocks 4 --fcblocks 0 --prior 0.7

# MixIT
python pulse.py --method 'MixIT' --lr_per_GPU 0.000034 --train_fname 'TIMIT_train_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT'  --blocks 4 --fcblocks 0 --prior 0.7
```

Here, the options `--train_fname`, `--val_fname`, and `--test_fname` specify the paths to the configuration files and the option `--clean_path` specifies the directory path of the clean speech dataset (i.e., TIMIT here). (If you also wish to use a noise dataset of your choice, you will also need to specify the directory path of the noise dataset using the option `--noise_path`.) The option `--fcblocks` is a hyperparameter related to the CNN architecture and `--prior` is the class prior for the positive class in PU learning, which have already been tuned using the SI-SNRi on the validation set.

## Multi-node processing
This code also supports multi-node data-parallel distributed training using `torch.nn.parallel.DistributedDataParallel` and [Slurm](https://slurm.schedmd.com/documentation.html). Here we show an example of using three NVIDIA DGX nodes with eight A100 GPUs each (i.e., 24 GPUs in total). 

```
# PULSE
srun -p <partition name> -N 3 --ntasks-per-node 8 --gpus-per-node 8 --cpus-per-task 10 --hint nomultithread python pulse.py --dist --prefix '<partition name>' --method 'PU' --lr_per_GPU 0.000037 --train_fname 'TIMIT_train_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT' --blocks 4 --fcblocks 1 --prior 0.7

# Supervised learning
srun -p <partition name> -N 3 --ntasks-per-node 8 --gpus-per-node 8 --cpus-per-task 10 --hint nomultithread python pulse.py --dist --prefix '<partition name>' --method 'PN' --lr_per_GPU 0.0002 --train_fname 'TIMIT_train_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT' --blocks 4 --fcblocks 0 --prior 0.7

# MixIT
srun -p <partition name> -N 3 --ntasks-per-node 8 --gpus-per-node 8 --cpus-per-task 10 --hint nomultithread python pulse.py --dist --prefix '<partition name>' --method 'MixIT' --lr_per_GPU 0.000034 --train_fname 'TIMIT_train_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT' --blocks 4 --fcblocks 0 --prior 0.7
```
You need to pass the options `--dist` and `--prefix` to pulse.py, where `--dist` is a flag indicating that distributed training is activated and `--prefix` specifies the partition name. `<partition name>` should be replaced with your partition name. For example, if your nodes are `node[01-03]`, `<partition name>` is `node`. The options for Slurm `srun` should be changed according to your cluster configuration.

## Options
### --p
The option `--p` specifies the exponent for the weight in the weighted sigmoid loss (see the paper [1] for details). The default value is `1.`, corresponding to weighting with the magnitude spectrogram of the observed noisy speech. On the other hand, it reduces to the unweighted sigmoid loss [8] with `--p .0` (as in an ablation study in [1]):
```
srun -p <partition name> -N 3 --ntasks-per-node 8 --gpus-per-node 8 --cpus-per-task 10 --hint nomultithread python pulse.py --dist --prefix '<partition name>' --p 0.0 --method 'PU' --lr_per_GPU 0.000037 --train_fname 'TIMIT_train_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT' --blocks 4 --fcblocks 1 --prior 0.7
```

### --mode
The option `--mode` specifies the type of the empirical risk in PU learning (`"nn"` or `"unbiased"`). The default value is `"nn"`, corresponding to the non-negative empirical risk in [8]. On the other hand, `--mode "unbiased"` corresponds to the unbiased empirical risk [2] (as in an ablation study in [1]):
```
srun -p <partition name> -N 3 --ntasks-per-node 8 --gpus-per-node 8 --cpus-per-task 10 --hint nomultithread python pulse.py --dist --prefix '<partition name>' --mode 'unbiased' --method 'PU' --lr_per_GPU 0.000037 --train_fname 'TIMIT_train_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT' --blocks 4 --fcblocks 1 --prior 0.7
```

### Other options
For other options, see help by `python pulse.py --help`.

## Code structure
`pulse.py` is the entrance of the program. The other files are explained in the following:

- `dataset.py` is used for creating data loaders.
- `model.py` is used for constructing models.
- `loss.py` is used for computing losses.
- `metric.py` is used for computing evaluation metrics.
- `distributed.py` is used for distributed training.
- `download.sh` is used for downloading speech and noise datasets.
- `*_set.txt` are configuration files and used for generating speech enhancement datasets.

## Citation
If you find this code useful, please cite [1]:
```
@misc{Ito2022arXiv10PULSE,
  doi = {10.48550/ARXIV.2210.15143},
  url = {https://arxiv.org/abs/2210.15143},
  author = {Ito, Nobutaka and Sugiyama, Masashi},
  title = {Audio Signal Enhancement with Learning from Positive and Unlabelled Data},
  howpublished = {arXiv},
  year = {2022},
}
```

## References
[1] N. Ito and M. Sugiyama, "Audio signal enhancement with learning from positive and unlabelled data," *arXiv*, https://arxiv.org/abs/2210.15143.

[2] M. Sugiyama, H. Bao, T. Ishida, N. Lu, T. Sakai, and G. Niu, *Machine Learning from Weak Supervision: An Empirical Risk Minimization Approach.* Cambridge, MA, USA: MIT Press, 2022.

[3] S. Wisdom, E. Tzinis, H. Erdogan, R. Weiss, K. Wilson, and J. Hershey, "Unsupervised sound separation using mixture invariant training," in *Proc. NeurIPS*, online, Dec. 2020.

[4] C. Valentini-Botinhao, X. Wang, S. Takaki, and J. Yamagishi, "Investigating RNN-based speech enhancement methods for noise-robust text-to-speech," in *Proc. ISCA Sp. Synth. Worksh.*, Sunnyvale, CA, USA, Sep. 2016, pp. 146-152.

[5] J. Thiemann, N. Ito, and E. Vincent, "The diverse environments multi-channel acoustic noise database (DEMAND): A database of multichannel environmental noise recordings," in *Proc. ICA*, Montreal, Canada, Jun. 2013.

[6] C. K. A. Reddy, V. Gopal, and R. Cutler, "DNSMOS: A nonintrusive perceptual objective speech quality metric to evaluate noise suppressors," in *Proc. ICASSP*, Canada, Jun. 2021, pp. 6493–6497.

[7] W. Fisher, G. Doddington, and K. Goudie-Marshall, "The DARPA speech recognition research database: specifications and status," in *Proc. DARPA Speech Recognition Workshop*, 1986, pp. 93–99.

[8] R. Kiryo, G. Niu, M. C. du Plessis, and M. Sugiyama, "Positive-unlabeled learning with non-negative risk estimator," in *Proc. NIPS*, CA, USA, Dec. 2017.
