# PU learning for audio signal enhancement (PULSE)
This code is a Pytorch implementation of *PU learning for audio signal enhancement (PULSE)*, a method for audio signal enhancement (SE) proposed in [1]. With this code, you can run speech enhancement experiments using PULSE and train and evaluate speech enhancement models from scratch. If you find this code useful, please cite [1]. 

SE is the task of extracting a desired class of sounds (a "clean signal") while suppressing the other classes of sounds ("noise") from an observed mixture of them (a "noisy signal"). Applications include automatic speech recognition (ASR), music information retrieval, and sound event detection. Although the mainstream SE approach is supervised learning, it is physically impossible to record required parallel training data consisting of both noisy signals and the corresponding clean signals. These data are thus synthesised in practice, which can severely degrade real-world performance due to a data mismatch. In contrast, PULSE enables SE using non-parallel training data consisting of noisy signals and noise, which can be easily recorded in the real world. PULSE is based on a weakly supervised learning framework called *learning from positive and unlabelled data (PU learning)* [2].

This code also includes implementations of ordinary supervised learning and mixture invariant training (MixIT) [3] as baseline methods. 

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

## Quick run
Here is how to quickly run a speech enhancement experiment using PULSE with publicly available speech [4] and noise datasets [5]. 

**NB:** This is intended to be a preliminary experiment to show the feasibility of SE using non-parallel training data consisting of noisy signals and noise through PULSE. As such, here we focus on synthetic data created from the speech [4] and the noise datasets [5] instead of real data. This simplifies evaluation because most evaluation metrics for speech enhancement, including scale-invariant SNR (SI-SNR), require parallel data, which cannot be recorded in the real world but can only be synthesised. Evaluation on real data without parallel data using ASR performance or the non-intrusive DNSMOS [6] will be covered in upcoming updates.

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

Here, the option `lr_per_GPU` specifies the learning rate per GPU. The option `blocks` is related to the number of layers, where `--blocks 2` corresponds to a 7-layer CNN and `--blocks 3` to a 9-layer one. These hyperparameters have already been tuned using the SI-SNRi on the validation set.

## Using datasets of your own choice
Here we explain how to run an experiment using a clean speech dataset of your own choice using the TIMIT dataset [7] (as in [1]) as an example. You can also use a noise dataset of your choice in a similar way.

### Prepare TIMIT
Put TIMIT in `pulse/TIMIT/`, which you need to purchase from [LDC](https://catalog.ldc.upenn.edu/LDC93S1). If you wish to use a speech dataset other than TIMIT (or a noise dataset of your choice), put it in `pulse/<dataset name>/`, where `<dataset name>` should be replaced by the name of the dataset.

### Prepare configuration files
Configuration files specify the information necessary to synthesise speech enhancement datasets from the speech and the noise datasets, including how to partition data into development, validation, and test sets. For TIMIT, we already prepared `pulse/TIMIT_dev_set.txt`, `pulse/TIMIT_val_set.txt`, and `pulse/TIMIT_test_set.txt` for the development, the validation, and the test sets, respectively. If you wish to use a speech dataset other than TIMIT (or a noise dataset of your choice), you will need to create configuration files by yourself.

<!---the clean speech file path, the noise file path for noisy speech data, the starting time of excerpt from the noise file for noisy speech data, the signal-to-noise ratio, the noise file path for noise data, and the starting time of excerpt from the noise file for noise data.--->

### Run an experiment
The following commands will run an experiment with TIMIT. 

```
# PULSE
python pulse.py --method 'PU' --lr_per_GPU 0.000037 --dev_fname 'TIMIT_dev_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT'  --blocks 4 --fcblocks 1 --prior 0.7 

# Supervised learning
python pulse.py --method 'PN' --lr_per_GPU 0.0002 --dev_fname 'TIMIT_dev_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT'  --blocks 4 --fcblocks 0 --prior 0.7

# MixIT
python pulse.py --method 'MixIT' --lr_per_GPU 0.000034 --dev_fname 'TIMIT_dev_set.txt' --val_fname 'TIMIT_val_set.txt' --test_fname 'TIMIT_test_set.txt' --clean_path 'TIMIT'  --blocks 4 --fcblocks 0 --prior 0.7
```

Here, the options `dev_fname`, `val_fname`, and `test_fname` specify the paths to the configuration files and the option `clean_path` specifies the directory path of the clean speech dataset (i.e., TIMIT here). (If you also wish to use a noise dataset of your choice, you will also need to specify the directory path of the noise dataset using the option `noise_path`.) The option `fcblocks` is a hyperparameter related to the CNN architecture and `prior` is a hyperparameter in PU learning (specifically the class prior for the positive class), which have already been tuned using the SI-SNRi on the validation set.

## References
[1] N. Ito and M. Sugiyama, "Audio signal enhancement with learning from positive and unlabelled data," arXiv, https://arxiv.org/abs/2210.15143.

[2] M. Sugiyama, H. Bao, T. Ishida, N. Lu, T. Sakai, and G. Niu, *Machine Learning from Weak Supervision: An Empirical Risk Minimization Approach.* Cambridge, MA, USA: MIT Press, 2022.

[3] S. Wisdom, E. Tzinis, H. Erdogan, R. Weiss, K. Wilson, and J. Hershey, "Unsupervised sound separation using mixture invariant training," in *Proc. NeurIPS*, online, Dec. 2020.

[4] C. Valentini-Botinhao, X. Wang, S. Takaki, and J. Yamagishi, "Investigating RNN-based speech enhancement methods for noise-robust text-to-speech," in *Proc. ISCA Sp. Synth. Worksh.*, Sunnyvale, CA, USA, Sep. 2016, pp. 146-152.

[5] J. Thiemann, N. Ito, and E. Vincent, "The diverse environments multi-channel acoustic noise database (DEMAND): A database of multichannel environmental noise recordings," in *Proc. ICA*, Montreal, Canada, Jun. 2013.

[6] C. K. A. Reddy, V. Gopal, and R. Cutler, "DNSMOS: A nonintrusive perceptual objective speech quality metric to evaluate noise suppressors," in *Proc. ICASSP*, Canada, Jun. 2021, pp. 6493–6497.

[7] W. Fisher, G. Doddington, and K. Goudie-Marshall, “The DARPA speech recognition research database: specifications and status,” in *Proc. DARPA Speech Recognition Workshop*, 1986, pp. 93–99.
