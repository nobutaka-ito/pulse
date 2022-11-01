#!/bin/bash

wget https://datashare.ed.ac.uk/download/DS_10283_1942.zip
unzip DS_10283_1942.zip
mkdir voicebank
unzip -d voicebank clean_testset_wav.zip
unzip -d voicebank/clean_trainset_wav clean_trainset_wav.zip

wget https://zenodo.org/record/1227121/files/DKITCHEN_16k.zip
unzip -d DEMAND DKITCHEN_16k.zip

wget https://zenodo.org/record/1227121/files/DLIVING_16k.zip
unzip -d DEMAND DLIVING_16k.zip

wget https://zenodo.org/record/1227121/files/DWASHING_16k.zip
unzip -d DEMAND DWASHING_16k.zip

wget https://zenodo.org/record/1227121/files/NFIELD_16k.zip
unzip -d DEMAND NFIELD_16k.zip

wget https://zenodo.org/record/1227121/files/NRIVER_16k.zip
unzip -d DEMAND NRIVER_16k.zip

wget https://zenodo.org/record/1227121/files/OHALLWAY_16k.zip
unzip -d DEMAND OHALLWAY_16k.zip

wget https://zenodo.org/record/1227121/files/OOFFICE_16k.zip
unzip -d DEMAND OOFFICE_16k.zip

wget https://zenodo.org/record/1227121/files/STRAFFIC_16k.zip
unzip -d DEMAND STRAFFIC_16k.zip

wget https://zenodo.org/record/1227121/files/TCAR_16k.zip
unzip -d DEMAND TCAR_16k.zip
