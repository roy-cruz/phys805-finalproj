# PHYS805 Final Project

This repository hosts the source code for the final project for the Physic 805: Research Methods in Machine Learning & Physics Fall 2025 course. In this project, we aimed to classify events between emerging jet events and events with just QCD jets. 

## Repo Structure

Most of the tools developed and used for this project are stored in the [`utils`](./utils/) directory. The following files are found here:

- [`model.py`](./utils/model.py): Hosts the classes for the models used in this project. This includes the transformer encoder and the classifier.
- [`dataloader.py`](./utils/dataloader.py): Stores the `JetDataset` loader used to handle jet data.
- [`training.py`](./utils/training.py): Tools used exclusively during training.
- [`metrics.py`](./utils/metric.py): Tools for evaluating the performance of the models.
- [`data_utils.py`](./utils/data_utils.py): Misc. data related utilities, such as a function for loading data from a `.root` file, or a function that compute the neccesary feature normalization constants.

These tools are all used in the project notebook [`finalproj_roycruz.ipynb`](./finalproj_roycruz.ipynb). Finally, the root files used are specified in [`datasets.yaml`](./datasets.yaml).

## Setup

This project is meant to be run with the `CMSSW_14_0_22` software toolkit. The original setup procedure consisted of running

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_14_0_22
cd CMSSW_14_0_22/src && cmsenv && cd -
pip3 install --no-deps torch wandb
```

However, the virtual environment can be reproduced by using the included [`requirements.txt`](./requirements.txt). Installing the required packages using this file can be done with the following commands.

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

To access the samples in order to re-run the notebook, you will need a valid CERN grid certificate and to be a member of CMS. Instructions on how to setup your certificate as well as the initialization of the VOMS proxy can be found in this [TWiki page](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookStartingGrid#ObtainingCert).