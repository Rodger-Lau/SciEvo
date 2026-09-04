# Neuro-inspired continual adaptation for data-scarce dynamic scientific systems
AI for Science has advanced rapidly in data-rich settings supported by large-scale observations, simulations or curated atlases. Yet many real-world scientific systems remain data-scarce, dynamically evolving and only partially observable. The key challenge is to update scientific knowledge from limited new evidence while retaining transferable structure learned from related domains. Here we introduce SciEvo, a neuro-inspired continual adaptation framework for data-scarce dynamic scientific systems. Inspired by human learning under uncertainty, SciEvo progressively extracts common motifs over scientific domains, regulates structural updates through policy-guided plasticity, and consolidates useful experience through memory-learning co-evolution. We evaluate SciEvo across nine datasets spanning macroscopic urban and environmental systems and microscopic brain-signal and molecular systems. In macroscopic settings, SciEvo supports robust forecasting and risk inference under sparse sensing, sensor failure, missing variables and temporal shifts. In microscopic settings, it adapts to new subjects, patient groups and chemical regimes of scarce labels and individualized distributions. Ablation and learning-behaviour analyses verify the critical roles of motif acquisition, selective updating and experience consolidation. These results demonstrate how fragmented cross-domain observations can be distilled into adaptive scientific priors under sparse and shifting evidence.

The code of SciEvo is released here. The experiments cover four different scientific systems: urban mobility system (NYC, CHI, SIP), environment system (Knowair), brain signal system (HGD, BCI2A, CHBMIT), and biological molecular system (BBBP, HIV). First, you should download 'data' in https://drive.google.com/drive/folders/1O0qG8428EuexbSgSGpBDo5GWQFXXJU7J?usp=sharing

## Urban mobility system

The implementation code of this system is provided in main_ST.py. The script for running the file is:

`python main_ST.py --dataset NYC --num_nodes 206`

`python main_ST.py --dataset CHI --num_nodes 220`

`python main_ST.py --dataset SIP --num_nodes 108`



## Environment system

The implementation code of this system is provided in main_Knowair.py. The script for running the file is:

`python main_Knowair.py`

where the parameters are set in `config.yaml`

## Brain signal system

The implementation code of this system is provided in main_HGD.py, main_BCI2a.py, and main_CHBMIT.py. The script for running the files is:

`python main_HGD.py`

`python main_BCI2a.py`
`python main_CHBMIT.py`

## Biological molecular system

The implementation code of this system is provided in main_BBBP.py, and main_HIV.py. The script for running the files is:

`python main_BBBP.py`

`python main_HIV.py`

The requirements of **brain signal system** are provided in requirements_brainsignals.txt, where the python version is 3.9.25.

The requirements of other systems are provided in requirements_others.txt, where the python version is 3.8.0
