# semi-supervised_learning_for_affordance_detection
semi-supervised learning for affordance detection with pytorch

# requirements
* python 3
* pytorch >= 0.4
* tensorboardx
* scipy, tqdm, pyyaml, addict


# dataset
use the following dataset:

[Part Affordance Dataset](http://users.umiacs.umd.edu/~amyers/part-affordance-dataset/)

Affordance Detection of Tool Parts from Geometric Features,  
Austin Myers, Ching L. Teo, Cornelia Fermüller, Yiannis Aloimonos.  
International Conference on Robotics and Automation (ICRA). 2015.  


# training on Part Affordance Dataset
please `run train.py` in the command line after downloading the dataset in 'part-affordance-dataset' directory.

if you run`python train.py -h`, you can see the help.
to run this code needs a configuration file written in .yaml format.


# added date
Dec. 20, 2018
