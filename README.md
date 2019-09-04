# CRAFT-Reimplementation
## Reimplementation：Character Region Awareness for Text Detection Reimplementation based on Pytorch

## Character Region Awareness for Text Detection
Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee
(Submitted on 3 Apr 2019)

The full paper is available at: https://arxiv.org/pdf/1904.01941.pdf                                                         

## Install Requirements:                                                                                                        
1、PyTroch>=0.4.1                                                                                                                             
2、torchvision>=0.2.1 			                                                    																			                             
3、opencv-python>=3.4.2                                                                                                       
4、check requiremtns.txt                                                                                                      
5、4 nvidia GPUs(we use 4 nvidia titanX)                                                                                      

## pre-trained model:
Syndata:[Syndata for baidu drive](https://pan.baidu.com/s/1MaznjE79JNS9Ld48ZtRefg) ||     [Syndata for google drive](https://drive.google.com/file/d/1FvqfBMZQJeZXGfZLl-840YXoeYK8CNwk/view?usp=sharing)                                                                                                    
Syndata+IC15:[Syndata+IC15 for baidu drive](https://pan.baidu.com/s/19lJRM6YWZXVkZ_aytsYSiQ) ||      [Syndata+IC15 for google
 drive](https://drive.google.com/file/d/1k17GuBG_omT91tJoIMSlLrorYbLXkq4z/view?usp=sharing)                                         


## Training                                                                                                                     
**This code supprts for Syndata and icdar2015, and we will release the training code for IC13 and IC17 as soon as possible.**

Methods           |dataset      |Recall      |precision      |H-mean
------------------|-------------|------------|---------------|------
Syndata           |ICDAR13      |71.93%      |81.31%         |76.33%                                                                          
Syndata+IC15      |ICDAR15      |76.12%      |84.55%         |80.11%      
Syndata+IC13+IC17 |             |            |               |          


# Contributing to the project
`We will release training code as soon as possible， and we have not yet reached the results given in the author's paper. Any pull requests or issues are welcome. We also hope that you could give us some advice for the project.`

