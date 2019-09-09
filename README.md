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
 Syndata+IC13+IC17:[Syndata+IC13+IC17 for baidu drive](https://pan.baidu.com/s/1L10JxOubAsW9CTO7xX4tlg)||      [Syndata+IC13+IC17 for google drive](https://drive.google.com/open?id=126743KYjbaUwZVWdgKhHiU61tR_7CpgC)   `(Note: the pre-trained model for 89.79% not 90.85%. I will upload it for 2 days later)`


## Training 

If you want to train for weak supervised:                                                                                                                 						    
1、You should first download the pre_trained model trained in the Syndata [baidu](https://pan.baidu.com/s/1MaznjE79JNS9Ld48ZtRefg)||[google](https://drive.google.com/file/d/1FvqfBMZQJeZXGfZLl-840YXoeYK8CNwk/view?usp=sharing).                                                                                                                                                      
2、change the data path and pre-trained model path.                                                                                                                                                         
3、run `python train.py`                                                                                                                                                                                    
`Note:I will give the clear instruction for training the Syndata,IC15 and IC15+IC17 tomorrow`
                                                                                                                    
**This code supprts for Syndata and icdar2015, and we will release the training code for IC13 and IC17 as soon as possible.**

Methods                    |dataset      |Recall      |precision      |H-mean
---------------------------|-------------|------------|---------------|------
Syndata                    |ICDAR13      |71.93%      |81.31%         |76.33%                                                                          
Syndata+IC15               |ICDAR15      |76.12%      |84.55%         |80.11%      
Syndata+IC13+IC17(deteval) |ICDAR13      |86.81%      |95.28%         |90.85%                                                 
                                                                                                                                          
                                                                                                                                                                                      
## Bad detection samples
**`There are our detection results with bad samples. We found that it's so terrible for detecting the big word. And the gaussian map can not split the character level gaussian region score map. We are trying to solve it, and any issues or advice are welcome.`**

<img src="https://github.com/backtime92/CRAFT-Reimplementation/blob/master/image/test1_score.jpg" width="384" height="512" /><img src="https://github.com/backtime92/CRAFT-Reimplementation/blob/master/image/test1_affinity.jpg" width="384" height="256" />                                 
<img src="https://github.com/backtime92/CRAFT-Reimplementation/blob/master/image/test2_score.jpg" width="384" height="512" /><img src="https://github.com/backtime92/CRAFT-Reimplementation/blob/master/image/test2_affinity.jpg" width="384" height="256" /> 


# Contributing to the project
`We will release training code as soon as possible， and we have not yet reached the results given in the author's paper. Any pull requests or issues are welcome. We also hope that you could give us some advice for the project.`



