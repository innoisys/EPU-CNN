# EPU-CNN
This is the official <b>Tensorflow</b> implementation of "E pluribus unum interpretable convolutional neural networks".

## Paper Abstract
The adoption of convolutional neural network (CNN) models in high-stake domains is hindered by their inability to meet 
society’s demand for transparency in decision-making. So far, a growing number of methodologies have emerged for 
developing CNN models that are interpretable by design. However, such models are not capable of providing 
interpretations in accordance with human perception, while maintaining competent performance. In this paper, 
we tackle these challenges with a novel, general framework for instantiating inherently interpretable CNN models, 
named E pluribus unum interpretable CNN (EPU-CNN). An EPU-CNN model consists of CNN sub-networks, each of which receives 
a different representation of an input image expressing a perceptual feature, such as color or texture. The output of an 
EPU-CNN model consists of the classification prediction and its interpretation, in terms of relative contributions of 
perceptual features in different regions of the input image. EPU-CNN models have been extensively evaluated on various 
publicly available datasets, as well as a contributed benchmark dataset. Medical datasets are used to demonstrate the 
applicability of EPU-CNN for risk-sensitive decisions in medicine. The experimental results indicate that EPU-CNN models 
can achieve a comparable or better classification performance than other CNN architectures while providing humanly 
perceivable interpretations.
