# Adaptive Multi-Source Domain Collaborative Fine-tuning
![fig2](https://github.com/fengledl/AMCF/assets/152671236/12b2e2ef-8940-4a2b-8448-9c6dcfce5422)
# Abstract
Fine-tuning is an important technique in transfer learning that has achieved significant success in tasks that lack training data. However, as it is difficult to extract effective features for single-source domain fine-tuning when the data distribution difference between the source and the target domain is large, we propose a transfer learning framework based on multi-source domain called adaptive multi-source domain collaborative fine-tuning (AMCF) to address this issue. AMCF utilizes multiple source domain models for collaborative fine-tuning, thereby improving the feature extraction capability of model in the target task. Specifically, AMCF employs an adaptive multi-source domain layer selection strategy to customize appropriate layer fine-tuning schemes for the target task among multiple source domain models, aiming to extract more efficient features. Furthermore, a novel multi-source domain collaborative loss function is designed to facilitate the precise extraction of target data features by each source domain model. Simultaneously, it works towards minimizing the output difference among various source domain models, thereby enhancing the adaptability of the source domain model to the target data. In order to validate the effectiveness of AMCF, it is applied to seven public visual classification datasets commonly used in transfer learning, and compared with the most widely used single-source domain fine-tuning methods. Experimental results demonstrate that, in comparison with the existing fine-tuning methods, our method not only enhances the accuracy of feature extraction in the model but also provides precise layer fine-tuning schemes for the target task, thereby significantly improving the fine-tuning performance.
# Requirements
Python (3.8)

PyTorch (2.0.1)

# Dataset download link
The following information was supplied regarding data availability: The public datasets are available at:

1). CIFAR100--https://www.cs.toronto.edu/~kriz/cifar.html

A zipped repository is available at Kaggle:
https://www.kaggle.com/datasets/aymenboulila2/cifar100

2)CUB-200--https://www.vision.caltech.edu/datasets/cub_200_2011/

A zipped repository is available at Kaggle: 
https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images

3). MIT Indoor--https://web.mit.edu/torralba/www/indoor.html

A zipped repository is available at Kaggle:
https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019

4). Stanford Dogs--http://vision.stanford.edu/aditya86/ImageNetDogs/

A zipped repository is available at Kaggle:
https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

5). Caltech 256-30 and Caltech 256-60--https://data.caltech.edu/records/nyy15-4j048

A zipped repository is available at Kaggle:
https://www.kaggle.com/datasets/mmoreaux/caltech256

6). Aircraft--https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

A zipped repository is available at Kaggle:
https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft

7). UCF-101--https://www.robots.ox.ac.uk/~vgg/decathlon/#download

8). Omniglot--https://www.robots.ox.ac.uk/~vgg/decathlon/#download
