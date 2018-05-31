# CNN-online-incremental-learning

This repo provide a method for incremental leraning with large no labeled data in CNN. It makes softmax have distance judgement like method in face recognition.


## Method

Main idea is using a pre-trained model to label data, then using labeld data to train. As training step, unlabeled data is divided into many parts and one of it is uesed to incremental learning.In incremental learning, two values is online caculated to control to label raw data. One is α, the other is β. The α describes standard probility distribution in softmax. The β describes distance between α and a raw data output in one category.
