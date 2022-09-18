# Facial Emotion Recognition

Classifying images of human faces into one of seven basic emotions.

<!--more-->

The goal of this project was to classify images of human faces into one of seven basic emotions. To do so, I used Convolutional Neural Networks as the advantage it had over its precedessors is that it can detect important features without human intervention. CNNs are computationally efficient as well due to the special convolution and pooling operations.

## 1. The Dataset
The FER-2013 dataset consists of 35,887 images, each of size 48x48 pixels. The data was split into 28,709 images for training and 7,178 images for testing.

## 2. The Model
A `ResNet50` architecture was used as the Convolutional Neural Network which was pretrained on the ImageNet dataset. This dataset has 1000 classes but for the case of facial emotion recognition, there are only 7. Hence, to fine tune the network, the shallower layers were frozen and the model was trained using the training set. The building of the fine-tuned network is shown below:

### Code
```python
     def fer_model_fine_tune(base_model, input_shape):
        fine_tune_start = 155

        # Freezing all layers before the fine_tune_start layer
        for layer in base_model.layers[: fine_tune_start]:
            layer.trainable = None

        # declaring the input
        inputs = Input(shape=input_shape)

        # obtaining image embeddings from the second last layer
        x = base_model(inputs, training=False)

        # adding global avg pooling
        x = GlobalAvgPool2D()(x)

        # including dropout with prob=0.2
        x = Dropout(rate=0.2)(x)

        # adding a flatten layer
        x = Flatten()(x)

        # adding the prediction layer with 7 units, softmax activation
        prediction_layer = Dense(units=7, activation='softmax')(x)

        # outputs
        outputs = prediction_layer

        model = Model(inputs, outputs)

        return model
```

## 3. Training and Performance
The final layer of the base pretrained ResNet50 was replaced with a 7 unit softmax activated layer. This model was trained for 20 epochs and it obtained an accuracy of 85.2%. To improve the performance on the FER dataset, freezing of some of the layers was done and retrained for 20 epochs. The accuracy increased from 85.2% to 86.3%.
```python
    loss, accuracy, precision, recall, auc = fer_model_2.evaluate(test_dataset)
    print('Accuracy: {}'.format(accuracy))
    113/113 [==============================] - 20s 176ms/step - loss: 1.6389 - accuracy: 0.8633 - precision: 0.6699 - recall: 0.0854 - auc: 0.7451
    Accuracy: 0.8633323907852173 
```
