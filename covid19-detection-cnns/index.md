# COVID-19 Detection using CNNs

Detecting the presence of COVID-19 in a patient's lungs by analyzing chest X-Rays using CNNs.

<!--more-->

COVID-19 is an infection that is caused by the SARS-Cov2 virus. There are multiple variants of this virus, as the world has come to know it, but one particularly dangerous one is the Delta variant, which is known to cause damage to the lungs of its host. This infection can be spotted via a chest X-RAY scan. This scan is the basis for this project. Using a Convolutional Neural Network, these scans are analyzed for signs of COVID-19 infection. The tricky bit is differentiating between scans of patients with viral pneumonia and COVID-19. The latter stages of the COVID infection is brought on by a pneumonia like infection in the lungs. This project was implemented in the PyTorch framework.

## 1. The Dataset
The COVID-19 Radiograpy Database is a collection of chest X-RAYS of patients with the following conditions:
* COVID-19 infection: 3616 images
* Normal lungs: 10,192 images
* Non-COVID lung infections: 6012 images
* Viral pneumonia infection: 1345 images This database was created by researchers from Qatar Univeristy and Univeristy of Dhaka in collaboration with medical professionals in their countries as well as Malaysia and Pakistan.


## 2. Custom Dataset
I created a custom dataset class that is helpful while training and testing the model. This class inherits from `torch.utils.data.Dataset` and implements the `__getitem()__` method.

### Code
```python
    class ChestXRayDataset(torch.utils.data.Dataset):
        def __init__(self, image_dirs, transform):
            # transform obj is used to do data augmentation
            def get_images(class_name):
                images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png')]
                print(f'Found {len(images)} {class_name} examples')
                return images

            self.images = {}
            self.class_names = ['normal', 'viral', 'covid']

            for c in self.class_names:
                self.images[c] = get_images(c)

            self.image_dirs = image_dirs
            self.transform = transform

        def __len__(self):
            return sum([len(self.images[c]) for c in self.class_names])

        def __getitem__(self, index):
            class_name = random.choice(self.class_names)
            index = index % len(self.images[class_name])
            image_name = self.images[class_name][index]
            image_path = os.path.join(self.image_dirs[class_name], image_name)
            image = Image.open(image_path).convert('RGB')

            return self.transform(image), self.class_names.index(class_name)
```

## 3. Image Tranforms
A few preprocessing steps were added to the pipeline before the model could be trained. For the training images:
* Resizing: match input dimensions for pretrained ResNet18
* Augmentation: `RandomHorizontalFlip`
* Normalization

And for the test images:
* Resizing
* Normalization

Normalization was done separately to avoid `Data Leakage`.

## 4. The Model
The Convolutional Neural Network used for this project is a relatively light-weight residual network, ResNet18. This model was chosen for ease of transfer learning. The model was pretrained on the ImageNet dataset, which consists of images from over 1000 classes. 

![](resnet18.png "ResNet Model")

### Code
```python
      resnet18 = torchvision.models.resnet18(pretrained=True)
```

## 5. Training and Performance
The model was trained for a few epochs to fine-tune the weights to improve the prediction accuracy. The performance criteria was set to 95% accuracy and the model was able to do so in under 2 epochs. The training loop is highlighted below, for full code, please visit my GitHub repository for this project. 

### Code
```python
    def train(epochs):
        for e in range(epochs):
            train_loss = 0
            resnet18.train()

            # batch of images
            for train_step, (images, labels) in enumerate(dl_train):
                optimizer.zero_grad()
                outputs = resnet18(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if train_step % 20 == 0:
                    # perform validation calculations
                    resnet18.train()

                    if acc > 0.95:
                        print('Performance condition satisfied')
                        return
            train_loss /= (train_step + 1)
            print(f'Training loss: {train_loss:.4f}')
```

## 6. Results
With an accuracy of `95%`, this model can be used as a tool to detect COVID-19 in the lungs of patients. `It cannot, in any capacity, be a subsitute for a medical professional.`
