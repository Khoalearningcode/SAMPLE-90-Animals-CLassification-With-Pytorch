- Using model from torchvision.models to train and classify 90 animals. Each animal has 40 images for train, 10 images for test, and 10 images for validation.
- Get data from : https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals.
- You can setup your virtual environment in your local computer and run this command to train the model: python train_v2.py or python train_v1.py
  Then you wil see something like this:
  
  ![Screenshot (7)](https://github.com/vuniem131104/90-Animals-CLassification-With-Pytorch/assets/124224840/0eacef0a-f90d-449d-8291-774ca8590bf3)
- Moreover, if you want to plot accuracy and loss please try function plot_loss_curves(results: pandas.DataFrame) in utils.py and you will have a picture like this:
  ![Loss_Accuracy](https://github.com/vuniem131104/90-Animals-CLassification-With-Pytorch/assets/124224840/fa0c473a-f4bd-4aab-ba9d-6e8495ae643c)
- I used Early Stopping and Learning Rate Scheduler to reduce overfitting in the model
