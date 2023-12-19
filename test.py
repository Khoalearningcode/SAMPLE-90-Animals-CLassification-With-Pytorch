from pathlib import Path
import pandas as pd
from PIL import Image
from dataset import test_dataloader
import torch
import random
import matplotlib.pyplot as plt
from dataset import test_dir, val_tf, class_names

def test_model(model, dataset=test_dataloader):
    test_acc = 0
    model.eval()
    with torch.inference_mode():
        for X, y in dataset:
            y_logit = model(X)
            y_preds = y_logit.argmax(dim=1)
            test_acc += (y == y_preds).sum().item() / len(y_preds)
        test_acc /= len(dataset)
    print(f'Test accuracy: {test_acc}')

def plot_predict_image(model, data=test_dir):
    list_images = list(Path(data).glob('*/*.jpg'))
    random_images = random.sample(list_images, k=20)
    plt.figure(figsize=(12, 8), edgecolor='red')
    for i, img_path in enumerate(random_images):
        plt.subplot(4, 5, i + 1)
        img = Image.open(img_path)
        un_tf_img = val_tf(img).unsqueeze(dim=0)
        model.eval()
        with torch.inference_mode():
            y_pred = model(un_tf_img).argmax(dim=1)
        plt.imshow(img.resize(size=(224,224)))
        plt.axis(False)
        if img_path.parent.name == class_names[y_pred]:
            plt.title(f'Truth: {img_path.parent.name}\nPredict: {class_names[y_pred]}', size=5, c='g', fontweight='bold')
        else:
            plt.title(f'Truth: {img_path.parent.name}\nPredict: {class_names[y_pred]}', size=5, c='r', fontweight='bold')
    plt.savefig('predict.png')
    plt.show()

def plot_most_wrong_no_transform_mode(list_wrong, wrong_label):
    plt.figure(figsize=(15, 6))
    for i, img in enumerate(list_wrong):
        plt.subplot(1, 5, i + 1)
        image = Image.open(img)
        plt.imshow(image.resize(size=(224,224)))
        plt.title(wrong_label[i],size=8, c='r')
        plt.axis(False)
    plt.show()

def statistic_data(test_dir, model):
    dt = {
    'sample': [],
    'label': [],
    'prediction': [],
    'pred prob': []
    }
    list_images = list(Path(test_dir).glob('*/*.jpg'))
    y_true = [i.parent.name for i in list_images]
    y_pred = []
    y_pred_prob = []
    for img in list_images:
        model.eval()
        with torch.inference_mode():
            image = Image.open(img)
            transformed_image = val_tf(image).unsqueeze(dim=0)
            y_logit = model(transformed_image)
            y_prob = y_logit.softmax(dim=1).max(dim=1)
            y_pred_prob.append(y_prob.values.item())
            y_pred_label = y_logit.argmax(dim=1)
            y_pred.append(class_names[y_pred_label.item()])
    dt['sample'] = list_images
    dt['label'] = y_true
    dt['prediction'] = y_pred
    dt['pred prob'] = y_pred_prob
    df = pd.DataFrame(dt)
    wrong_df = df.where(df['label'] != df['prediction']).dropna()
    wrong_df.sort_values(by='pred prob', axis=0, ascending=False, inplace=True)
    most_wrong_predict = list(wrong_df.head()['sample'])
    wrong_data = wrong_df.head().drop(labels='sample', axis=1)
    wrong_label_prob = [f'Truth: {i[0]}\nPred: {i[1]}\nProb: {"{:.2f}".format(i[2])}' for i in wrong_data.values]
    plot_most_wrong_no_transform_mode(most_wrong_predict, wrong_label_prob)

