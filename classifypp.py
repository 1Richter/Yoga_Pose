import os
from os import listdir
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


# a helper function
def add_to_dict(d, key, val):
    if key in d:
        d[key].append(val)
    else:
        d[key] = [val]

class DataHandler:
    def __init__(self, root_dir, batch_size):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.class_to_label = {'downdog': 0, 'warrior2': 1, 'goddess': 2, 'tree': 3, 'plank': 4}

    def load_data(self, folder):
        datasets = {}
        all_data = []
        all_targets = []

        folder_path = os.path.join(self.root_dir, 'csv', folder)
        for file in listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                datasets[file] = pd.read_csv(file_path, header=None)

                data_tensor = torch.tensor(datasets[file].values, dtype=torch.float32)
                class_name = file.split("_")[0].split(".")[0]
                if class_name not in self.class_to_label:
                    print(f"Skipping file {file} with unknown class name {class_name}")
                    continue
                target_tensor = torch.tensor([self.class_to_label[class_name]]*data_tensor.shape[0], dtype=torch.int64)
                all_data.append(data_tensor)
                all_targets.append(target_tensor)

        all_data_tensor = torch.cat(all_data)
        all_targets_tensor = torch.cat(all_targets)
        combined_ds = TensorDataset(all_data_tensor, all_targets_tensor)
        if folder == "test":
            erg = DataLoader(combined_ds, shuffle=False, batch_size=self.batch_size, num_workers=0, drop_last=True)
        else:
            erg = DataLoader(combined_ds, shuffle=True, batch_size=self.batch_size, num_workers=0, drop_last=True)
        return erg
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(99, 30)
        self.conv1 = nn.Conv1d(1, 1, 3, padding=1) # kernel size should be hiher like
        self.lin2 = nn.Linear(30, 20)
        self.lin25 = nn.Linear(20, 11)
        self.lin3 = nn.Linear(11, 5)

    def forward(self, X):
        X1 = torch.relu(self.lin1(X))
        X15 = torch.relu(self.conv1(X1.unsqueeze(1)).squeeze(1))
        X2 = torch.relu(self.lin2(X1))
        X25 = torch.relu(self.lin25(X2))
        X3 = torch.softmax(self.lin3(X25), dim=-1)
        return X3
    
    def loss(self, Y_true, Y_pred):                                       # we put the loss fct in the class, this is not required
        Y_true_oh = torch.nn.functional.one_hot(Y_true, num_classes=5)   # recode to a one-hot tensor
        sample_loss = -(Y_true_oh*torch.log(Y_pred+1e-7)).sum(axis=1)
        return sample_loss.mean()
        # use crossentropie
        # return torch.nn.functional.cross_entropy(sample_loss, Y_true)

    def metrics(self, Y_true, Y_pred):
        Y_pred_class = torch.argmax(Y_pred, dim=-1)
        acc = (Y_true == Y_pred_class).float().mean()
        return {'acc': acc}


class Trainer:
    def __init__(self, model, train_loader, test_loader, learning_rate):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train_step(self, X, Y_true, mdl, opt):
        Y_pred = mdl(X)                        # predict
        L = mdl.loss(Y_true, Y_pred)           # compute the loss
        L.backward()                           # compute the gradients for the optimizer
        opt.step()                             # call the optimizer, modifies the weights
        opt.zero_grad()  
        L.detach().numpy()              # we dont do further computations with L, the numpy value is sufficient for reporting
        M = mdl.metrics(Y_true, Y_pred)        # compute the metrics
        for k in M:
            M[k] = M[k].detach().numpy()       # we dont do further computations with M, the numpy value is sufficient for reporting
        M["loss"] = L
        return M

    def val_step(self, X, Y_true, mdl):
        Y_pred = mdl(X)                        # predict
        L = mdl.loss(Y_true, Y_pred)           # compute the loss
        L = L.detach().numpy()              # we dont do further computations with L, the numpy value is sufficient for reporting
        M = mdl.metrics(Y_true, Y_pred)
        for k in M:
            M[k] = M[k].detach().numpy()
        M['loss'] = L
        return M  

    def train(self, train_dl, mdl, alpha, n_epochs, test_dl):
        opt = torch.optim.Adam(mdl.parameters(), lr=alpha)                        # choose the optimizer
        hist = { 'loss': [] }
        for epoch in range(n_epochs):    
            # for dl in train_dl:                                         # repeat for n epochs
            for step, (X, Y_true) in enumerate(train_dl):                         # repeat for all mini-batches
                mdl.train()
                M = self.train_step(X, Y_true, mdl, opt)                                                  # set the model to training mode
                for key in M:
                    add_to_dict(hist, key, M[key])                   # logging
                    # add_to_dict(hist, "time_val", step)
                if step % 100 == 0:
                    # total_ds_size = sum([len(dl.dataset) for dl in train_dl])
                    print(f'Epoch: {epoch}, step {step*batch_size:5d}/{len(train_dl.dataset)}:  ', end='')
                    for i, key in enumerate(M):
                        print(f'{(", " if i>0 else "")+key}: {M[key]:.6f}', end='')
                    print()
                if step % 300 == 0 or step == len(train_dl)-1 and epoch == n_epochs-1:
                    mdl.eval()
                    M_sum = {}
                    #for testdl in test_dl:
                    for step_val, (X_val, Y_val) in enumerate(test_dl):
                        M = self.val_step(X_val, Y_val, mdl)
                        if not M_sum:
                            M_sum = M
                        else:
                            for key in M:
                                M_sum[key] += M[key]
                        for key in M_sum:
                            add_to_dict(hist, key + '_val', M_sum[key]/(step_val+1))
                        add_to_dict(hist, 'time_val', step+1)                              # logging
                    print(f'>>>  Validation:             ', end='')
                    for i, key in enumerate(M):
                        print(f'{(", " if i>0 else "")+key}_val: {M[key]:.6f}', end='')
                    print()
                    
        return hist
    def plot_train(self, hist):
        fig, axs = plt.subplots(2, 1)
        loss_values = [item.detach().numpy() for item in hist['loss']]
        axs[0].plot(loss_values, 'b')
        axs[0].plot(hist['time_val'], hist['loss_val'], 'r')
        axs[0].grid()
        axs[1].plot(hist['acc'], 'b')
        axs[1].plot(hist['time_val'], hist['acc_val'], 'r')
        axs[1].grid()
        plt.show(block=True)
        
class Tester:
    def __init__(self, model, test_loader, net, sample_size, class_to_label = {'downdog': 0, 'warrior2': 1, 'goddess': 2, 'tree': 3, 'plank': 4}):
        # get the class_to_label dict from Daatahandler
        self.model = model
        self.model.load_state_dict(torch.load(net))
        self.model.eval()
        self.sample_size = sample_size
        self.test_loader = test_loader
        self.class_to_label = class_to_label

    def test(self, test_dl):
            # Get a list of all data loaders
        all_dls = list(test_dl)

        for e in range(self.sample_size):
            # Select dataloader e without random choice
            # dl = all_dls[e]

            # Get a random batch of images
            dl = random.choice(all_dls)
            images= next(iter(dl))

            # Get the model's predictions
            preds = self.model(images)

            # Get the predicted class for each image
            pred_classes = preds.argmax(dim=1)

            # Print the most likely exercise for each image
            for i in range(len(images)):
                image = images[i]
                pred_class = pred_classes[i]
                print(f"Image {[i]}: Most likely exercise is {pred_class}")
                image_np = image.numpy()
                self.show_data(image_np, pred_class)

    def show_data(self, test_data, test_label):
        # MediaPipe Pose landmarks and connections
        mpPoseLandmarks = [i for i in range(33)]
        mpPoseConnections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (11, 23), (23, 25),
            (25, 27), (27, 29), (29, 31), (12, 14), (14, 16), (16, 18), (12, 24),
            (24, 26), (26, 28), (28, 30), (30, 32), (11, 12)
        ]
        data_list = test_data.tolist()
            # elements (x, y, z) into a tuple and converting them into floats
        landmarks = [(float(data_list[i]), float(data_list[i+1]), float(data_list[i+2])) for i in range(0, len(data_list), 3)]

        # Create a black background
        img = np.zeros((500, 500, 3), dtype=np.uint8)

        # Plot each landmark
        for x, y, z in landmarks:
            # Scale x, y to fit in the image
            # This assumes x, y are normalized coordinates in range [0, 1]
            x = int(x * img.shape[1])
            y = int(y * img.shape[0])

            # Draw the landmark as a red dot
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        # Draw lines between pairs of points
        for i, j in mpPoseConnections:
            if i < len(landmarks) and j < len(landmarks):
                x1, y1, _ = landmarks[i]
                x2, y2, _ = landmarks[j]

                # Scale x, y to fit in the image
                x1 = int(x1 * img.shape[1])
                y1 = int(y1 * img.shape[0])
                x2 = int(x2 * img.shape[1])
                y2 = int(y2 * img.shape[0])

                # Draw the line
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the image
        # take this from Datahandler self.class_to_label = {'downdog': 0, 'warrior2': 1, 'goddess': 2, 'tree': 3, 'plank': 4} and get the key from the value
        for key, value in self.class_to_label.items():
            if value == test_label:
                print(key)
                plt.title(key)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()



        

if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    model_name = 'model.pt'

    dl = DataHandler(root_dir, batch_size)
    train_loader = dl.load_data('train')
    test_loader = dl.load_data('test')

    model = Model()
    # trainer = Trainer(model, train_loader, test_loader, learning_rate)
    # train = trainer.train(train_loader, mdl = model, alpha = learning_rate, n_epochs=epochs, test_dl=test_loader)
    # torch.save(model.state_dict(), model_name)
    # trainer.plot_train(train)

    tester = Tester(model, test_loader, model_name, 7)
    tester.test(test_loader)

