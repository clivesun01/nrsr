import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from skimage.filters import threshold_otsu


def collate_fn(data):
    return itertools.chain(*data)


def real_error(predicted_angles, target_angles):
    error = torch.abs(predicted_angles - target_angles)
    mean_absolute_error = error.mean().item()
    return mean_absolute_error


class Train:
    def __init__(
            self,
            dataset,
            net,
            model_path=None,
            hidden_size=128,
            batch_size=256,
            train_set_ratio=0.8,
            learning_rate=0.001,
            weight_decay=1e-5,
            lr_decay_interval=20,
            lr_decay_rate=0.9,
            num_epochs=20,
            save_interval=100,
            device='',
            job_name=''
    ):

        self.dataset = dataset
        self.net = net
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.job_name = job_name
        self.save_interval = save_interval

        input_dim = self.dataset.get_input_dim()
        self.var_mask = self.dataset.get_var_mask()

        if device == '':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        # create model
        self.model = self.net(input_size=input_dim, hidden_size=self.hidden_size)
        if model_path:
            # load model
            self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)

        # loss function
        self.loss_fn_mse = nn.MSELoss()
        self.l0_loss = lambda penalty: 1 / self.batch_size * penalty

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_decay_interval, gamma=lr_decay_rate)

        # split the dataset to train and val subset
        train_size = int(train_set_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size
        print('Train data size: {} | Val data size: {}'.format(train_size, val_size))
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def run(self):
        train_steps = len(self.train_dataloader)
        val_steps = len(self.val_dataloader)

        train_losses = []
        val_losses = []
        train_mse_list = []
        val_mse_list = []

        total_masks = []

        for epoch in range(1, self.num_epochs + 1):
            # train phase
            train_mse = 0.0
            train_loss = 0.0
            val_mse = 0.0
            val_loss = 0.0

            val_mask_list = []

            self.model.train()
            for features, labels in self.train_dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs, penalty, mask = self.model(features)
                outputs = outputs.squeeze(-1)

                loss_mse = self.loss_fn_mse(outputs, labels)
                loss_l0 = self.l0_loss(penalty)
                loss = loss_mse + 0.25 * loss_l0

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.detach().item()
                train_mse += real_error(outputs, labels)

            # val phase
            self.model.eval()
            with torch.no_grad():
                for features, labels in self.val_dataloader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs, penalty, mask = self.model(features)
                    outputs = outputs.squeeze(-1)

                    loss_mse = self.loss_fn_mse(outputs, labels)
                    loss_l0 = self.l0_loss(penalty)
                    loss = loss_mse + 0.25 * loss_l0

                    mask = mask.cpu().numpy()
                    val_mask_list.append(mask)

                    val_loss += loss.detach().item()
                    val_mse += real_error(outputs, labels)

            self.scheduler.step()

            epoch_train_loss = train_loss / train_steps
            epoch_train_mse = train_mse / train_steps

            train_losses.append(epoch_train_loss)
            train_mse_list.append(epoch_train_mse)

            epoch_val_loss = val_loss / val_steps
            epoch_val_mse = val_mse / val_steps

            val_losses.append(epoch_val_loss)
            val_mse_list.append(epoch_val_mse)

            # mask with otsu
            val_mask_list_mean = np.mean(np.array(val_mask_list), 0)
            val_thresh = threshold_otsu(np.array(val_mask_list))
            val_mask_processed = np.where(val_mask_list_mean > 1.05 * val_thresh, 1, 0)

            total_masks.append(val_mask_processed)

            print('Epoch: %d/%d | Train Loss: %.4f | Train MSE: %.4f | Val Loss: %.4f  | Val MSE: %.4f ' % (
                epoch, self.num_epochs, epoch_train_loss, epoch_train_mse, epoch_val_loss, epoch_val_mse))

            print('The average parameters of L0 gating layer is :', list(val_mask_list_mean))
            print('The L0 gates is :', list(val_mask_processed))

        # if self.job_name:
        #     epoch_model_name = f"models/{self.job_name}_epoch_{str(epoch).zfill(5)}.pth"
        # else:
        #     epoch_model_name = f"models/epoch_{str(epoch).zfill(5)}.pth"
        # torch.save(self.model.state_dict(), epoch_model_name)
        #
        # print('true var mask is : {}, pred var mask is {}'.format(self.var_mask, list(new_labels)))
        #
        # if list(val_mask_processed) == self.var_mask:
        #     mask_result_1 = True
        # else:
        #     mask_result_1 = False

        return val_mask_processed


def plot(result_dict, job_name=''):
    train_losses = result_dict['train_losses']
    val_losses = result_dict['val_losses']
    train_acc_list = result_dict['train_acc_list']
    val_acc_list = result_dict['val_acc_list']

    if job_name:
        loss_curve_name = f'results/{job_name}_loss_curve.png'
        accuracy_name = f'results/{job_name}_accuracy.png'
        mse_name = f'results/{job_name}_mse.png'
    else:
        loss_curve_name = 'results/loss_curve.png'
        accuracy_name = 'results/accuracy.png'
        mse_name = 'results/mse.png'

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_curve_name, dpi=500)
    # plt.show()

    plt.figure()
    plt.plot(train_acc_list, label='Training Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(accuracy_name, dpi=500)
    # plt.show()

    if 'train_mse_list' in result_dict.keys():
        train_mse_list = result_dict['train_mse_list']
        val_mse_list = result_dict['val_mse_list']
        plt.figure()
        plt.plot(train_mse_list, label='Training Angle Diff')
        plt.plot(val_mse_list, label='Validation Angle Diff')
        plt.xlabel('Epoch')
        plt.ylabel('Angle Diff')
        plt.legend()
        plt.savefig(mse_name, dpi=500)
        # plt.show()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--job', default='regression', type=str, help='fill the job name')
#     args = parser.parse_args()
#
#     dataset = RegressionDataset('npz_data/regression/data_Livermore-11_n0.00_s0.npz')
#     net = L0Network
#
#     trainer = Train(dataset,
#                     net,
#                     job_name=args.job)
#     results = trainer.run()
