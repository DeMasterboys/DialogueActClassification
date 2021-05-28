# Libraries
import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import json

# Training / Testing
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from preprocess import PATH, get_csv_files, preprocess_text, make_df_equal
from dataloader import load_train_iter, load_test_iter
from lstm import LSTM
from testset import get_test_set

SAVE_PATH = 'saved'
RESULTS_PATH = 'results'


def main(args):
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    check_paths()
    get_csv_files(args)

    # Configure LSTM model
    text_field, train_iter, valid_iter = load_train_iter(device, args)
    model = LSTM(text_field, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Test the model
    if args.test:
        _ = load_checkpoint(SAVE_PATH + f'/lstm_model_{int(args.equal)}.pt', model, optimizer, device)
        save_file = 'results/results_equal.txt' if args.equal else 'results/results.txt'
        with open(save_file, 'w') as f:
            f.write('[')
            for i in np.arange(1, 31, 1):
                args.length = i

                if args.equal:
                    df_test = make_df_equal(preprocess_text('test', args.type, args.length, test=True))
                    save_path = f'{PATH}/test_{args.type}_equal.csv'
                else:
                    df_test = preprocess_text('test', args.type, args.length, test=True)
                    save_path = f'{PATH}/test_{args.type}.csv'
                df_test.to_csv(save_path, index=False)

                print("NORMAL RESULTS")
                load_path = f'{PATH}/test_{args.type}_equal.csv' if args.equal else f'{PATH}/test_act.csv'
                test_iter_normal = load_test_iter(device, args, load_path)
                f.write('[')
                if len(test_iter_normal) > 0:
                    report_normal = test(args=args, model=model, test_loader=test_iter_normal, device=device, name='Normal')
                    f.write(json.dumps(report_normal))
                    f.write(',\n')
                else:
                    f.write('0,\n')

                print("PERTURBATED RESULTS")
                get_test_set(length=args.length)
                path = f'{PATH}/testset_equal.csv' if args.equal else f'{PATH}/testset.csv'
                test_iter_perturbated = load_test_iter(device, args, path)
                if len(test_iter_perturbated) > 0:
                    report_perturbated = test(model=model, test_loader=test_iter_perturbated, device=device, name='Perturbated')
                    f.write(json.dumps(report_perturbated))
                    f.write('],\n')
                else:
                    f.write('0],\n')
                f.write('\n')
            f.write(']')

    # Train the model
    else:
        criterion = nn.NLLLoss()
        train(args=args, model=model, optimizer=optimizer, criterion=criterion, train_loader=train_iter,
              valid_loader=valid_iter, eval_every=len(train_iter) // 2, file_path=SAVE_PATH,
              best_valid_loss=float("Inf"), num_epochs=args.epochs, device=device)


def test(args, model, test_loader, device, name):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, (text, text_len)), _ in tqdm(test_loader, ncols=75):
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    labels = ['inform', 'question', 'directive', 'commissive']
    report = classification_report(y_true, y_pred, target_names=labels, digits=4, output_dict=True)
    print(report)
    print()

    if args.length == 30:
        cm = confusion_matrix(y_true, y_pred)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

        ax.set_title(f'Confusion Matrix {name}')

        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.show()

    return report


def train(args, model, optimizer, criterion, train_loader, valid_loader,
          eval_every, file_path, best_valid_loss, num_epochs, device):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()

    for epoch in range(num_epochs):
        for (labels, (text, text_len)), _ in tqdm(train_loader, ncols=75):
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            labels = labels.long()
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                accuracy = 0
                with torch.no_grad():
                    # validation loop
                    for (labels, (text, text_len)), _ in valid_loader:
                        labels = labels.to(device)
                        text = text.to(device)
                        text_len = text_len.to(device)
                        output = model(text, text_len)
                        pred = output.argmax(1, keepdim=True)
                        accuracy += pred.eq(labels.view_as(pred)).sum().item()
                        labels = labels.long()
                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()
                accuracy = accuracy / len(valid_loader.dataset) * 100

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print()
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Accuracy: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss, accuracy))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + f'/lstm_model_{int(args.equal)}.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + f'/lstm_metrics_{int(args.equal)}.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + f'/lstm_metrics_{int(args.equal)}.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def check_paths():
    try:
        os.mkdir(SAVE_PATH)
    except OSError:
        pass

    try:
        os.mkdir(RESULTS_PATH)
    except OSError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Arguments
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--type', default='act', type=str, choices={'emotion', 'act'})
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--equal', default=False, type=bool)
    parser.add_argument('--length', default=np.inf, type=float)
    args = parser.parse_args()
    main(args)
