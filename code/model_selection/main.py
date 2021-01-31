
import os
import time
import matplotlib

matplotlib.use('Agg')
import numpy as np
import torch
from architecture import Net
from widis_lstm_tools.nn import LearningRateDecay
from widis_lstm_tools.utils.config_tools import get_config
from widis_lstm_tools.preprocessing import PadToEqualLengths, random_dataset_split
from dataset import TensorDataset
from widis_lstm_tools.measures import bacc
from widis_lstm_tools.utils.collection import TeePrint, SaverLoader, close_all
from sklearn.metrics import roc_auc_score as rocac
import warnings
from collections import Counter
import utils
warnings.filterwarnings("ignore")
import pickle
import h5py

# Reading in all the preprocessed data from the preprocessing directory


seq_src = '../../preprocessing/fullsequence_swiss_human.pkl'
uniprot_src = '../../preprocessing/uni2go_human.pkl'
ggid_src = '../../preprocessing/goid_human.pkl'
gid_src = '../../preprocessing/one_hot_go.h5py'
full_sequences= utils.readbin(seq_src)# reading in all the sequences which will be used as input (X)
uniprot_data = utils.readbin(uniprot_src)# Loading in the uniprot goid relationship dictionary created in preprocessing
ggid = utils.readbin(ggid_src)# loading in the goids which are specific to humans
gid = h5py.File(gid_src, 'r')  # loading in the one hot econding relationship matrix
print("End of Reading")
# end of Reading

validauc = []
testauc = []
validbacc = []
testbacc = []

def main():
    # Read config file path and set up results folder
    config, resdir = get_config()
    logfile = os.path.join(resdir, 'log.txt')
    checkpointdir = os.path.join(resdir, 'checkpoint')
    os.makedirs(resdir, exist_ok=True)
    os.makedirs(checkpointdir, exist_ok=True)
    tee_print = TeePrint(logfile)
    tprint = tee_print.tee_print

    # Set up PyTorch and set random seeds
    torch.set_num_threads(config['num_threads'])
    torch.manual_seed(config['rnd_seed'])
    np.random.seed(config['rnd_seed'])
    device = torch.device(config['device'])  # e.g. "cpu" or "cuda:0"

    # Get dataset and split it into training-, validation-, and testset
    full_dataset = TensorDataset(uniprot_data, full_sequences, gid, ggid)

    training_set, validation_set, test_set = random_dataset_split(dataset=full_dataset, split_sizes=(.6, .2, .2),
                                                                  rnd_seed=config['rnd_seed'])
    n_class = torch.Tensor([i for i in range(full_dataset.n_allowed_goids)])
    print(n_class)
    # Set up sequence padding
    padder = PadToEqualLengths(
        padding_dims=(0, None, None),  # only pad the first entry (sequences) in sample at dimension 0 (=seq.len.)
        padding_values=(0, None, None)  # pad with zeros
    )
    # Get Dataloaders
    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=config['batch_size'],
                                                      shuffle=True, num_workers=0, collate_fn=padder.pad_collate_fn, )
    validation_set_loader = torch.utils.data.DataLoader(validation_set, batch_size=config['batch_size'] * 4,
                                                        shuffle=False, num_workers=0, collate_fn=padder.pad_collate_fn)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'] * 4,
                                                  shuffle=False, num_workers=0, collate_fn=padder.pad_collate_fn)

    # Go through all labels in each dataloader and find the overlapping labels in each dataloader
    trainlabels = [s for mb in training_set_loader for s in mb[1]]

    validationlabels = [s for mb in validation_set_loader for s in mb[1]]
    testlabels = [s for mb in test_set_loader for s in mb[1]]
    n_train_samples = len(trainlabels)
    n_test_samples = len(testlabels)
    n_validation_samples = len(validationlabels)
    traindindices = torch.stack(trainlabels, dim=0).detach().cpu().numpy().sum(axis=0)
    traindindices = np.logical_and(traindindices > 0, traindindices < n_train_samples)
    testdindices = torch.stack(testlabels, dim=0).detach().cpu().numpy().sum(axis=0)
    testdindices = np.logical_and(testdindices > 0, traindindices < n_test_samples)
    validdindices = torch.stack(validationlabels, dim=0).detach().cpu().numpy().sum(axis=0)
    validdindices = np.logical_and(validdindices > 0, traindindices < n_validation_samples)
    finalindices = torch.Tensor(np.asarray(testdindices * validdindices * traindindices, dtype=np.bool)).to(
        dtype=torch.bool)

    # Create Network
    net = Net(n_input_features=full_dataset.n_features, n_lstm=config['n_lstm'], n_outputs=full_dataset.n_classes,
              kernel_size=config["kernel_size"], use_cnn=config["use_cnn"],
              use_prefinal_layer=config["use_prefinal_layer"])
    net.to(device)

    # Get some loss functions
    #
    '''occur = []
    for i in range(len(finalindices)):
        if finalindices[i] == 1:
            occur.append(list(ggid.values())[i])
    print(Counter(occur))'''

    LossFunction = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['L2'])

    # Get some optimizer

    if config["opt"] == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], weight_decay=config['L2'])

    # Get a linear learning rate decay
    lr_decay = LearningRateDecay(max_n_updates=config['n_updates'], optimizer=optimizer, original_lr=config['lr'])

    # Create a dictionary with objects we want to have saved and loaded if needed
    state = dict(net=net, optimizer=optimizer, update=0, best_validation_loss=np.inf, validation_bacc=0)

    # Use the SaverLoader class to save/load our dictionary to files or to RAM objects
    saver_loader = SaverLoader(save_dict=state, device=device, save_dir=checkpointdir, n_savefiles=1, n_inmem=1)

    def calc_score(scoring_dataloader, scoring_dataset):
        """Compute scores on dataset"""
        with torch.no_grad():
            tp_sum = 0.
            tn_sum = 0.
            p_sum = 0.
            avg_loss = 0.
            alloutputs = []
            alllabels = []
            for scoring_data in scoring_dataloader:
                # Get samples
                inputs, labels, _ = scoring_data
                labels = labels[:, finalindices]

                padded_sequences, seq_lens = inputs
                padded_sequences, labels = padded_sequences.to(device), labels.float().to(device)

                # Get outputs for network
                outputs = net(padded_sequences, seq_lens)[:, finalindices]

                # Add loss to mean loss over testset
                if config['classification'] == 'binary':
                    loss = LossFunction(outputs, labels)
                else:
                    labels_mask = torch.zeros(labels.shape[1])
                    labels_mask[labels.shape[1]-1] = 1.
                    for i in range(labels.shape[1]-2):
                        labels_mask[i] = .8 / (labels.shape[1]-1)
                    loss_main = LossFunction(outputs, labels)
                    loss = torch.sum(loss_main * labels_mask)

                avg_loss += (loss * (len(labels) / len(scoring_dataset)))

                # Store sum of tp, tn, t for BACC calculation\
                labels = labels.float()
                p_sum += labels.sum(dim=0)  # number of positive samples
                predictions = (outputs > 0).to(dtype=torch.float)
                alloutputs.append(outputs)
                alllabels.append(labels)
                try:
                    tp_sum += (predictions * labels).sum(dim=0)
                except Exception as e:
                    print(predictions.shape, labels.shape)
                    raise e
                tn_sum += ((1 - predictions) * (1 - labels)).sum(dim=0)

            # Compute balanced accuracy
            n_sum = len(scoring_dataset) - p_sum
            cf = '\n psum=' + str(p_sum) + '\nnsum=' + str(n_sum) + '\ntp_sum' + str(tp_sum) + '\ntn_sum' + str(
                tn_sum) + '\nfp_sum=' + str(p_sum - tp_sum) + '\nfn_sum=' + str(n_sum - tn_sum)

            bacc_scoress = bacc(tp=tp_sum, tn=tn_sum, p=p_sum, n=n_sum).detach().cpu().numpy()

            bacc_scoresu = np.nan_to_num(bacc_scoress)
            bacc_scores = bacc_scoresu[bacc_scoresu != 0]
            avg_loss = avg_loss.cpu().item()

            alloutputse = torch.cat(alloutputs, dim=0).detach().cpu().numpy()
            alllabelse = torch.cat(alllabels, dim=0).detach().cpu().numpy()
            # compute aucscores
            try:
                aucscores = rocac(y_true=alllabelse,
                                  y_score=alloutputse, average=None)


            except Exception as b:
                print('mean = ' + str(alllabelse.mean(axis=0)))
                print(alllabelse.shape, alloutputse.shape)
                f = alllabelse.mean(axis=0)
                for x in f:
                    if x > .4:
                        print(x)
                raise b
            print('aucs', aucscores)

        return bacc_scores, avg_loss, aucscores, cf  # vector of all the classes

    # Get current state for some Python variables
    update, best_validation_loss, validation_bacc = (state['update'], state['best_validation_loss'],
                                                     state['validation_bacc'])

    # Save initial model as first model
    saver_loader.save_to_ram(savename=str(update))

    #
    # Start training
    #
    tprint("# settings: {}".format(config))
    while update < config['n_updates']:
        running_loss = 0.
        start_time = time.time()

        for data in training_set_loader:

            # Get and set current learning rate
            lr = lr_decay.get_lr(update)

            # Get samples
            inputs, labels, sample_id = data
            padded_sequences, seq_lens = inputs
            labels = labels[:, finalindices]
            padded_sequences, labels = padded_sequences.to(device), labels.float().to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs for network
            outputs = net(padded_sequences, seq_lens)[:, finalindices]

            # Calculate loss, do backward pass, and update
            if config['classification'] == 'binary':
                loss = LossFunction(outputs, labels)
            else:
                # class masks
                labels_mask = torch.zeros(labels.shape[1])
                labels_mask[labels.shape[1]-1] = 1.
                for i in range(labels.shape[1]-2):
                    labels_mask[i] = .8 / (labels.shape[1]-1)
                loss_main = LossFunction(outputs, labels)
                loss = torch.sum(loss_main * labels_mask)
            loss.backward()
            optimizer.step()
            update += 1

            # Update running losses for our statistic
            running_loss += loss.item() if hasattr(loss, 'item') else loss

            # Print current status and score
            if update % config['print_stats_at'] == 0 and update > 0:
                run_time = (time.time() - start_time) / config['print_stats_at']
                running_loss /= config['print_stats_at']
                train_baccs, train_loss, train_aucs, train_cf = calc_score(
                    scoring_dataloader=training_set_loader,
                    scoring_dataset=training_set)
                tprint(
                    f"[train] u: {update:07d}; loss: {train_loss:8.7f}; baccs: {train_baccs}; train_aucs:{train_aucs};"f" bacc: {np.mean(train_baccs):5.4f}; average_auc:{np.mean(train_aucs):5.4f}")
                running_loss = 0.
                start_time = time.time()

            # Do some plotting using the LSTMLayer plotting function
            if update % config['plot_at'] == 0:
                # This will plot the LSTM internals for sample 0 in minibatch
                mb_index = 0
                net.lstm1.plot_internals(
                    filename=os.path.join(resdir, 'lstm_plots',
                                          f'u{update:07d}_id{sample_id[0]}.png'),
                    mb_index=mb_index, fdict=dict(figsize=(50, 10), dpi=100))
                start_time = time.time()

            if update % config['validate_at'] == 0 or update == config['n_updates']:
                # Calculate scores and loss on validation set
                print("  Calculating validation score...", end='')
                validation_start_time = time.time()
                validation_baccs, validation_loss, validation_aucs, validation_cf = calc_score(
                    scoring_dataloader=validation_set_loader,
                    scoring_dataset=validation_set)
                validauc.append(validation_aucs)
                validbacc.append(validation_baccs)
                print(f" ...done! ({time.time() - validation_start_time})")
                with open('confusion_matrix.txt', 'a') as f:
                    f.write(str(update))
                    f.write(str(validation_cf))
                tprint(
                    f"[validation] u: {update:07d}; loss: {validation_loss:8.7f}; baccs: {validation_baccs}; validation_aucs:{validation_aucs};"
                    f" bacc: {np.mean(validation_baccs):5.4f}; validation_auc:{np.mean(validation_aucs):5.4f}")

                # If we have a new best loss on the validation set, we save the model as new best model
                if best_validation_loss > validation_loss:
                    best_validation_loss = validation_loss
                    tprint(f"  New best validation loss: {validation_loss}")
                    # Save current state as RAM object
                    state['update'], state['best_validation_loss'], state['validation_bacc'] = \
                        update, best_validation_loss, validation_bacc
                    saver_loader.save_to_ram(savename=str(update))

            if update >= config['n_updates']:
                break

    print('Finished Training! Starting evaluation on test set...')

    #
    # Now we have finished training and stored the best model in 'checkpoint/best.tar.gzip' and want to see the
    # performance of this best model on the test set
    #

    # Load best model from newest RAM object
    state = saver_loader.load_from_ram()
    update, best_validation_loss, validation_bacc = (state['update'], state['best_validation_loss'],
                                                     state['validation_bacc'])

    # Save model to file
    saver_loader.save_to_file(filename='best' + str(resdir) + '.tar.gzip')

    # Calculate scores and loss on test set
    print("  Calculating testset score...", end='')
    test_start_time = time.time()
    test_baccs, test_loss, test_aucs, test_cf = calc_score(scoring_dataloader=test_set_loader, scoring_dataset=test_set)
    print(f" ...done! ({time.time() - test_start_time})")

    # Print results
    tprint(f"[test] u: {update:07d}; loss: {test_loss:8.7f}; baccs: {test_baccs}; test_aucs:{test_aucs};"
           f" bacc: {np.mean(test_baccs):5.4f}; test_auc:{np.mean(test_aucs):5.4f}")
    testauc.append(test_aucs)
    testbacc.append(test_baccs)
    with open('results.txt', 'a') as f:
        f.write(str(resdir) + ': ' + str(np.mean(test_aucs)))
    print('Done!')


if __name__ == '__main__':
    try:
        main()
    finally:
        close_all()
