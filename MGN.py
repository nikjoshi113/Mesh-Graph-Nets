import copy
import enum
import random
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_scatter
from matplotlib import animation
from matplotlib import tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn import LayerNorm, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing
from tqdm import tqdm, trange
from tqdm.contrib import tzip

from helpers import memory_usage

#from meshgraphnets.helpers import memory_usage

# ---------------------------------------------------------------------------------------------
#   Utility functions
# ---------------------------------------------------------------------------------------------


def normalize(to_normalize, mean_vec, std_vec):
    return (to_normalize-mean_vec)/std_vec


def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize*std_vec+mean_vec


def get_stats(data_list):
    '''
    Method for normalizing processed datasets. Given  the processed data_list, 
    calculates the mean and standard deviation for the node features, edge features, 
    and node outputs, and normalizes these using the calculated statistics.
    '''

    # mean and std of the node features are calculated
    mean_vec_x = torch.zeros(data_list[0].x.shape[1:])
    std_vec_x = torch.zeros(data_list[0].x.shape[1:])

    # mean and std of the edge features are calculated
    mean_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])

    # mean and std of the output parameters are calculated
    mean_vec_y = torch.zeros(data_list[0].y.shape[1:])
    std_vec_y = torch.zeros(data_list[0].y.shape[1:])

    # Define the maximum number of accumulations to perform such that we do
    # not encounter memory issues
    max_accumulations = 10**6

    # Define a very small value for normalizing to
    eps = torch.tensor(1e-8)

    # Define counters used in normalization
    num_accs_x = 0
    num_accs_edge = 0
    num_accs_y = 0

    # Iterate through the data in the list to accumulate statistics
    for dp in data_list:

        # Add to the
        mean_vec_x += torch.sum(dp.x, dim=0)
        std_vec_x += torch.sum(dp.x**2, dim=0)
        num_accs_x += dp.x.shape[0]

        mean_vec_edge += torch.sum(dp.edge_attr, dim=0)
        std_vec_edge += torch.sum(dp.edge_attr**2, dim=0)
        num_accs_edge += dp.edge_attr.shape[0]

        mean_vec_y += torch.sum(dp.y, dim=0)
        std_vec_y += torch.sum(dp.y**2, dim=0)
        num_accs_y += dp.y.shape[0]

        if (num_accs_x > max_accumulations or num_accs_edge > max_accumulations or num_accs_y > max_accumulations):
            break

    mean_vec_x = mean_vec_x/num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(
        std_vec_x/num_accs_x - mean_vec_x**2), eps)

    mean_vec_edge = mean_vec_edge/num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(
        std_vec_edge/num_accs_edge - mean_vec_edge**2), eps)

    mean_vec_y = mean_vec_y/num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(
        std_vec_y/num_accs_y - mean_vec_y**2), eps)

    mean_std_list = [mean_vec_x, std_vec_x, mean_vec_edge,
                     std_vec_edge, mean_vec_y, std_vec_y]

    return mean_std_list


def build_optimizer(args, params):
    weight_decay = args.model.architecture.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.model.architecture.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.model.architecture.lr,
                               weight_decay=weight_decay)
    elif args.model.architecture.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.model.architecture.lr,
                              momentum=0.95, weight_decay=weight_decay)
    elif args.model.architecture.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            filter_fn, lr=args.model.architecture.lr, weight_decay=weight_decay)
    elif args.model.architecture.opt == 'adagrad':
        optimizer = optim.Adagrad(
            filter_fn, lr=args.model.architecture.lr, weight_decay=weight_decay)
    if args.model.architecture.opt_scheduler == 'none':
        return None, optimizer
    elif args.model.architecture.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.model.architecture.opt_decay_step, gamma=args.model.architecture.opt_decay_rate)
    elif args.model.architecture.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.model.architecture.opt_restart)
    return scheduler, optimizer


def control_randomness(seed=5):
    """
    To ensure reproducibility the best we can, here we control the sources of
    randomness by seeding the various random number generators used in this Colab
    For more information, see: https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)  # Torch
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy


def save_plots(args, losses, test_losses, feature_val_losses):
    """
    """
    args.dataset.dir.postprocess = str(
        args.dataset.dir.postprocess+"/"+args.model.settings.name)
    # df = pd.DataFrame({"training loss": losses,"test loss": test_losses})
    f = plt.figure()
    plt.title('Losses Plot')
    plt.plot(losses, label="training loss" +
             " - " + args.model.settings.model_type)
    plt.plot(test_losses, label="test loss" +
             " - " + args.model.settings.model_type)
    if (args.model.postprocessing.save_feature_val):
        plt.plot(feature_val_losses, label="feature loss" +
                 " - " + args.model.settings.model_type)
        # df = pd.concat([df, pd.DataFrame({"feature loss": feature_val_losses})], index=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.show()
    f.savefig(os.path.join(str(args.dataset.dir.postprocess+'_losses.pdf')), bbox_inches='tight')
    # df.to_csv(os.path.join(str(args.dataset.dir.postprocess+'_losses.csv')),index_label="epochs",sep=";")




def triangles_to_edges(faces):
    """Computes mesh edges from triangles.
       Note that this triangles_to_edges method was provided as part of the
       code release for the MeshGraphNets paper by DeepMind, available here:
       https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    if ('tf' in sys.modules) and ('tf' in dir()):
        import tensorflow.compat.v1 as tf

    # collect edges from triangles
    edges = tf.concat([faces[:, 0:2],
                       faces[:, 1:3],
                       tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers = tf.reduce_min(edges, axis=1)
    senders = tf.reduce_max(edges, axis=1)
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
    # remove duplicates and unpack
    #   print(packed_edges.shape)
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)
    # create two-way connectivity
    return (tf.concat([senders, receivers], axis=0),
            tf.concat([receivers, senders], axis=0))

# ---------------------------------------------------------------------------------------------
#   Classes
# ---------------------------------------------------------------------------------------------


class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    """
    NORMAL = 0
    BOTTOM = 1
    BUILDPLATE_BOUNDARY = 2
    ZERO = 3
    PART_BOUNDARY = 4
    TOP_LAYER = 5
    NOT_IN_USE = 6


class MeshGraphNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, args, emb=False):
        super(MeshGraphNet, self).__init__()
        """
        MeshGraphNet model. This model is built upon Deepmind's 2021 paper.
        This model consists of three parts: (1) Preprocessing: encoder (2) Processor
        (3) postproccessing: decoder. Encoder has an edge and node decoders respectively.
        Processor has two processors for edge and node respectively. Note that edge attributes have to be
        updated first. Decoder is only for nodes.

        Input_dim: dynamic variables + node_type + node_position
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: temperature changes (1)

        """

        self.num_layers = args.model.architecture.num_layers

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear(input_dim_edge, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim)
                                       )

        self.processor = nn.ModuleList()
        assert (self.num_layers >=
                1), 'Number of message passing layers is not >=1'

        processor_layer = self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim, hidden_dim))

        # decoder: only for node embeddings
        self.decoder = Sequential(Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, output_dim)
                                  )

    def build_processor_model(self):
        return ProcessorLayer

    def forward(self, data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = normalize(x, mean_vec_x, std_vec_x)
        edge_attr = normalize(edge_attr, mean_vec_edge, std_vec_edge)

        # Step 1: encode node/edge features into latent node/edge embeddings
        # output shape is the specified hidden dimension
        x = self.node_encoder(x)

        # output shape is the specified hidden dimension
        edge_attr = self.edge_encoder(edge_attr)

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest

        return self.decoder(x)

    def loss(self, pred, inputs, mean_vec_y, std_vec_y):
        # Define the node types that we calculate loss for
        #normal = torch.tensor(int(NodeType.NORMAL))
        #top_layer = torch.tensor(int(NodeType.TOP_LAYER))

        # Get the loss mask for the nodes of the types we calculate loss for
        #loss_mask = torch.logical_or((torch.argmax(inputs.x[:, 2:], dim=1) == normal),
        #                             (torch.argmax(inputs.x[:, 2:], dim=1) == top_layer))
        # print(mean_vec_y.max(),mean_vec_y.min())
        # Normalize labels with dataset statistics
        labels = normalize(inputs.y, mean_vec_y, std_vec_y)

        # Find sum of square errors
        error = torch.sum((labels-pred)**2, axis=1)

        # Root and mean the errors for the nodes we calculate loss for
        loss = torch.abs(torch.mean(error))
        #loss = torch.mean(error[loss_mask])
        # Use Mean square Error

        return loss


class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """

        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.edge_mlp = Sequential(Linear(3 * in_channels, out_channels),
                                   ReLU(),
                                   Linear(out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.node_mlp = Sequential(Linear(2 * in_channels, out_channels),
                                   ReLU(),
                                   Linear(out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """

        # out has the shape of [E, out_channels]
        out, updated_edges = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=size)

        # Complete the aggregation through self-aggregation
        updated_nodes = torch.cat([x, out], dim=1)

        updated_nodes = x + self.node_mlp(updated_nodes)  # residual connection

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """

        # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edges = self.edge_mlp(updated_edges)+edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """

        # The axis along which to index number of nodes.
        node_dim = 0

        out = torch_scatter.scatter(
            updated_edges, edge_index[0, :], dim=node_dim, reduce='sum')

        return out, updated_edges

# ---------------------------------------------------------------------------------------------
#   Main functions
# ---------------------------------------------------------------------------------------------


def train_mgn(dataset, device, args):
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''
    if not os.path.isdir(args.dataset.dir.checkpoint):
        os.mkdir(args.dataset.dir.checkpoint)

    df = pd.DataFrame(columns=['epoch', 'train_loss',
                      'test_loss', 'feature_val_loss'])

    stats_list = get_stats(dataset)

    # torch_geometric DataLoaders are used for handling the data of lists of graphs
    loader = DataLoader(dataset[:(args.model.data.trajectories_train*args.dataset.specs.number_timesteps)],
                        batch_size=args.model.architecture.batch_size, #num_workers=args.model.settings.num_workers,
                        shuffle=False,
                        pin_memory=args.model.settings.pin_memory,
                        drop_last = True)  # , pin_memory=False)  # num_workers=args.num_workers,

    test_loader = DataLoader(dataset[(args.model.data.trajectories_train*args.dataset.specs.number_timesteps):],
                             batch_size=args.model.architecture.batch_size, #num_workers=args.model.settings.num_workers,
                             shuffle=False,
                             pin_memory=args.model.settings.pin_memory,
                             drop_last = True)  # )  # num_workers=args.num_workers,

    # The statistics of the data are decomposes
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge,
        mean_vec_y, std_vec_y] = stats_list

    # for k in stats_list:
    #     print(k)

    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device), std_vec_x.to(device), mean_vec_edge.to(device),
        std_vec_edge.to(device), mean_vec_y.to(device), std_vec_y.to(device))

    # build model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 1  # the dynamic variables have the shape of 2 (velocity)

    model = MeshGraphNet(
        num_node_features, num_edge_features,
        args.model.architecture.hidden_dim, num_classes, args
    ).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_losses = []
    feature_val_losses = []
    best_test_loss = np.inf
    best_model = None
    best_model_no = 0
    with trange(args.model.architecture.epochs, desc="Training", unit="Epochs") as pbar:
        for epoch in pbar:
            total_loss = 0
            model.train()
            num_loops = 0

            for batch in loader:
                # Note that normalization must be done before it's called. The unnormalized
                # data needs to be preserved in order to correctly calculate the loss
                batch = batch.to(device)
                opt.zero_grad()  # zero gradients each time
                pred = model(batch, mean_vec_x, std_vec_x,
                             mean_vec_edge, std_vec_edge)
                loss = model.loss(pred, batch, mean_vec_y, std_vec_y)
                loss.backward()  # backpropagate loss
                opt.step()
                total_loss += loss.item()
                num_loops += 1
            total_loss /= num_loops
            losses.append(total_loss)
            # print(losses)
            # Every tenth epoch, calculate acceleration test loss and velocity validation loss
            #if best_model == None:
                #best_model = copy.deepcopy(model)
            if epoch % 10 == 0:
                if (args.model.postprocessing.save_feature_val):
                    # save velocity evaluation
                    test_loss, feature_val_rmse = test(test_loader, device, model, mean_vec_x, std_vec_x, mean_vec_edge,
                                                       std_vec_edge, mean_vec_y, std_vec_y, args.model.postprocessing.save_feature_val)
                    feature_val_losses.append(feature_val_rmse.item())
                else:
                    test_loss, _ = test(test_loader, device, model, mean_vec_x, std_vec_x, mean_vec_edge,
                                        std_vec_edge, mean_vec_y, std_vec_y, args.model.postprocessing.save_feature_val)

                test_losses.append(test_loss.item())

                # save the model if the current one is better than the previous best
                args.best_model.file_csv = str(
                    args.dataset.dir.checkpoint+"/"+args.model.settings.name+'.csv')
                df.to_csv(os.path.join(args.best_model.file_csv), index=False,sep=";")

                # save the model if the current one is better than the previous best
                if test_loss < best_test_loss:
                    best_model_no = epoch
                    best_test_loss = test_loss
                    best_model = copy.deepcopy(model)

            else:
                # If not the tenth epoch, append the previously calculated loss to the
                # list in order to be able to plot it on the same plot as the training losses
                test_losses.append(test_losses[-1])
                if (args.model.postprocessing.save_feature_val):
                    feature_val_losses.append(feature_val_losses[-1])

            if (args.model.postprocessing.save_feature_val):
                # df = df.append({'epoch': epoch,'train_loss': losses[-1],
                #                 'test_loss':test_losses[-1],
                #                'feature_val_loss': feature_val_losses[-1]}, ignore_index=True)
                df = pd.concat([df, pd.DataFrame({
                    'epoch': epoch, 'train_loss': losses[-1],
                    'test_loss': test_losses[-1],
                    'feature_val_loss': feature_val_losses[-1]}, index=[0])], ignore_index=True)
            else:
                # df = df.append({'epoch': epoch, 'train_loss': losses[-1], 'test_loss': test_losses[-1]}, ignore_index=True)
                df = pd.concat([df, pd.DataFrame({
                    'epoch': epoch, 'train_loss': losses[-1],
                    'test_loss': test_losses[-1]}, index=[0])], ignore_index=True)
            if (epoch % 100 == 0):
                if (args.model.settings.verbose > 0):
                    memory_usage(print_status=True)
                    if (args.model.postprocessing.save_feature_val):
                        print("train loss", str(round(total_loss, 2)),
                              "test loss", str(round(test_loss.item(), 2)),
                              "velo loss", str(round(feature_val_rmse.item(), 5)))
                    else:
                        print("train loss", str(round(total_loss, 2)),
                              "test loss", str(round(test_loss.item(), 2)))

                if (args.model.postprocessing.save_best_model):
                    #print(epoch,type(best_model))
                    args.best_model.file_pt = str(
                        args.dataset.dir.checkpoint+"/"+args.model.settings.name+".pt")
                    torch.save(best_model.state_dict(),os.path.join(args.best_model.file_pt))

            pbar.set_postfix_str("train loss {}, test loss {}, best_model {}".format(
                str(round(total_loss, 2)), str(round(test_loss.item(), 2)), best_model_no))

        return test_losses, losses, feature_val_losses, best_model, best_test_loss, test_loader


def test(loader, device, test_model,
         mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, is_validation,
         delta_t=0.01, save_model_preds=False, model_type=None):
    '''
    Calculates test set losses and validation set errors.
    '''
    
    loss = 0
    temp_rmse = 0
    num_loops = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
        
            # calculate the loss for the model given the test set
            pred = test_model(data, mean_vec_x, std_vec_x,
                              mean_vec_edge, std_vec_edge)
            loss += test_model.loss(pred, data, mean_vec_y, std_vec_y)

            # calculate validation error if asked to
            if (is_validation):

                # Like for the MeshGraphNets model, calculate the mask over which we calculate
                # flow loss and add this calculated RMSE value to our val error
                normal = torch.tensor(0)
                #outflow = torch.tensor(5)
                loss_mask = torch.logical_or((torch.argmax(data.x[:, 2:], dim=1) == normal))
                                             #(torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(5)))

                # * delta_t
                feature = data.x[:, 0:2] + \
                    unnormalize(pred[:], mean_vec_y, std_vec_y) * data.dt[:]
                # * data.dt[:]  # * delta_t
                ground_truth = data.x[:, 0:2] + data.y[:]*data.dt[:]

                error = torch.sum((feature - ground_truth) ** 2, axis=1)
                temp_rmse += torch.abs(torch.mean(error)) #[loss_mask]
        
        num_loops += 1
        # if velocity is evaluated, return velo_rmse as 0
    return loss/num_loops, temp_rmse/num_loops
