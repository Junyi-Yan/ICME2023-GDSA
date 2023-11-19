import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import os, copy, random, argparse, sys, time
import scipy.io as sio
from layers import GCN
from utils import process
from numpy import mat
from sklearn.metrics import roc_auc_score
start = time.time()
os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class GDSA_train(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GDSA_train, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.lin = nn.Linear(n_h, n_h)

    def forward(self, seq1, seq2, adj, sparse):
        h_1 = self.gcn(seq1, adj, sparse)
        h_2 = self.gcn(seq2, adj, sparse)
        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1)).unsqueeze(0)
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1)).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)
        return logits

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = self.gcn(seq, adj, sparse)
        h_2 = h_1.clone().squeeze(0)
        for i in range(5):
            h_2 = adj @ h_2

        h_2 = h_2.unsqueeze(0)
        return h_1.detach(), h_2.detach()

class GDSA_test(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GDSA_test, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.lin = nn.Linear(n_h, n_h)

    def forward(self, seq,adj, sparse):
        h = self.gcn(seq, adj, sparse)
        sc = ((self.lin(h.squeeze(0))).sum(1)).unsqueeze(0)

        logits = sc
        return logits,h

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

# Load real datasets
def load_mat(dataset):
    """Load .mat dataset."""
    data = sio.loadmat("dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    ano_labels = np.squeeze(np.array(label))

    return adj, feat, ano_labels

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def aug_feature_dropout(input_feat, drop_percent=0.2):
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)  # 列表解析式，在指定整行的特征量的范围内生成自定义的数值个数
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

if __name__ == '__main__':
    auc_results = []
    f1_results = []
    import warnings

    warnings.filterwarnings("ignore")

    #setting arguments
    parser = argparse.ArgumentParser('GDSA')
    parser.add_argument('--classifier_epochs', type=int, default=100, help='classifier epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--np_epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=500, help='Patience')
    parser.add_argument('--lr', type=float, default=0.001, help='Patience')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='l2 coef')
    parser.add_argument('--hid_units', type=int, default=512, help='Top-K value')
    parser.add_argument('--sparse', action='store_true', help='Whether to use sparse tensors')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name: cora, citeseer, Flickr,BlogCatalog')
    parser.add_argument('--n_trials', type=int, default=3, help='number of trails')
    parser.add_argument('--m', type=int)  # dis
    parser.add_argument('--n', type=int)
    parser.add_argument('--k', type=int, default=50)  # num of clusters 50
    parser.add_argument('--seed', type=int, default=1)  # random seed

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    auc_res = []
    # training params
    n_trails = args.n_trials
    for i in range(n_trails):
        torch.cuda.set_device(int(1))
        dataset = args.dataset
        batch_size = args.batch_size
        nb_epochs = args.np_epochs
        patience = args.patience
        classifier_epochs = args.classifier_epochs
        seed = args.seed
        lr = args.lr
        l2_coef = args.l2_coef
        hid_units = args.hid_units
        m = args.m  # num of fully connected nodes
        k = args.k
        sparse = True
        nonlinearity = 'prelu'

        # Load the original datasets
        if dataset in ['cora','citeseer']:
            adj, features = process.load_data(dataset)
        else:
            adj, features = process.load_mat(dataset)

        actual_adj, actual_features, actual_label = load_mat(dataset)

       # m and n are calculated according to the Equation for each dataset
        if args.n is None:
            if dataset == 'cora':
                n = 3
            elif dataset == 'citeseer':
                n = 7
            elif dataset == 'BlogCatalog':
                n = 10
            elif dataset == 'Flickr':
                n = 6
        else:
            n = args.n

        if args.m is None:
            if dataset == 'cora':
                m = 5
            elif dataset == 'citeseer':
                m = 8
            elif dataset == 'BlogCatalog':
                m = 15
            elif dataset == 'Flickr':
                m = 17
        else:
            m = args.m

        dis_adj = torch.FloatTensor(np.array(adj.todense(), dtype=np.float64))

        #preprocessing and initialisation
        features, _ = process.preprocess_features(features)
        features = torch.FloatTensor(features)

        actual_features, _ = process.preprocess_features(actual_features)
        actual_adj = process.normalize_adj(actual_adj + sp.eye(actual_adj.shape[0]))
        actual_adj = process.sparse_mx_to_torch_sparse_tensor(actual_adj)

        actual_adj = torch.tensor(actual_adj, dtype=torch.float32)
        actual_features = torch.FloatTensor(actual_features[np.newaxis]).cuda()

        nb_nodes = features.shape[0]
        nb_classes = len(np.unique(actual_label))
        ft_size = features.shape[1]

        GDSA = GDSA_train(ft_size, hid_units, nonlinearity)
        optimiser_disc = torch.optim.Adam(GDSA.parameters(), lr=lr, weight_decay=l2_coef)

        if torch.cuda.is_available():
            GDSA.cuda()
            features = features.cuda()
            dis_adj = dis_adj.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        # Train
        for epoch in range(nb_epochs):
            GDSA.train()
            optimiser_disc.zero_grad()

            # Structure disturbance (view2)
            dis_adj = np.array(adj.todense(), dtype=np.float64)
            # Random pick anomaly nodes
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            anomaly_idx1 = all_idx[:m * n]
            structure_anomaly_idx = anomaly_idx1[:m * n]

            print('Structure disturbance of view 2...')
            ori_num_edge = np.sum(dis_adj)
            for n_ in range(n):
                current_nodes = structure_anomaly_idx[n_ * m:(n_ + 1) * m]
                for i in current_nodes:
                    for j in current_nodes:
                        dis_adj[i, j] = 1.

                dis_adj[current_nodes, current_nodes] = 0.

            num_add_edge = np.sum(dis_adj) - ori_num_edge
            print('Done. {:d} disturbanced nodes are constructed. ({:.0f} edges are added) \n'.format(
                len(structure_anomaly_idx), num_add_edge))

            aug_fts = aug_feature_dropout(features).unsqueeze(0)  # Feature augmentation
            idx = np.random.permutation(nb_nodes)
            shuf_fts = aug_fts[:, idx, :]  # Structure disturbance (view1)

            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_2, lbl_1), 1).cuda()

            dis_adj = process.normalize_adj(dis_adj + sp.eye(dis_adj.shape[0]))
            dis_adj = process.sparse_mx_to_torch_sparse_tensor(dis_adj)
            dis_adjs = torch.tensor(dis_adj, dtype=torch.float32)
            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                aug_fts = aug_fts.cuda()
                dis_adjs = dis_adjs.cuda()

            logits_1 = GDSA(aug_fts, shuf_fts, dis_adjs, sparse=True)
            loss = b_xent(logits_1, lbl)
            print('Loss:', loss)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(GDSA.state_dict(), 'best_model.pth')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break
            loss.backward()
            optimiser_disc.step()

        GDSA_Test = GDSA_test(ft_size, hid_units, nonlinearity)
        GDSA_Test = GDSA_Test.cuda()
        save_model = torch.load('best_model.pth')

        model_dict = GDSA_Test.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
        GDSA_Test.load_state_dict(model_dict)

        aucs = []

        actual_label = torch.tensor(actual_label, dtype = torch.float32)
        actual_adj = actual_adj.cuda()
        logits, _ = GDSA_Test(actual_features, actual_adj, sparse=True)

        preds = torch.tensor(torch.sigmoid(logits).squeeze(0), dtype=torch.float32)
        result = np.where((logits > 0), 1, 0).squeeze(0)
        auc = roc_auc_score(actual_label, preds)

        aucs.append(auc)
        aucs = torch.tensor(aucs, dtype=torch.float32)
        aucs = torch.cat([aucs])

        print(aucs.mean())
        auc_results.append(aucs.mean().cpu().numpy())

    AUC = np.mean(auc_results)
    print('AUC:{:.4f}'.format(AUC))

    with open('GDSAlog_{}.txt'.format(args.dataset), 'a') as f:
        f.write(str(args))
        f.write('\n' + str(np.mean(auc_results)) + '\n')
        f.write(str(np.std(auc_results)) + '\n')


