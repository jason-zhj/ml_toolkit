from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def process_CNN_feature_file(filename):
    """
    :param filename: file should be \t delimited, in the format of filename \t label \t feature
    :return:{"domain_name":{"label":[feature_vector1,feature_vector2,...],...}}
    """
    return_dict = {}
    with open(filename) as f:
        lines = f.readlines()[1:]
        for line in lines:
            file,label,feature = line.split("\t")
            domain = file
            feature_vector = [float(i) for i in feature.split(",")]
            if (domain in return_dict.keys()):
                domain_dict = return_dict[domain]
                if (label in domain_dict.keys()):
                    domain_dict[label].append(feature_vector)
                else:
                    domain_dict[label] = [feature_vector]
            else:
                return_dict[domain] = {label:[feature_vector]}
    return return_dict


def _plot_embedding(X, y, d, title=None,show=True,save_to=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    # plt.figure(figsize=(10,10))
    # ax = plt.subplot(111)
    plt.figure()
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=d[i],
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    if (show):
        plt.show()
    if (save_to):
        plt.savefig(save_to)

def save_2d_features(features,domains,labels,save_dest):
    "save a tab delimited csv file, in the format of domain\tlabel\tx\ty containing features"
    assert len(features) == len(domains) == len(labels)
    to_write = [] # list of domain \t label \t x,y
    for i in range(len(features)):
        x,y = features[i]
        to_write.append("{}\t{}\t{},{}".format(domains[i],labels[i],x,y))

    with open(save_dest,"w") as f:
        f.write("\n".join(to_write))


def visualize_CNN_features(cnn_feature_dict,v_domains,v_labels,domain_colors,label_symbols,save_dest=None):
    """
    :param cnn_feature_dict: {"domain_name":{"label":[feature_vector1,feature_vector2,...],...}}, output of `process_CNN_feature`
    Plot CNN features using T-SNE
    :param v_domains: list of domains to be visualized
    :param v_labels: list of labels to be visualized
    :param domain_colors: dict {domain_name:color}, used for plotting
    :param label_symbols: dict {label_name:char}, used for plotting
    """
    feature_stack = []
    domain_stack = []
    label_stack = []
    for domain in v_domains:
        domain_dict = cnn_feature_dict[domain]
        for label in v_labels:
            feature_ls = domain_dict[label]
            feature_stack += feature_ls
            domain_stack += [domain for _ in range(len(feature_ls))]
            label_stack += [label for _ in range(len(feature_ls))]
    print("start running tsne")
    # run TSNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    features_2d = tsne.fit_transform(feature_stack)
    print("finish tsne")
    # save the features
    if (save_dest):
        save_2d_features(features=features_2d,domains=domain_stack,labels=label_stack,save_dest=save_dest)
    # plot scatter
    _plot_embedding(X=np.array(features_2d),
                    y=[label_symbols[x] for x in label_stack],
                    d=[domain_colors[x] for x in domain_stack],
                    title="TSNE Visualization")
    plt.show()
