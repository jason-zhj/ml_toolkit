from visualization_toolkit.feature_visualize import *

if __name__ == "__main__":
    CNN_feature_dict = process_CNN_feature_file("cnn_feats_aftadap.csv")
    print("Finish preprocessing")
    visualize_CNN_features(cnn_feature_dict=CNN_feature_dict,v_domains=["amazon","non-amazon"],
                           v_labels=['2','3','4'],domain_colors={"amazon":"red","non-amazon":"blue"},
                           label_symbols={"0":"O","1":"1","2":"2","3":"3","4":"4"})