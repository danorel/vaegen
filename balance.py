import os

import numpy as np

from load_data import get_adata
from dotenv import load_dotenv



import warnings
warnings.simplefilter("ignore", UserWarning)

load_dotenv()
CONDITION_KEY, CELL_TYPE_KEY = os.getenv('CONDITION_KEY'), os.getenv('CELL_TYPE_KEY')




def balance_classes(adata, 
                       class_key=CELL_TYPE_KEY,
                       verbose=False,
                       n=800):
    """
        Perform balancing of classes. 
    """
    np.random.seed(43)
    
    class_name_count = adata.obs.groupby(class_key).size().to_dict()
    class_names, class_counts = list(class_name_count.keys()), list(class_name_count.values())
    n_total = np.sum(class_counts)

    if verbose:
        print("Before balancing:")
        for class_name in class_names:
            n_class = class_name_count[class_name]
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))    

    indexes_classes_balanced = []
    for class_name in class_names:
        index_cls = adata[adata.obs[class_key] == class_name].obs.index.values
        index_balanced = np.random.choice(index_cls, n)
        indexes_classes_balanced.append(index_balanced)
   
    index_balanced = np.concatenate(indexes_classes_balanced)
    adata_balanced = adata[index_balanced].copy()
    n_total = adata_balanced.obs.shape[0]
    
    if verbose:
        print("\nAfter balancing:")
        for class_name in class_names:
            n_class = np.sum(adata_balanced.obs[class_key] == class_name)
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))
        
    return adata_balanced, index_balanced
        


def balance_classes_up(adata, 
                       class_key=CELL_TYPE_KEY,
                       verbose=False):
    """
        Perform balancing of classes. 
    """
    np.random.seed(43)
    
    class_name_count = adata.obs.groupby(class_key).size().to_dict()
    class_names, class_counts = list(class_name_count.keys()), list(class_name_count.values())
    n_total = np.sum(class_counts)
    n_max = np.max(class_counts)

    if verbose:
        print("Before balancing:")
        for class_name in class_names:
            n_class = class_name_count[class_name]
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))    

    indexes_classes_balanced = []
    for class_name in class_names:
        index_cls = adata[adata.obs[class_key] == class_name].obs.index.values
        n_add = n_max - len(index_cls)   # for classes with less instances that we need, add more instances by sampling
        index_add = np.random.choice(index_cls, n_add)
        index_balanced = np.concatenate([index_cls, index_add])
        indexes_classes_balanced.append(index_balanced)
   
    index_balanced = np.concatenate(indexes_classes_balanced)
    adata_balanced = adata[index_balanced].copy()
    n_total = adata_balanced.obs.shape[0]
    
    if verbose:
        print("\nAfter balancing:")
        for class_name in class_names:
            n_class = np.sum(adata_balanced.obs[class_key] == class_name)
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))
        
    return adata_balanced, index_balanced


    
def balance_classes_down(adata, 
                       class_key=CELL_TYPE_KEY,
                       verbose=False):
    """
        Perform balancing of classes. 
    """
    np.random.seed(43)
    
    class_name_count = adata.obs.groupby(class_key).size().to_dict()
    class_names, class_counts = list(class_name_count.keys()), list(class_name_count.values())
    n_total = np.sum(class_counts)
    n_min = np.min(class_counts)

    if verbose:
        print("Before balancing:")
        for class_name in class_names:
            n_class = class_name_count[class_name]
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))    

    indexes_classes_balanced = []
    for class_name in class_names:
        index_cls = adata[adata.obs[class_key] == class_name].obs.index.values
        # n_add = n_max - len(index_cls)   # for classes with less instances that we need, add more instances by sampling
        index_balanced = np.random.choice(index_cls, n_min)
        indexes_classes_balanced.append(index_balanced)
   
    index_balanced = np.concatenate(indexes_classes_balanced)
    adata_balanced = adata[index_balanced].copy()
    n_total = adata_balanced.obs.shape[0]
    
    if verbose:
        print("\nAfter balancing:")
        for class_name in class_names:
            n_class = np.sum(adata_balanced.obs[class_key] == class_name)
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))
        
    return adata_balanced, index_balanced
           



def balance_classes_inside_segments(adata, 
                                    n=800,
                                    class_key=CELL_TYPE_KEY, 
                                    segment_key=CONDITION_KEY, 
                                    verbose=False):
    """
        Perform balancing of classes inside each segment.
        For example, we may want to have class balance in both `control` and `stimulated` groups (in that case segment_key="condition").
    """
    segments = adata.obs[segment_key].drop_duplicates().to_list()
    index_all = []
    for segment in segments:
        adata_segment = adata[adata.obs[segment_key] == segment]
        if verbose:
            print(f'\n----Inside {segment_key} \"{segment}\"' + '-'*30)
        _, index_segment_balanced = balance_classes(adata_segment, class_key, verbose, n=n)
        index_all.append(index_segment_balanced)
    index_balanced = np.concatenate(index_all)
    adata_balanced = adata[index_balanced].copy()
    return adata_balanced


        
def balance_classes_up_inside_segments(adata, 
                                        class_key=CELL_TYPE_KEY, 
                                        segment_key=CONDITION_KEY, 
                                        verbose=False):
    """
        Perform balancing of classes inside each segment.
        For example, we may want to have class balance in both `control` and `stimulated` groups (in that case segment_key="condition").
    """
    segments = adata.obs[segment_key].drop_duplicates().to_list()
    index_all = []
    for segment in segments:
        adata_segment = adata[adata.obs[segment_key] == segment]
        if verbose:
            print(f'\n----Inside {segment_key} \"{segment}\"' + '-'*30)
        _, index_segment_balanced = balance_classes_up(adata_segment, class_key, verbose)
        index_all.append(index_segment_balanced)
    index_balanced = np.concatenate(index_all)
    adata_balanced = adata[index_balanced].copy()
    return adata_balanced


        
def balance_classes_down_inside_segments(adata, 
                                    class_key=CELL_TYPE_KEY, 
                                    segment_key=CONDITION_KEY, 
                                    verbose=False):
    """
        Perform balancing of classes inside each segment.
        For example, we may want to have class balance in both `control` and `stimulated` groups (in that case segment_key="condition").
    """
    segments = adata.obs[segment_key].drop_duplicates().to_list()
    index_all = []
    for segment in segments:
        adata_segment = adata[adata.obs[segment_key] == segment]
        if verbose:
            print(f'\n----Inside {segment_key} \"{segment}\"' + '-'*30)
        _, index_segment_balanced = balance_classes_down(adata_segment, class_key, verbose)
        index_all.append(index_segment_balanced)
    index_balanced = np.concatenate(index_all)
    adata_balanced = adata[index_balanced].copy()
    return adata_balanced


if __name__ == "__main__":
    # demo
    adata = get_adata()
    adata = balance_classes_up_inside_segments(adata, 
                                               verbose=True)
    # Output:
    #
    # ----Inside condition "stimulated"------------------------------
    # Before balancing:
    # CD4T         |  3127 instances, 35%
    # CD14+Mono    |   615 instances, 7%
    # B            |   993 instances, 11%
    # CD8T         |   541 instances, 6%
    # NK           |   646 instances, 7%
    # FCGR3A+Mono  |  2501 instances, 28%
    # Dendritic    |   463 instances, 5%

    # After balancing:
    # CD4T         |  3127 instances, 14%
    # CD14+Mono    |  3127 instances, 14%
    # B            |  3127 instances, 14%
    # CD8T         |  3127 instances, 14%
    # NK           |  3127 instances, 14%
    # FCGR3A+Mono  |  3127 instances, 14%
    # Dendritic    |  3127 instances, 14%

    # ----Inside condition "control"------------------------------
    # Before balancing:
    # CD4T         |  2437 instances, 30%
    # CD14+Mono    |  1946 instances, 24%
    # B            |   818 instances, 10%
    # CD8T         |   574 instances, 7%
    # NK           |   517 instances, 6%
    # FCGR3A+Mono  |  1100 instances, 14%
    # Dendritic    |   615 instances, 8%

    # After balancing:
    # CD4T         |  2437 instances, 14%
    # CD14+Mono    |  2437 instances, 14%
    # B            |  2437 instances, 14%
    # CD8T         |  2437 instances, 14%
    # NK           |  2437 instances, 14%
    # FCGR3A+Mono  |  2437 instances, 14%
    # Dendritic    |  2437 instances, 14%
