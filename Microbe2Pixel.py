# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 21:39:43 2022

@author: basvo
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import pandas as pd
import keras
from tensorflow.keras import layers, optimizers, losses, Model, applications
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import copy


startTime = time.time()
explain_check = True
retrain = True
train = True
stability_samples_run_number = 25
val_data_ratio =  0.2
train_data_ratio = 0.8
test_data_ratio = 0.2 
subset_percentage = 100
n_subset_runs = 1
root = r'data/images'

#Initialization for the explanation algorithms
cluster_df = pd.read_csv('bugs_locations_im+abundances.csv',index_col = 'Unnamed: 0')

# Remove constant values from that dataframe
coord_df = cluster_df[['X','Y']]
abundances_df = cluster_df.drop(['X','Y'],axis=1)

bugs = list(coord_df.index)
sample_names = list(abundances_df.columns)

X_coords = np.asarray(coord_df)
max_coords = np.max(X_coords)
n_clusters = 50

X = list([])
y = list([])
IDs = list([])
classes = [0,1]

check = True
for c in classes:
    curroot = root+r'/class_'+str(c)
    for path, subdirs, files in os.walk(curroot):
        for name in files:
            img_path = os.path.join(path,name) 
            img = load_img(img_path)
            img = img_to_array(img)
            
            X.append(img)
            y.append(c)
            IDs.append(name)
            
X = np.asarray(X,dtype='int')
y = np.asarray(y)
IDs = np.asarray(IDs)


for subset_run in range(n_subset_runs):
    np.random.seed(subset_run)
    output_path = 'output/transfer_learning/'

    pred_labels = []
    true_labels = []
    train_retrain = []

    if train == False:
        train_retrain = np.asarray(pd.read_csv(output_path+'train_retrain.csv',index_col = 'Unnamed: 0')) 
    print(train_retrain)

    StratShufSpl=StratifiedShuffleSplit(stability_samples_run_number,
                                    test_size=test_data_ratio,random_state=subset_run)
    val_split = StratifiedShuffleSplit(1,
                                    test_size=test_data_ratio,random_state=subset_run)

    shuffle_counter=0
    auc_list = []
    for samples,test in StratShufSpl.split(X,y):
    
        samples = samples[:int(len(samples)*(subset_percentage/100))]
        
        shuffle_counter+=1
        
        x_train,y_train=X[samples],y[samples]
        x_test,y_test=X[test],y[test]
        test_ids =   IDs[test]

        n_val = int(len(samples)*val_data_ratio)
        
        for training, validation in val_split.split(x_train,y_train):
            x_val,y_val = x_train[validation],y_train[validation]
            x_train,y_train = x_train[training],y_train[training]
        
        # Build the model
        base_model = applications.EfficientNetB7(weights = 'imagenet', include_top = False, input_shape = (448, 449, 3))
        base_model.trainable = False

        inputs = keras.Input(shape=((448, 449, 3)))

        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        predictions = layers.Dense(1, activation = 'sigmoid')(x)
        checkpoint_filepath = r'output/transfer_learning/checkpoints/checkpoint_'+'run'+str(subset_run)+'_'+str(shuffle_counter)+'.hdf5'
        
        head_model = Model(inputs = inputs, outputs = predictions)
        head_model.compile(optimizer=optimizers.Adam(learning_rate=0.5e-3), 
                        loss=losses.binary_crossentropy, 
                        metrics=keras.metrics.AUC(name='auc')
                        )
        if train:
            print('TRAINING THE TOP MODEL')
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True,
                                                                    monitor='val_auc',
                                                                    verbose=1,
                                                                    mode='max',
                                                                    save_best_only=True)                                                                
            callbacks = [EarlyStopping(monitor = 'val_auc', 
                                    patience = 10,
                                    mode='max',
                                    verbose=1),
                                    model_checkpoint_callback]
            history = head_model.fit(x_train, y_train, 
                                validation_data=(x_val, y_val), 
                                batch_size=8, 
                                epochs=50,
                                use_multiprocessing=True,
                                callbacks = callbacks)
            
            # load the weights with the best val_auc
            head_model.load_weights(checkpoint_filepath)

        if retrain:
            for layer in base_model.layers[:-5]:
                layer.trainable = False
            
            for layer in base_model.layers[-5:]:
                layer.trainable = True
            
            head_model.compile(optimizer=keras.optimizers.Adam(0.1e-5),  
                loss=losses.binary_crossentropy,
                metrics=keras.metrics.AUC(name='auc'))
            print('FINE TUNING THE ENTIRE MODEL')
            
            history_retrain = head_model.fit(x_train, y_train, 
                                    epochs=50, 
                                    batch_size = 8, 
                                    validation_data=(x_val, y_val), 
                                    callbacks=callbacks)
            
            # load the weights with the best val_auc
            if np.max(history.history['val_auc']) < np.max(history_retrain.history['val_auc']):
                head_model.load_weights(checkpoint_filepath)
                train_retrain.append('retrain')
            else:
                base_model.trainable = False
                head_model.compile(optimizer=optimizers.Adam(learning_rate=0.5e-3), 
                        loss=losses.binary_crossentropy, 
                        metrics=keras.metrics.AUC(name='auc')
                        )
                head_model.load_weights(checkpoint_filepath)
                train_retrain.append('train')
    
        if explain_check:
            print('GETTING EXPLANATIONS')
            if train == False:
                if train_retrain[shuffle_counter-1] == 'retrain':
                    for layer in base_model.layers[:-5]:
                        layer.trainable = False
                
                    for layer in base_model.layers[-5:]:
                        layer.trainable = True
                
                    head_model.compile(optimizer=keras.optimizers.Adam(0.1e-5),  
                        loss=losses.binary_crossentropy,
                        metrics=keras.metrics.AUC(name='auc'))
                    head_model.load_weights(checkpoint_filepath)
                else:
                    head_model.load_weights(checkpoint_filepath)

            for i in tqdm(range(len(x_test))):           
                col = np.asarray(abundances_df[test_ids[i][:-5]])  
                
                map_3d = np.c_[X_coords,col]    

                model = KMeans(n_clusters = n_clusters)
                kmeans = model.fit(map_3d)
                clusters = model.predict(map_3d)
        
                clusters_in_im = np.c_[X_coords,clusters]
                
                num_perturb=200
                perturbation = np.random.binomial(1, 0.5, size=(num_perturb, n_clusters))
                
                original = np.zeros((1,n_clusters))
                distances = pairwise_distances(perturbation,original).ravel()
                
                perturbed_ims= []
                
                for j in range(num_perturb):
                    #deepcopy the original to a image to perturb
                    new_img = copy.deepcopy(x_test[i])             
                    
                    #carry out the permutations in the permutation vector
                    for k in range(n_clusters):
                            curr_perturbation =  perturbation[j,k]
                            if perturbation[j,k] == 0:
                                pixels_in_cluster = clusters_in_im[clusters_in_im[:,2]==k]
                                for l in range(len(pixels_in_cluster)):
                                    new_img[pixels_in_cluster[l,0],pixels_in_cluster[l,1],:]=0
                    
                    perturbed_ims.append(new_img)
                
                # Get predictions for the perturbed images
                perturbed_ims = np.asarray(perturbed_ims)
                perturbed_preds = head_model.predict(perturbed_ims,batch_size=8)
                
                # run the distances through the kernel function to get weighting
                weights = 1/distances
                
                #Fit the linear model
                simpler_model = LinearRegression()
                simpler_model.fit(X=perturbation, y=perturbed_preds, sample_weight=weights)
                coeff = simpler_model.coef_
                
                feats_coeff_matched = []
                for j in range(len(clusters)):
                    curclust = clusters[j]
                    feats_coeff_matched.append(coeff[0,curclust])
                
                sample_importances = np.c_[list(coord_df.index),np.asarray(feats_coeff_matched,'float')]
                
                if i == 0 :
                    global_importances = np.asarray(feats_coeff_matched,'float')
                else:
                    global_importances =np.c_[global_importances,np.asarray(feats_coeff_matched,'float')]

            imp_df = pd.DataFrame(global_importances,index=list(coord_df.index),columns=test_ids)    
            imp_df.to_csv(output_path+'importances_run'+str(subset_run)+'shuffle_'+str(shuffle_counter)+'.csv')
        
        if train == False and explain_check==False:
            if train_retrain[shuffle_counter-1] == 'retrain':
                for layer in base_model.layers[:-5]:
                        layer.trainable = False
                
                for layer in base_model.layers[-5:]:
                        layer.trainable = True
                
                head_model.compile(optimizer=keras.optimizers.Adam(0.1e-5),  
                                    loss=losses.binary_crossentropy,
                                    metrics=keras.metrics.AUC(name='auc'))
                head_model.load_weights(checkpoint_filepath)
            else:
                head_model.load_weights(checkpoint_filepath)

        print('\n','\n','Evaluating the model:')

        y_pred = head_model.predict(x_test,batch_size=1)
        pred_labels.append(y_pred)
        true_labels.append(y_test)
        fpr,tpr,thresholds = roc_curve(y_test,y_pred)
        cur_auc = auc(fpr,tpr)
        auc_list.append(cur_auc)
        
        endTime = time.time()
        runTime=endTime-startTime
        print('')
        print(('Shuffle %i is done!!!' %shuffle_counter),('   %.2f seconds' %runTime))
        print('auc is equal to',cur_auc)
        if train:
            pred_labels_arr = np.asarray(pred_labels)
            true_labels_arr = np.asarray(true_labels)

            pred_labels_arr = pred_labels_arr.reshape(pred_labels_arr.shape[0],pred_labels_arr.shape[1])
            true_labels_arr = true_labels_arr.reshape(true_labels_arr.shape[0],true_labels_arr.shape[1])

            pred_df = pd.DataFrame(pred_labels_arr)
            true_df = pd.DataFrame(true_labels_arr)

            pred_df.to_csv(output_path+'pred.csv')
            true_df.to_csv(output_path+'true.csv')

    y_pred = pd.read_csv(output_path+'pred.csv')
    y_true = pd.read_csv(output_path+'true.csv')

    if train and retrain:
        train_retrain_df = pd.DataFrame(train_retrain)
        train_retrain_df.to_csv(output_path+'train_retrain.csv')

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Luck', alpha=.8)
    for i in range(len(y_pred)):
        cur_pred = y_pred.iloc[i,:]
        cur_true = y_true.iloc[i,:]
        fpr, tpr, thresholds = roc_curve(cur_true,cur_pred,pos_label=1)
        
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        
        ax.plot(fpr,tpr,lw=0.5,alpha=0.5)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ax.plot(mean_fpr, mean_tpr, color='k',
            lw=2, alpha=.8,label=r'Mean ROC (AUC = %0.2f $/pm$ %0.2f)' % (np.mean(auc_list), np.std(auc_list)))

    # Fill the +- 1 std interval around the average ROC
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$/pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(output_path+'Mean_AUC.pdf')

    if train:
        print("average AUC of this round is equal to",np.mean(auc_list))