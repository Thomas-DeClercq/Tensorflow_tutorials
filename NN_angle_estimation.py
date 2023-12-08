import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import time, random, os

import scipy
from scipy.interpolate import RBFInterpolator, Rbf

import sys
sys.path.insert(1,'/home/thomas/pythonScripts/PressureArray_v2/PR_cython')
from PressureReconstruction_update210623 import calc_Z, Optimization

def getValue(X,Y,Z,xi,yi):
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    point = list(zip(xi,yi))
    #point = [(xi,yi)]
    zi = scipy.interpolate.griddata((X,Y),Z,point)
    zi = np.nan_to_num(zi)
    return zi

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim(bottom=0)
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show(block=False)

def pressureReconstruction_df(df,shape,spacing,n=50,m=10,p_m=0.75,iterations=5):
    X = np.linspace(0, (shape[0]+1)*spacing, n)
    Y = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(X, Y)
    
    X_rbf = np.linspace(0, (shape[0]+1)*spacing, m)
    Y_rbf = np.linspace(0, (shape[1]+1)*spacing, m)
    X_rbf, Y_rbf = np.meshgrid(X_rbf, Y_rbf)
    
    xi = []
    yi = []
    for x_i in range(shape[0]):
        for y_i in range(shape[1]):
            xi.append(spacing+x_i*spacing)
            yi.append(spacing+y_i*spacing)
            
    xi = np.array(xi)
    yi = np.array(yi)
    
    timesteps = int(df['Timestep'].max() + 1)
    it_in = int(df['iteration'].max() + 1)
    it_out = iterations
    
    list_x0 = np.zeros((it_out,8))
    df_new = np.zeros((timesteps*it_out,11))
    
    for t in range(timesteps):
        Z_n = np.zeros(X.shape)
        for i in range(it_in):
            params_i = df.loc[df['Timestep']==t].loc[df['iteration']==i].values[0][2:10]
            Z_1 = calc_Z(X.flatten(),Y.flatten(),*params_i)
            Z_1 = np.reshape(Z_1,X.shape)
            Z_n = Z_n + Z_1
        
        Z_i = getValue(X, Y, Z_n, xi, yi)
        
        rbfi = Rbf(xi,yi,Z_i,function='gaussian') #always +-2.5 ms
        Z_rbf = rbfi(X_rbf,Y_rbf)
        array_rbf = np.array([X_rbf.flatten(),Y_rbf.flatten(),Z_rbf.flatten()]).T
    
        idxs = []
        for idx,el in enumerate(array_rbf):
            if el[0] < spacing or el[0] > shape[0]*spacing:
                idxs.append(idx)
            elif el[1] < spacing or el[1] > shape[1]*spacing:
                idxs.append(idx)
        array_rbf = np.delete(array_rbf,idxs,axis=0).T
        array, list_E = Optimization(xi,yi,Z_i.copy(),shape,spacing, array_rbf, list_x0, n=round(p_m*array_rbf.shape[1]),it_max=it_out,t_max=50)
        list_x0 = np.zeros((it_out,8))
        for idx,E in enumerate(list_E):
            list_x0[idx] = E.x
            
        for idx1 in range(it_out):
            df_new[t*it_out+idx1,0] = t
            df_new[t*it_out+idx1,1] = idx1
            df_new[t*it_out+idx1,10] = df.loc[df['Timestep']==t].values[-1][-1]
            for idx2 in range(8):
                df_new[t*it_out+idx1,2+idx2] = list_x0[idx1,idx2]
                
    columns = ['Timestep','iteration','p0','std','lx','ly','r_curve','theta','x0','y0','angle']
    df_new = pd.DataFrame(df_new,columns=columns)        
    return df_new

def pressureReconstruction_df_withNoise(df,shape,spacing,max_noise=0.1,n=50,m=10,p_m=0.75,iterations=5):
    X = np.linspace(0, (shape[0]+1)*spacing, n)
    Y = np.linspace(0, (shape[1]+1)*spacing, n)
    X, Y = np.meshgrid(X, Y)
    
    X_rbf = np.linspace(0, (shape[0]+1)*spacing, m)
    Y_rbf = np.linspace(0, (shape[1]+1)*spacing, m)
    X_rbf, Y_rbf = np.meshgrid(X_rbf, Y_rbf)
    
    xi = []
    yi = []
    for x_i in range(shape[0]):
        for y_i in range(shape[1]):
            xi.append(spacing+x_i*spacing)
            yi.append(spacing+y_i*spacing)
            
    xi = np.array(xi)
    yi = np.array(yi)
    
    timesteps = int(df['Timestep'].max() + 1)
    it_in = int(df['iteration'].max() + 1)
    it_out = iterations
    
    list_x0 = np.zeros((it_out,8))
    df_new = np.zeros((timesteps*it_out,11))
    
    for t in range(timesteps):
        Z_n = np.zeros(X.shape)
        for i in range(it_in):
            params_i = df.loc[df['Timestep']==t].loc[df['iteration']==i].values[0][2:10]
            Z_1 = calc_Z(X.flatten(),Y.flatten(),*params_i)
            Z_1 = np.reshape(Z_1,X.shape)
            Z_n = Z_n + Z_1
        
        Z_i = getValue(X, Y, Z_n, xi, yi)
        for i,z in enumerate(Z_i):
            Z_i[i] = z + random.uniform(-max_noise*z,max_noise*z)
        
        rbfi = Rbf(xi,yi,Z_i,function='gaussian') #always +-2.5 ms
        Z_rbf = rbfi(X_rbf,Y_rbf)
        array_rbf = np.array([X_rbf.flatten(),Y_rbf.flatten(),Z_rbf.flatten()]).T
    
        idxs = []
        for idx,el in enumerate(array_rbf):
            if el[0] < spacing or el[0] > shape[0]*spacing:
                idxs.append(idx)
            elif el[1] < spacing or el[1] > shape[1]*spacing:
                idxs.append(idx)
        array_rbf = np.delete(array_rbf,idxs,axis=0).T
        array, list_E = Optimization(xi,yi,Z_i.copy(),shape,spacing, array_rbf, list_x0, n=round(p_m*array_rbf.shape[1]),it_max=it_out,t_max=50)
        list_x0 = np.zeros((it_out,8))
        for idx,E in enumerate(list_E):
            list_x0[idx] = E.x
            
        for idx1 in range(it_out):
            df_new[t*it_out+idx1,0] = t
            df_new[t*it_out+idx1,1] = idx1
            df_new[t*it_out+idx1,10] = df.loc[df['Timestep']==t].values[-1][-1]
            for idx2 in range(8):
                df_new[t*it_out+idx1,2+idx2] = list_x0[idx1,idx2]
                
    columns = ['Timestep','iteration','p0','std','lx','ly','r_curve','theta','x0','y0','angle']
    df_new = pd.DataFrame(df_new,columns=columns)        
    return df_new

def build_and_compile_model(norm,custom=False):
  if not custom:
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
  else:
        hidden_layers = 3
        npl = 512
        dropout = 0.3
        learning_rate = 0.0001
        layers_list = [norm]
        for _ in range(hidden_layers):
            layers_list.append(layers.Dense(npl, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
            if dropout > 0:
                layers_list.append(layers.Dropout(dropout))
        layers_list.append(layers.Dense(1))

        model = keras.Sequential(layers_list)

        model.compile(loss='mean_absolute_error',
                        optimizer=tf.keras.optimizers.Adam(learning_rate))
  return model

if __name__ == "__main__":
    t0 = time.time()
    shape = [4,8]
    spacing = 4.5
    plot_contour = False
    n = 50
    dir_data = "/home/thomas/pythonScripts/PressureArray_v2/angle_estimation/Data_RT/TrainingData/"
    training_data = 'Data_realistic_DxDy'
    test_data = 'Data_realistic_DxDy'
    save = True
    plot = True

    PR_it = 1
    training_size = 100000
    noise = 0

    all_training_files = os.listdir(f"{dir_data}/{training_data}")
    random.shuffle(all_training_files)
    training_files = all_training_files[:training_size]

    test_files = all_training_files[-1000:]
    #all_validation_files = os.listdir(f"{dir_data}/{test_data}")
    
    #error_array = np.zeros((len(all_files),2))
    #file = all_files[10]
    
    time_steps = 1 #currently max of 1, can later be increased --> doens't improve accuracy, training time x2
    
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    
    t0 = time.time()
    progress_counter = 0.1
    for idx,file in enumerate(training_files):
        if (idx/len(training_files)) >= progress_counter:
            print(f'{round(progress_counter*100)}% done')
            progress_counter = progress_counter + 0.1
        #df = pd.read_csv("./"+data+"/"+file)
        df = pd.read_csv(f"{dir_data}/{training_data}/{file}")
        
        df = pressureReconstruction_df_withNoise(df,shape,spacing,max_noise=noise,iterations=PR_it)
        
        amount_of_timesteps = int(df["Timestep"].max()-time_steps+1)
        
        for t in range(0,amount_of_timesteps+1):
            df_t0 = df.loc[df["Timestep"] == 0]
            df_ti = df.loc[df["Timestep"] >= t]
            df_ti = df_ti.loc[df_ti["Timestep"] <= t+time_steps-1]
            #df_ti = df.loc[df["Timestep"] == t]
            df_t = pd.concat([df_t0,df_ti])
            #df_t = df.loc[df["Timestep"] >= t]
            #df_t = df_t.loc[df_t["Timestep"] <= t+time_steps]
            #important to check these df's when adding complexity
            
            try:
                current_timestep = df_t["Timestep"].max()
                x_train_new = df_t.values.T[2:10].T.flatten()
                x_train = np.vstack((x_train,x_train_new))
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_train = np.vstack((y_train,angles))

            except: #only for the first time ever
                current_timestep = df_t["Timestep"].max()
                x_train_new = df_t.values.T[2:10].T.flatten()
                x_train = np.append(x_train,x_train_new)
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_train = np.append(y_train,angles)
    
    #print(pd.DataFrame(x_train.T[8:].T,columns=['p0','std','lx','ly','r_curve','theta','x0','y0']).describe())
    print('Training data converted')

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)
    print(normalizer.mean.numpy())

    dnn_model = build_and_compile_model(normalizer,custom=True)
    dnn_model.summary()
     
    #t0 = time.time()
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)]
    history = dnn_model.fit(
                    x_train,
                    y_train,
                    validation_split=0.2,
                    verbose=0, epochs=10000,
                    callbacks=callback)
    print(f'training time : {(time.time()-t0)/60} min')
    plot_loss(history)

    t0 = time.time()
    progress_counter = 0.1
    for idx,file in enumerate(test_files):
        if (idx/len(test_files)) >= progress_counter:
            print(f'{round(progress_counter*100)}% done')
            progress_counter = progress_counter + 0.1
        #df = pd.read_csv("./"+data+"/"+file)
        df = pd.read_csv(f"{dir_data}/{test_data}/{file}")
        
        df = pressureReconstruction_df_withNoise(df,shape,spacing,max_noise=noise,iterations=PR_it)
        
        amount_of_timesteps = int(df["Timestep"].max()-time_steps+1)
        
        for t in range(0,amount_of_timesteps+1):
            df_t0 = df.loc[df["Timestep"] == 0]
            df_ti = df.loc[df["Timestep"] >= t]
            df_ti = df_ti.loc[df_ti["Timestep"] <= t+time_steps-1]
            #df_ti = df.loc[df["Timestep"] == t]
            df_t = pd.concat([df_t0,df_ti])
            #df_t = df.loc[df["Timestep"] >= t]
            #df_t = df_t.loc[df_t["Timestep"] <= t+time_steps]
            #important to check these df's when adding complexity
            
            try:
                current_timestep = df_t["Timestep"].max()
                x_test_new = df_t.values.T[2:10].T.flatten()
                x_test = np.vstack((x_test,x_test_new))
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_test = np.vstack((y_test,angles))

            except: #only for the first time ever
                current_timestep = df_t["Timestep"].max()
                x_test_new = df_t.values.T[2:10].T.flatten()
                x_test = np.append(x_test,x_test_new)
                angles = df_t.loc[df["Timestep"]==current_timestep]['angle'].values[0] #- df_t.loc[df["Timestep"]==(current_timestep-1)]['angle'].values[0]
                #angles = df_t['angle'].values[-1] - df_t['angle'].values[-2]
                y_test = np.append(y_test,angles)

    #print(pd.DataFrame(x_test.T[8:].T,columns=['p0','std','lx','ly','r_curve','theta','x0','y0']).describe())
    y_pred = dnn_model.predict(x_test)

    print("error calculation")
    error_array_rel = y_test.copy().flatten()
    error_array_rel = np.vstack((error_array_rel,y_pred.flatten()))
    error_array_rel = np.vstack((error_array_rel,abs(error_array_rel[1] - error_array_rel[0])))

    print(f"average: {np.average(error_array_rel[2])}")
    print(f"median: {np.median(error_array_rel[2])}")
    print(f"std: {np.std(error_array_rel[2])}")

    if save:
        dnn_model.save(f'NN_{PR_it}it_{int(training_size/1000)}k_custom')

    if plot:
        #timesteps = int(data.split('_')[1])
        timesteps = 10
        for i in range(4):
            fig1 = plt.figure()
            ax = plt.subplot(111)
            ax.plot(error_array_rel[0][i*timesteps:(i+1)*timesteps-1],label='real angle')
            ax.plot(error_array_rel[1][i*timesteps:(i+1)*timesteps-1],label='estimated angle')
            ax.legend()
            plt.show(block=False)

        fig2 = plt.figure()
        ax2 = plt.subplot(211)
        ax2.plot(error_array_rel[0],error_array_rel[1],'.')
        ax2.set_xlabel('real angle')
        ax2.set_ylabel('estimated angle')
        ax2.plot([0,90],[0,90],color='k')
        ax3 = plt.subplot(212)
        ax3.boxplot(error_array_rel[2], showfliers=False)
        plt.show(block=False)

        input()


