import copy
import os

import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from matplotlib import tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm, trange
from tqdm.contrib import tzip
import numpy as np

plt.rcParams.update({'font.size': 18})

def Plot(A,P,title):
    t = np.arange(0,np.shape(A)[0]) 
    plt.figure(figsize = (10,8))   
    plt.plot(t,A,'r',label='Actual Temperature')
    plt.plot(t,P,'--' ,label='Predicted Temperature')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.legend(loc = 'lower right')
    plt.show()


def ErrorPlot(E):
    t = np.arange(0,np.shape(E)[0])    
    plt.plot(t,E,'r')
    #plt.plot(t,P,'--' ,label='Predicted Temperature')
    plt.title("Error Plot")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.show()


def make_animation(loader ,gs, pred, evl, path, name , skip = 1, save_anim = True, plot_variables = False):
    '''
    input gs is a dataloader and each entry contains attributes of many timesteps.

    '''
    # coordinates at which you want to plot temprature at that point throughout the LPBF process
    x = 0   #keep it near zero as much as possible
    y = 15

    print('Generating temperature fields...')
    A_temp = []
    P_temp = []
    Error = []

    for num in range(0,len(loader),1):
        
        step = num
        traj = 0
        # use max and min temperature of gs dataset at the first step for both 
        # the first column is temperature 

        count = 0
        pos = loader[step].mesh_pos 
        faces = loader[step].cells

        ind = np.where((pos[:,0]>x-0.75) & (pos[:,0]<0.75+x) & (pos[:,1]>y-0.75) & (pos[:,1]<0.75+y))
       
        a_temperature = gs[step][ind[0][0], 0:1].numpy()
        p_temperature = pred[step][ind[0][0], 0:1].numpy()
        error = evl[step][ind[0][0], 0:1].numpy()

        
        x = int(np.array(pos[ind[0][0],0]))
        y = int(np.array(pos[ind[0][0],1]))
        #print('y: ',np.array(pos[ind[0][0],1]), ' T: ',a_temperature)
        
        if (step%19 == 0 or step == (len(loader)-1)):
            title = "Plotting Temperature at x: " + str(x)+ " y: "+ str(y)
            Plot(A_temp,P_temp,title)
            #ErrorPlot(Error) 
            A_temp = []
            P_temp = [] 
            Error = []
            A_temp.append(a_temperature)
            P_temp.append(p_temperature)
            Error.append(error)  

        elif (step%19 != 0 or step == 0):
            A_temp.append(a_temperature)
            P_temp.append(p_temperature)
            Error.append(error)

        
        
def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize*std_vec+mean_vec

def visualize(loader, best_model, file_dir, args, device, gif_name, stats_list,
              delta_t = 0.01, skip = 1):

    best_model.eval()
    #device = args.device
    viz_data = []
    gs_data = []
    eval_data = []
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
            std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))

    for data, in tqdm(zip(loader), total=len(loader)):
        data=data.to(device) 
        with torch.no_grad():
            pred = best_model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            # pred gives the learnt accelaration between two timsteps
            # next_vel = curr_vel + pred * delta_t  
            v_data = data.x[:, 0:1] + unnormalize(pred[:],mean_vec_y,std_vec_y)  
            g_data = data.x[:, 0:1] + data.y
            e_data = abs(data.y - unnormalize(pred[:],mean_vec_y,std_vec_y))
        viz_data.append(v_data)
        gs_data.append(g_data)
        eval_data.append(e_data)

    print(np.shape(viz_data))
    make_animation(loader, gs_data, viz_data, eval_data, file_dir, gif_name, skip, True, False)

