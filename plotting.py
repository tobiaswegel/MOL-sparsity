import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
large_font_size: int = 12
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
cs = list(mcolors.TABLEAU_COLORS)
plt.rcParams["font.size"] = large_font_size
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

gt_color = 'grey'
gt_style = 'solid'
gt_marker = 'x'


colors = {
    "Method1_Impl1": "#1f77b4",
    "Method1_Impl2": "#aec7e8",
    "Method2_Impl1": "#ff7f0e",
    "Method2_Impl2": "#ffbb78",
}

dr_ensemble_color =  'tab:orange'
dr_ensemble_style = 'solid'
dr_ensemble_marker = 'v'

ts_ensemble_color = 'tab:blue'
ts_ensemble_style = 'solid'
ts_ensemble_marker = '^'

dr_hypernetwork_color = 'tab:purple'
dr_hypernetwork_style = 'dotted'
dr_hypernetwork_marker = '+'

ts_hypernetwork_color = 'tab:red'
ts_hypernetwork_style = 'dotted'
ts_hypernetwork_marker = '*'

def plot_Pareto_fronts(errors, title, save = False):
    fig, ax = plt.subplots(figsize=(4,3))

    num_experiments = errors['gt'].shape[0]

    for i in range(num_experiments):
        ax.plot(errors['gt'][i,:,0],errors['gt'][i,:,1],c=gt_color,alpha=0.08)
        ax.plot(errors['dr_ensemble'][i,:,0],errors['dr_ensemble'][i,:,1],c=dr_ensemble_color,alpha=0.08)
        ax.plot(errors['ts_ensemble'][i,:,0],errors['ts_ensemble'][i,:,1],c=ts_ensemble_color,alpha=0.08)
        ax.plot(errors['dr_hypernetwork'][i,:,0],errors['dr_hypernetwork'][i,:,1],c=dr_hypernetwork_color,alpha=0.08)
        ax.plot(errors['ts_hypernetwork'][i,:,0],errors['ts_hypernetwork'][i,:,1],c=ts_hypernetwork_color,alpha=0.08)
   
    line1, = ax.plot(np.average(errors['gt'][:,:,0],axis=0),np.average(errors['gt'][:,:,1],axis=0),c=gt_color,linestyle = gt_style, marker= gt_marker,alpha=1)
    line2, = ax.plot(np.average(errors['dr_ensemble'][:,:,0],axis=0),np.average(errors['dr_ensemble'][:,:,1],axis=0),c=dr_ensemble_color,linestyle = dr_ensemble_style, marker = dr_ensemble_marker ,alpha=1)
    line3, = ax.plot(np.average(errors['ts_ensemble'][:,:,0],axis=0),np.average(errors['ts_ensemble'][:,:,1],axis=0),c=ts_ensemble_color,linestyle = ts_ensemble_style, marker = ts_ensemble_marker, alpha=1)
    line4, = ax.plot(np.average(errors['dr_hypernetwork'][:,:,0],axis=0),np.average(errors['dr_hypernetwork'][:,:,1],axis=0),c=dr_hypernetwork_color, linestyle = dr_hypernetwork_style, marker = dr_hypernetwork_marker, alpha=1)
    line5, = ax.plot(np.average(errors['ts_hypernetwork'][:,:,0],axis=0),np.average(errors['ts_hypernetwork'][:,:,1],axis=0),c=ts_hypernetwork_color, linestyle = ts_hypernetwork_style, marker = ts_hypernetwork_marker, alpha=1)
    
    ax.legend([line1,line2,line3,line4,line5],['Pareto front', 'directly regularized ensemble','two-stage estimator ensemble','directly regularized hypernetwork','two-stage hypernetwork'])
    ax.set_xlabel('square loss on first distribution')
    ax.set_ylabel('square loss on second distribution')
    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xlim(left=-0.1)
    ax.set_ylim(bottom=-0.1)

    plt.title(title)

    #ax.scatter(control[0],control[1])
    if save:
        plt.savefig('multiple_regression.png',dpi=500,bbox_inches='tight')
    plt.show()



def plot_sparsity_unlabeled_matrix(errors, N_array, s_array, save = False):
    excess_loss = np.array(
        [[np.log((np.sum(errors[(s, N)]['ts_loss'])-np.sum(errors[(s, N)]['gt_loss']))/2) for N in N_array] for s in s_array]
        )

    # Plot the results
    fig, axs = plt.subplots(1, 1, figsize=(4.5, 3.5))

    extent = [N_array[0], N_array[-1], s_array[0], s_array[-1]]
    dx, = np.diff(extent[:2])/(excess_loss.shape[1]-1)
    dy, = -np.diff(extent[2:])/(excess_loss.shape[0]-1)
    extent = [N_array[0]-dx/2, N_array[-1]+dx/2, s_array[0]+dy/2, s_array[-1]-dy/2]

    # Plot ours losses
    im3 = axs.imshow(excess_loss, cmap='plasma',extent=extent, aspect='auto')
    axs.set_title('log-excess scalarized loss')
    axs.set_xlabel(r'number of additional unlabeled points $N$')
    axs.set_ylabel(r'sparsity $s$')
    cbar = fig.colorbar(im3, ax=axs)
    axs.annotate(r'$\log\mathcal{E}_{\lambda}$',xy=(55,0),annotation_clip=False)

    axs.set_xticks(N_array)
    axs.set_xticklabels(N_array)
    axs.set_yticks(np.flip(s_array))
    axs.set_yticklabels(s_array)

    plt.gca().invert_yaxis()

    plt.tight_layout()
    if save:
        plt.savefig('sparsity_vs_unlabeleddata.png',bbox_inches='tight',dpi=500)
    plt.show()


def plot_fairness_Pareto_fronts(errordict, title, xlabels, save, xmax, xmin, ymax):
    num_plots = len(errordict)
    num_experiments = next(iter(errordict.values()))['gt'].shape[0]

    fig, ax = plt.subplots(ncols = num_plots,nrows=1,figsize=(num_plots*3,2))

    for pltindex, (key, errors) in enumerate(errordict.items()):

        for i in range(num_experiments):
            ax[pltindex].plot(errors['gt'][i,:,0],errors['gt'][i,:,1],c=gt_color,alpha=0.08)
            ax[pltindex].plot(errors['dr_ensemble'][i,:,0],errors['dr_ensemble'][i,:,1],c=dr_ensemble_color,alpha=0.08)
            ax[pltindex].plot(errors['ts_ensemble'][i,:,0],errors['ts_ensemble'][i,:,1],c=ts_ensemble_color,alpha=0.08)
    
        line1, = ax[pltindex].plot(np.average(errors['gt'][:,:,0],axis=0),np.average(errors['gt'][:,:,1],axis=0),c=gt_color,linestyle = gt_style, marker= gt_marker,alpha=1)
        line2, = ax[pltindex].plot(np.average(errors['dr_ensemble'][:,:,0],axis=0),np.average(errors['dr_ensemble'][:,:,1],axis=0),c=dr_ensemble_color,linestyle = dr_ensemble_style, marker = dr_ensemble_marker ,alpha=1)
        line3, = ax[pltindex].plot(np.average(errors['ts_ensemble'][:,:,0],axis=0),np.average(errors['ts_ensemble'][:,:,1],axis=0),c=ts_ensemble_color,linestyle = ts_ensemble_style, marker = ts_ensemble_marker, alpha=1)
        
        ax[pltindex].set_xlabel(xlabels[pltindex])
        if pltindex == 0:
            ax[pltindex].set_ylabel('demographic parity on test data')
        ax[pltindex].spines[['right', 'top']].set_visible(False)

        #ax[pltindex].yaxis.set_major_formatter(ticker.ScalarFormatter())
        #ax[pltindex].tick_params(axis='y', which='major', style='sci')
        ax[pltindex].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        
        if key == 'communities':
            ax[pltindex].set_title('Communities')
        elif key == 'adult':
            ax[pltindex].set_title('Adult')
        elif key == 'hsls':
            ax[pltindex].set_title('HSLS')
        elif key == 'enem':
            ax[pltindex].set_title('ENEM')
        else: 
            raise RuntimeError('key not known')

        if not (xmin == None or xmax == None or ymax == None):
            ax[pltindex].set_xlim(left=xmin[pltindex],right=xmax[pltindex])
            ax[pltindex].set_ylim(top=ymax[pltindex])

        # Save each subplot as a separate file
        if save:
            single_fig, single_ax = plt.subplots(figsize=(3, 2))
            for i in range(num_experiments):
                single_ax.plot(errors['gt'][i,:,0],errors['gt'][i,:,1],c=gt_color,alpha=0.08)
                single_ax.plot(errors['dr_ensemble'][i,:,0],errors['dr_ensemble'][i,:,1],c=dr_ensemble_color,alpha=0.08)
                single_ax.plot(errors['ts_ensemble'][i,:,0],errors['ts_ensemble'][i,:,1],c=ts_ensemble_color,alpha=0.08)
            single_ax.plot(np.average(errors['gt'][:, :, 0], axis=0), 
                           np.average(errors['gt'][:, :, 1], axis=0), 
                           c=gt_color, linestyle=gt_style, marker=gt_marker, alpha=1)
            single_ax.plot(np.average(errors['dr_ensemble'][:, :, 0], axis=0), 
                           np.average(errors['dr_ensemble'][:, :, 1], axis=0), 
                           c=dr_ensemble_color, linestyle=dr_ensemble_style, marker=dr_ensemble_marker, alpha=1)
            single_ax.plot(np.average(errors['ts_ensemble'][:, :, 0], axis=0), 
                           np.average(errors['ts_ensemble'][:, :, 1], axis=0), 
                           c=ts_ensemble_color, linestyle=ts_ensemble_style, marker=ts_ensemble_marker, alpha=1)
            single_ax.set_xlabel(xlabels[pltindex])
            single_ax.spines[['right', 'top']].set_visible(False)
            single_ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            single_ax.set_title(ax[pltindex].get_title())
            if not (xmin is None or xmax is None or ymax is None):
                single_ax.set_xlim(left=xmin[pltindex], right=xmax[pltindex])
                single_ax.set_ylim(top=ymax[pltindex])
            if pltindex == 0:
                single_ax.set_ylabel('demographic parity on test data')
            if pltindex == 3:
                single_ax.legend([line1, line2, line3],
                                ['Pareto front \n of test data', 
                                'directly regularized \n ensemble', 
                                'two-stage estimator \n ensemble'],
                                loc=(1, 0.3))
            single_fig.savefig(f"{title}_{key}.png", dpi=500, bbox_inches='tight')
            plt.close(single_fig)
    
    ax[pltindex].legend(
        [line1,line2,line3],
        ['Pareto front \n of test data', 'directly regularized \n ensemble','two-stage estimator \n ensemble'],
        loc=(1,0.3)
        )
    
    #plt.tight_layout()

    #ax.scatter(control[0],control[1])
    if save:
        plt.savefig(title+'.png',dpi=500,bbox_inches='tight')
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument("--plot", choices=["sparsity-unlabeled", "pareto-fronts", "fairness"], help="Choose the plot to generate.")
    parser.add_argument("--errors", type=str, required=True, help="Path to the errors file to load.")
    args = parser.parse_args()

    # Load errors from the specified file
    with open(args.errors, "rb") as f:
        errors = pickle.load(f)

    if args.plot == "sparsity-unlabeled":
        plot_sparsity_unlabeled_matrix(
            errors=errors,
            N_array=np.arange(15, 55, 5),
            s_array=np.arange(5, 50, 5),
            save=False
        )

    elif args.plot == "pareto-fronts":
        plot_Pareto_fronts(
            errors=errors,
            title=' ',
            save=False
        )

    elif args.plot == "fairness":
        plot_fairness_Pareto_fronts(
            errordict=errors,
            title='fairness',
            xlabels=['square loss on test data', 'error rate on test data', 'error rate on test data', 'error rate on test data', 'error rate on test data'],
            save=False,
            xmax=[0.4, 0.23, 0.34, 0.45],
            xmin=[0, 0.2, 0.25, 0.28],
            ymax=[0.2, 0.013, 0.0025, 0.01]
        )

if __name__ == "__main__":
    main()