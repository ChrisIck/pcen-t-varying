import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seaborn import pointplot
from tqdm.autonotebook import tqdm


import sys
sys.path.append('/home/ci411/pcen-t-varying/')

from pcen_t.utils import *
from pcen_t.models import MODELS
#import eval_milsed

MODEL_PATH = '/beegfs/ci411/pcen/models/models_122'

URBANSED_CLASSES = ['air_conditioner',
                    'car_horn',
                    'children_playing',
                    'dog_bark',
                    'drilling',
                    'engine_idling',
                    'gun_shot',
                    'jackhammer',
                    'siren',
                    'street_music']

COLORMAP = 'viridis'

metric_dict = {'f1':'F1-Score', 'p':'Precision', 'r':'Recall', 'e':'Error Rate', 'overall':'Overall'}


from collections import OrderedDict
from keras.models import model_from_yaml

from pcen_t.utils import * 
from pcen_t.models import MODELS, SqueezeLayer
from pcen_t import pcen_pump
sys.modules['pcen_pump'] = pcen_pump

def collect_models(model_path):
    models = os.listdir(model_path)
    results = {}
    missing_models = []
    for model in models:
        path = os.path.join(model_path, model)
        if os.path.isdir(path):
            try:
                with open(os.path.join(path, 'results.json'), 'r') as fd:
                    result = json.load(fd)
                results[model] = result
            except FileNotFoundError:
                print("No results.json file for model '{}'".format(model))
                missing_models.append(model)
    return list(results.keys()), results, missing_models

def report_results(OUTPUT_PATH, version, results_name='results'):
    # Load results
    resultsfolder = os.path.join(OUTPUT_PATH, version)
    resultsfile = os.path.join(resultsfolder, results_name + '.json')
    with open(resultsfile, 'r') as fp:
        results = json.load(fp)

    # report
    print('{:<10}{}'.format('Model', version))

    print('\nStrong:')
    strong_f = results['overall']['f_measure']
    strong_e = results['overall']['error_rate']
    print('{:<10}{:.3f}'.format('precision', strong_f['precision']))
    print('{:<10}{:.3f}'.format('recall', strong_f['recall']))
    print('{:<10}{:.3f}'.format('f1', strong_f['f_measure']))
    print('{:<10}{:.3f}'.format('e_rate', strong_e['error_rate']))

    print('\n{:<40}P\tR\tF\tE'.format('Strong per-class:'))
    strong_c = results['class_wise']
    c_sorted = [c for c in strong_c.keys()]
    c_sorted = sorted(c_sorted)
    for c in c_sorted:
        r_c = strong_c[c]['f_measure']
        r_ce = strong_c[c]['error_rate']
        print('{:<40}{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(c, r_c['precision'],
                                                            r_c['recall'],
                                                            r_c['f_measure'],
                                                            r_ce['error_rate']))

    # # Load training history
    # history_file = os.path.join(resultsfolder, 'history.pkl')
    # with open(history_file, 'rb') as fp:
    #     history = pickle.load(fp)

    # Load dynamic history CSV file
    csvfile = os.path.join(resultsfolder, 'history_csvlog.csv')
    history = pd.read_csv(csvfile)

    # Set sns style
    #sns.set()

    print('\nLoss:')

    # Visualize training history
    plt.figure(figsize=(9,6))
    plt.subplot(2,1,1)
    plt.plot(history['loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    #plt.axvline(np.argmin(history['val_loss']), color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss: {}'.format(version))
    # plt.grid()
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(history['accuracy'], label='training accuracy')
    plt.plot(history['val_accuracy'], label='validation accuracy')
    #plt.axvline(np.argmax(history['accuracy']), color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy: {}'.format(version))
    plt.legend()
    plt.tight_layout()

    plt.show()
    
    
def extract_slices(models):
    return [[int(single_slice) for single_slice in slices.split('_')[2]] for slices in models]

def results_to_df(OUTPUT_PATH, versions, sort=False,
                    is_ensemble=False, results_name='results'):
    results = OrderedDict({})
    params = OrderedDict({})
    n_weights = OrderedDict({})

    # Load results
    for version in tqdm(versions):

        # Load overall results
        resultsfile = os.path.join(OUTPUT_PATH, version, results_name + '.json')
        with open(resultsfile, 'r') as fp:
            results[version] = json.load(fp)

        if is_ensemble:
            n_weights[version] = 'ensemble'
            params[version] = {'modelname': version}
        else:
            modelfile = os.path.join(OUTPUT_PATH, version, 'model.yaml')
            with open(modelfile, 'r') as yaml_file:
                params[version] = yaml_file.read()
            
            # Compute model size
            #HACK ALERT
            model = model_from_yaml(params[version], custom_objects={'SqueezeLayer':SqueezeLayer})
            n_weights[version] = model.count_params()
        
    # Convert to dataframe
    classes = results[list(results.keys())[0]]['class_wise'].keys()
    classes = sorted(classes)
    
    metrics =  ['precision', 'recall', 'f_measure', 'error_rate']
    metric_map = {'precision':'_p', 'recall':'_r', 'f_measure':'_f1', 'error_rate':'_e'}
    
    columns = ['version', 'model', 'n_weights',  'overall_p', 'overall_r', 'overall_f1', 'overall_e']
    for c in classes:
        for m in metrics:
            columns.append(c+metric_map[m])
    
    df = pd.DataFrame(columns=columns, dtype=float)
    
    for k in results.keys():
        r = results[k]
        strong_f = r['overall']['f_measure']
        strong_e = r['overall']['error_rate']
        strong_c = r['class_wise']
        data = [
            k, 'cnn_l3_strong', n_weights[k],
            strong_f['precision'], strong_f['recall'],
            strong_f['f_measure'], strong_e['error_rate']]
        for c in classes:
            for m in metrics:
                if m != 'error_rate':
                    data.append(strong_c[c]['f_measure'][m])
                else:
                    data.append(strong_c[c]['error_rate'][m])
        df.loc[len(df), :] = tuple(data)

    columns.insert(1, 'training_set')
    columns.insert(2, 'slices')
    df['training_set'] = df['version'].map(lambda x : x.split('_')[2])
    df['slices'] = df['version'].map(lambda x : set(x.split('_')[1]))
    df = df[columns]
        
    if sort:
        df = df.sort_values('version')
    return df

def compare_results_df(df, models, metric='overall', classes=URBANSED_CLASSES, dataset_name='Urbansed_Unlabeled', ymax=1, ax=None):
    n_models = len(models)
    data = []
    cmap = plt.get_cmap(COLORMAP)
    
    if metric=='overall':
        metrics_short = ['f1', 'p', 'r', 'e']
        columns = ['overall_'+m for m in metrics_short]
        labels = [metric_dict[l] for l in metrics_short]
        n_bars = 4
    else:
        columns = [c+'_'+metric for c in classes]
        n_bars = len(classes)
        labels = classes
    x = np.arange(n_bars)

    plt.figure()
    
    if ax==None:
        subplot = False
        ax = plt.gca()
    else:
        subplot = True
    
    
    full_width = .95 
    width = full_width/(n_models+1)
    for i, model in enumerate(models):
        model_idx = df['version']==model
        if (len(df[model_idx][columns].values.tolist())!=0):
            results = df[model_idx][columns].values.tolist()[0]
            ax.bar(x-(full_width/2.)+(width*i), results, width, label=model, color=cmap(i/n_models))

        else:
            print("No model {}".format(model))
            pass
    
    if subplot==True:
        plt.close()
        return ax
    
    else:
        plt.title("{} Comparison Results on {}".format(metric_dict[metric], dataset_name))
        plt.xticks(x, labels)
        plt.ylim((0,ymax))
        plt.legend()
        
def difference_plot(datasets, modelsets, dataset_names, metric='overall', ax=None, title='Results', classes=URBANSED_CLASSES):
    n_models = len(modelsets[0])
    data = []
    cmap = plt.get_cmap(COLORMAP)
    
    if metric=='overall':
        metrics_short = ['f1', 'p', 'r', 'e']
        columns = ['overall_'+m for m in metrics_short]
        labels = [metric_dict[l] for l in metrics_short]
        n_bars = 4
    else:
        columns = [c+'_'+metric for c in classes]
        n_bars = len(classes)
        labels = classes
    y = np.arange(n_bars)[::-1]

    
    
    if ax==None:
        plt.figure(figsize=(10,10))
        subplot = False
        ax = plt.gca()
        #ax_label = ax.twinx()
    else:
        subplot = True
        #ax_label = ax.twinx()
    
    
    full_height = .95 
    height = full_height/(n_models+1)
    
    model_yticks = []
    model_labels = []
    for i in range(n_models)[::-1]:
        model0_idx = datasets[0]['version']==modelsets[0][i]
        model1_idx = datasets[1]['version']==modelsets[1][i]
        
        results0 = np.array(datasets[0][model0_idx][columns].values.tolist()[0])
        results1 = np.array(datasets[1][model1_idx][columns].values.tolist()[0])
        
        model_label = "{} on {} / {} on {}".format(modelsets[0][i],dataset_names[0],modelsets[1][i],dataset_names[1])
        model_labels.append(model_label)
        
        model_ytick = y+(height*i)
        model_yticks.append(model_ytick)
        
        results = results1 - results0
        ax.barh(model_ytick, results, height, color=cmap(i/n_models), label = model_label)

    model_yticks = np.array(model_yticks).flatten()
    model_yticks.sort()
    
    model_labels *= n_bars

    if subplot==True:
        plt.close()
        return ax, ax_label
    
    else:
        plt.title(title)
        ax.set_yticks(y+height)
        ax.set_yticklabels(labels)
        
        '''ax_label.tick_params(axis='y', right=True)
        ax_label.set_yticks(model_yticks)
        ax_label.set_yticklabels(model_labels)'''
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.8,
                 box.width, box.height * 0.8])
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True)
        

        
def compare_model_sets(datasets, modelsets, dataset_names, ymax, metric='overall', compare_plot=False, classes=URBANSED_CLASSES):
    
    N = len(datasets)
    if compare_plot:
        N+=1

    fig, axs = plt.subplots(ncols=N, sharey=True, figsize=(20,8))
    plt.tight_layout()
    plt.ylim((0,ymax))
    
    if metric=='overall':
        metrics_short = ['f1', 'p', 'r', 'e']
        columns = ['overall_'+m for m in metrics_short]
        labels = [metric_dict[l] for l in metrics_short]
        n_bars = 4
    else:
        columns = [c+'_'+metric for c in classes]
        n_bars = len(classes)
        labels = classes
        
    x = np.arange(n_bars)
    
    for i, dataset in enumerate(datasets):
        compare_results_df(dataset, modelsets[i], ax=axs[i], dataset_name=dataset_names[i], metric=metric, ymax=ymax)
        
        axs[i].set_title("{} results for {}".format(metric_dict[metric], dataset_names[i]))
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels)
        axs[i].legend()
                
    return fig, axs
    

def point_comparison_plot(dataset, modelsets, modelset_names, metric='f1', class_label='overall', ax=None, title='Results', cm=None, classes=URBANSED_CLASSES):
    n_models = len(modelsets[0])
    data = []
    
    if cm is None:
        cmap = plt.get_cmap(COLORMAP)
    else:
        cmap = plt.get_cmap(cm)
    
    metric_label = metric_dict[metric]
    
  
    if ax==None:
        plt.figure(figsize=(10,10))
        subplot = False
        ax = plt.gca()
    else:
        subplot = True
    
    x = np.arange(n_models)
    column = class_label+'_'+metric
    
    for i, models in enumerate(modelsets):
        modelset_results = []
        for model in models:
            model_idx = dataset['version']==model
            
            results = np.array(dataset[model_idx][column].values.tolist()[0])
                
            modelset_results.append(results)
        
        pointplot(x=x, y=modelset_results, ax=ax, color=cmap(i/n_models), label=modelset_names[i])

    if subplot==True:
        plt.close()
        return ax, ax_label
    
    else:
        plt.title(title)
        
        plt.legend()
        
        
def samples_to_df(OUTPUT_PATH, versions, sort=False, classes=URBANSED_CLASSES, melt=True):
    
    classes = sorted(classes)
    
    metrics =  ['precision', 'recall', 'f_measure', 'error_rate']
    metric_map = {'precision':'_p', 'recall':'_r', 'f_measure':'_f1', 'error_rate':'_e'}
    
    columns = ['version', 'eval_dataset', 'model', 'overall_p', 'overall_r', 'overall_f1', 'overall_e']
    for c in classes:
        for m in metrics:
            columns.append(c+metric_map[m])
    
    df = pd.DataFrame(columns=columns, dtype=float)
    
    # Load results
    print('Loading results...')
    for version in tqdm(versions):

        # Load overall results
        results_path = os.path.join(OUTPUT_PATH, version, 'sampled_results')
        if not os.path.isdir(results_path):
            print('Missing sampled results at {}'.format(results_path))
            continue
        results_samples = [sample for sample in os.listdir(results_path) if '.json' in sample]
        for sample in tqdm(results_samples, desc='Read: {}'.format(version), leave=False):
            results_file = os.path.join(results_path, sample)
            sample_name = base(sample)
            
            try:
                _, eval_dataset, sample_number = sample_name.split('_')
                version_number = '{}_{}_{}'.format(version, eval_dataset, sample_number)
            except:
                print("Bad name {} at {}".format(sample_name, results_path))
                eval_dataset = sample_name
                sample_number = 404

            #modelfile = os.path.join(OUTPUT_PATH, version, 'model.yaml')
            #with open(modelfile, 'r') as yaml_file:
            #    params = yaml_file.read()

            # Compute model size
            #HACK ALERT
            #model = model_from_yaml(params, custom_objects={'SqueezeLayer':SqueezeLayer})
            #n_weights = model.count_params()
            
            with open(results_file, 'r') as fp:
                results = json.load(fp)
            
            strong_f = results['class_wise_average']['f_measure'] #macro-averaged
            strong_e = results['class_wise_average']['error_rate'] #macro-averaged
            #strong_f = results['overall']['f_measure'] #micro-averaged
            #strong_e = results['overall']['error_rate'] #micro-averaged
            strong_c = results['class_wise'] 
                
            data = [
                version, eval_dataset, 'cnn_l3_strong',
                strong_f['precision'], strong_f['recall'],
                strong_f['f_measure'], strong_e['error_rate']]
            for c in classes:
                for m in metrics:
                    if m != 'error_rate':
                        data.append(strong_c[c]['f_measure'][m])
                    else:
                        data.append(strong_c[c]['error_rate'][m])
            df.loc[len(df), :] = tuple(data)
            
            del results
            
            #del model
            #del params

    columns.insert(1, 'training_set')
    columns.insert(2, 'slices')
    df['training_set'] = df['version'].map(lambda x : x.split('_')[0])
    df['slices'] = df['version'].map(lambda x : str(x.split('_')[1]))
    df = df[columns]
        
    if sort:
        df = df.sort_values('version')
    if melt:
        print('Melting dataframe...')
        return melt_sample_df(df)
    else:
        return df


default_ids = ['version','training_set','slices','eval_dataset','model']
def melt_sample_df(df, id_labels=default_ids):
    metric_labels = ['_p', '_r', '_f1', '_e']
    metrics =  ['precision', 'recall', 'f_measure', 'error_rate']
    metric_label_map = {}
    for i in range(4):
        metric_label_map[metric_labels[i].replace('_','')] = metrics[i]
    
    label_columns = [column for column in df.columns if any(metric in column for metric in metric_labels)]
    melted_df = df.melt(id_vars=id_labels, value_vars=label_columns, var_name='class/metric')
    melted_df['metric'] = melted_df['class/metric'].apply(lambda x:metric_label_map[x.split('_')[-1]])
    melted_df['class'] = melted_df['class/metric'].apply(lambda x:'_'.join(x.split('_')[:-1]))
    melted_df.drop(columns=['class/metric'])
    return melted_df

