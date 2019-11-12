import sed_eval

def score_model(test_idx, test_features, model, labels):
        
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=labels)
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=labels)

    for filename in test_idx:
        test_feature_loc = os.path.join(test_features, filename + '.h5')
        test_feature = load_h5(test_feature_loc)
        datum = test_feature['PCEN/mag']
        ytrue = max_pool(test_feature['dynamic/tags'][0])[0]
        ypred = model.predict(datum)[0]
        
        ytrue_dict = convert_ts_to_dict(ytrue, labels, filename)
        ypred_dict = convert_ts_to_dict(ypred, labels, filename, threshold=0.5)

        segment_based_metrics.evaluate(reference_event_list=ytrue_dict,\
                                       estimated_event_list=ypred_dict)

        event_based_metrics.evaluate(reference_event_list=ytrue_dict,\
                                     estimated_event_list=ypred_dict)
        


    # Get only certain metrics
    overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
    print("Accuracy:", overall_segment_based_metrics['accuracy']['accuracy'])

    # Or print all metrics as reports
    print(segment_based_metrics)
    print(event_based_metrics)

def convert_ts_to_dict(predictions, labels, fname, threshold=None, real_length = 10.):
    predictions = predictions.T
    out_dicts = []
    sr = real_length/predictions.shape[1]
    
    for i, label in enumerate(labels):
        if threshold is not None:
            high_low_array = (predictions[i]>threshold).astype(int)
        else:
            high_low_array = predictions[i]
            
        label_data = np.concatenate((np.zeros(1), high_low_array, np.zeros(1)))
        onsets = np.argwhere(np.diff(label_data)==1) -1
        offsets = np.argwhere(np.diff(label_data)==-1) -1

        
        for i in range(len(onsets)):
            new_dict = {}
            new_dict['event_label']=label
            new_dict['event_onset']=onsets[i][0]*sr
            new_dict['event_offset']=offsets[i][0]*sr
            new_dict['file']=fname
            new_dict['scene_label']= 'UrbanSED'
            out_dicts.append(new_dict)
    return out_dicts
    
if __name__=='__main__':
    ### Evaluate model
    print('Evaluate model...')
    # Load best params
    model.load_weights(weight_path)

    with open(os.path.join(DATA_LOC, 'index_test.json'), 'r') as fp:
        test_idx = json.load(fp)['id']

    # Compute eval scores
    results = score_model(test_idx, TEST_FEATURES_LOC, model, URBANSED_CLASSES)

    # Save results to disk
    results_file = os.path.join(MODEL_LOC, modelid, 'results.json')
    with open(results_file, 'w') as fp:
        json.dump(results, fp, indent=2)

    print('Done!')