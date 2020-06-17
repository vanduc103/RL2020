#!/usr/bin/env python
import os
import sys
import glob
import traceback
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from autosklearn import classification
from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels
from dataset import load_data

''' Env '''
class Env():
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.X_all = []
        self.y_all = []
        self.X_train_all = []
        self.y_train_all = []
        self.X_test_all = []
        self.y_test_all = []
        self.feat_type_all = []
        self.dataset_name_all = []
        self.size = 0

        self.scaling_list = "minmax, none, normalize, quantile_transformer, robust_scaler, standardize".replace(" ","").split(",")
        self.preprocessor_list = "extra_trees_preproc_for_classification, fast_ica, feature_agglomeration, kernel_pca, kitchen_sinks, liblinear_svc_preprocessor, no_preprocessing,nystroem_sampler, pca, polynomial, random_trees_embedding, select_percentile_classification, select_rates".replace(" ","").split(",")
        self.classifier_list = "adaboost, bernoulli_nb, decision_tree, extra_trees, gaussian_nb, gradient_boosting, k_nearest_neighbors, lda, liblinear_svc, libsvm_svc,multinomial_nb, passive_aggressive, qda, random_forest, sgd".replace(" ","").split(",")
        self.action_vocab = {}

        self.supermeta = {'ClassEntropy': [0.03853718088787195, 5.634725237579686, 1.2833706866488395],
                         'SymbolsSum': [0.0, 287, 7.169642857142857],
                         'SymbolsSTD': [0, 3.0952509611333126, 0.11628641612175684],
                         'SymbolsMean': [0, 6.666666666666667, 0.46413053712160857],
                         'SymbolsMax': [0, 12, 0.7053571428571429],
                         'SymbolsMin': [0, 4, 0.32142857142857145],
                         'ClassProbabilitySTD': [0.0, 0.49588477366255146, 0.1477539836312912],
                         'ClassProbabilityMean': [0.02, 0.5, 0.40916001657073103],
                         'ClassProbabilityMax': [0.025714285714285714, 0.9958847736625515, 0.5713088860091884],
                         'ClassProbabilityMin': [0.00013793103448275863, 0.5, 0.26446024662686446],
                         'InverseDatasetRatio': [0.105, 4833.333333333333, 549.0306747222126],
                         'DatasetRatio': [0.00020689655172413793, 9.523809523809524, 0.6832655137201668],
                         'RatioNominalToNumerical': [0.0, 3.5, 0.07844387755102042],
                         'RatioNumericalToNominal': [0.0, 7.0, 0.10442830978545266],
                         'NumberOfCategoricalFeatures': [0, 60, 1.9464285714285714],
                         'NumberOfNumericFeatures': [0, 10935, 1091.875],
                         'NumberOfMissingValues': [0.0, 0.0, 0.0],
                         'NumberOfFeaturesWithMissingValues': [0.0, 0.0, 0.0],
                         'NumberOfInstancesWithMissingValues': [0.0, 0.0, 0.0],
                         'NumberOfFeatures': [1.0, 10935.0, 1093.8214285714287],
                         'NumberOfClasses': [2.0, 50.0, 4.035714285714286],
                         'NumberOfInstances': [700.0, 70000.0, 7262.517857142857],
                         'LogInverseDatasetRatio': [-2.2537949288246137, 8.483291639740557, 4.4724832256630895],
                         'LogDatasetRatio': [-8.483291639740557, 2.2537949288246137, -4.4724832256630895],
                         'PercentageOfMissingValues': [0.0, 0.0, 0.0],
                         'PercentageOfFeaturesWithMissingValues': [0.0, 0.0, 0.0],
                         'PercentageOfInstancesWithMissingValues': [0.0, 0.0, 0.0],
                         'LogNumberOfFeatures': [0.0, 9.299723933110869, 3.6941254838551267],
                         'LogNumberOfInstances': [6.551080335043404, 11.156250521031495, 8.16660870951822]}

        vocab_list = []
        vocab_list.append("<START>")
        vocab_list.append(self.scaling_list)
        vocab_list.append(self.preprocessor_list)
        vocab_list.append(self.classifier_list)
        vocab_list.append("<END>")
        for i in range(len(vocab_list)):
            self.action_vocab[str(vocab_list[i])] = i

        # load training data
        default_dataset_path = 'dataset'
        files = []
        if self.dataset_path != None:
            files = glob.glob(os.path.join(self.dataset_path, "*.csv"))
        else:
            files = glob.glob(os.path.join(default_dataset_path, "*.csv"))
        for filepath in files:    
            dataset_id = os.path.basename(filepath)
            dataset_id = dataset_id.replace("dataset","").replace(".csv", "").replace("_","")
            print('Load data from dataset ' + dataset_id)
            self.dataset_name_all.append(dataset_id)

            X, y, feat_type = load_data(filepath, dataset_id)
            self.X_all.append(X)
            self.y_all.append(y)
            self.feat_type_all.append(feat_type)
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=1)
            self.X_train_all.append(X_train)
            self.X_test_all.append(X_test)
            self.y_train_all.append(y_train)
            self.y_test_all.append(y_test)
        self.size = len(self.X_all)

    def run_actions(self, actions, batch_idx=0, batch_size=1):
        (rescaling_action, preprocessor_action, classifier_action) = actions
        rewards = []
        done = []
        action_idx = 0
        for i in range(batch_size*batch_idx, min(self.size, batch_size*(batch_idx+1))):
            # run machine learning pipeline
            X_train = self.X_train_all[i]
            y_train = self.y_train_all[i]
            X_test = self.X_test_all[i]
            y_test = self.y_test_all[i]
            feat_type = self.feat_type_all[i]

            rescaling = self.scaling_list[rescaling_action[action_idx]]
            preprocessor = self.preprocessor_list[preprocessor_action[action_idx]]
            classifier = self.classifier_list[classifier_action[action_idx]]
            action_idx += 1
            print(rescaling)
            print(preprocessor)
            print(classifier)
            # run autosklearn hyperparameter optimization step
            automl = classification.AutoSklearnClassifier(
                        time_left_for_this_task=30,
                        per_run_time_limit=10,
                        initial_configurations_via_metalearning=0,
                        include_datapreprocessors=[rescaling],
                        include_preprocessors=[preprocessor],
                        include_estimators=[classifier],
                        ensemble_size=1)
            try:
                automl.fit(X_train, y_train, feat_type=feat_type)
                accuracy = metrics.accuracy_score(y_test, automl.predict(X_test))
            except Exception as e:
                #traceback.print_exc(file=sys.stdout)
                accuracy = 0

            # return
            rewards.append([accuracy])
            done.append([True])
        states = self.data_state(batch_idx, batch_size)
        return states, np.asarray(rewards), np.asarray(done), actions


    # get data state information
    def data_state(self, batch_idx=0, batch_size=1):
        data_states = []
        for i in range(batch_size*batch_idx, min(self.size, batch_size*(batch_idx+1))):
            categorical = None
            feat_type = self.feat_type_all[i]
            if feat_type != None:
                categorical = [True if data_feat_type.lower() in ['categorical'] else False
                                for data_feat_type in feat_type]
            X, y = self.X_all[i], self.y_all[i]
            dataset_name = self.dataset_name_all[i]
            result = calculate_all_metafeatures_with_labels(
                X, y, categorical=categorical, dataset_name=dataset_name)
            states = []
            meta = {}
            for key in list(result.metafeature_values.keys()):
                if result.metafeature_values[key].type_ == 'METAFEATURE':
                    meta[key] = result.metafeature_values[key].value
            for key in meta:
                value = meta[key]
                minvalue = self.supermeta[key][0]
                maxvalue = self.supermeta[key][1]
                if maxvalue == 0 and minvalue == 0:
                    value = 0
                else:
                    value = (value - minvalue)/(maxvalue-minvalue)
                states.append(value)
            #print(states)
            data_states.append(np.asarray(states))
        return data_states

if __name__ == "__main__":
    main()
