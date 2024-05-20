import os
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from core.rawdata_process import CreateDataset
from utils.customclass import NormalizeActivation


class ClassificationModel:
    """
    Build classification model
    """
    def __init__(self):
        super().__init__()
        self.cl_model = None
        self.val_accuracy = None
        self.cl_best_params = None
        self.cl_inputs = None
        self.cl_outputs = None

        self.cl_x_train = None
        self.cl_x_test = None
        self.cl_y_train = None
        self.cl_y_test = None

    @staticmethod
    def calc_accuracy(model, x_test, y_test):
        """
        Calculate the accuracy of the model.

        """
        # Predictions
        predictions = model.predict(x_test)
        threshold = 0.5
        predicted_labels = (predictions > threshold).astype(int)

        error = predicted_labels - y_test
        error = abs(error)
        err_num = 0
        for i in range(error.shape[0]):
            if sum(error[i, :]) > 0:
                err_num += 1

        accuracy = 1 - err_num / y_test.shape[0]

        return accuracy

    def cl_prepare_for_training(self, inputs, outputs):
        """
        Dataset split for training the classification model.
        """
        self.cl_inputs = inputs
        self.cl_outputs = outputs
        self.cl_x_train, self.cl_x_test, self.cl_y_train, self.cl_y_test = train_test_split(inputs, outputs,
                                                                                            test_size=0.2,
                                                                                            random_state=42)

    def cl_generate_model(self, first_layer_neuron=64, second_layer_neuron=128, third_layer_neuron=64):
        """
        Generate the classification model.

        Notes
        -----
        Each time using this function, the model will be re-initialized and stored in self.cl_model.

        """
        num_features = self.cl_inputs.shape[1]  # Number of input features
        num_classes = self.cl_outputs.shape[1]  # Number of labels

        # Create the model
        model = Sequential()
        model.add(Dense(first_layer_neuron, activation='relu', input_shape=(num_features,)))
        # model.add(BatchNormalization())
        model.add(Dense(second_layer_neuron, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dense(third_layer_neuron, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))  # Sigmoid activation for multi-label

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.cl_model = model

    def cl_model_training(self, epoch=3000, batch_size=32000):
        """
        Raise model training.

        Parameters
        ----------
        epoch : int
            Number of epochs.
        batch_size : int
            Batch size.

        Returns
        -------
        float
            Validation_accuracy.

        """
        his = self.cl_model.fit(self.cl_x_train, self.cl_y_train, epochs=epoch, batch_size=batch_size,
                                validation_data=(self.cl_x_test, self.cl_y_test))

        test_loss, test_accuracy = self.cl_model.evaluate(self.cl_x_test, self.cl_y_test)

        validation_accuracy = self.calc_accuracy(self.cl_model, self.cl_x_test, self.cl_y_test)  # Calculate accuracy

        self.val_accuracy = validation_accuracy

        return validation_accuracy

    def cl_train_model(self, bayesian_opt=False, **kwargs):
        """
        Train the classification model.

        Parameters
        ----------
        bayesian_opt : bool
            Whether to use Bayesian optimization.
        **kwargs
            first_layer : int or list
                Number of neurons in the first layer or range of neurons.
            second_layer : int or list
                Number of neurons in the second layer or range of neurons.
            third_layer : int or list
                Number of neurons in the third layer or range of neurons.
            epoch : int or list
                Number of epochs or range of epochs.
            max_evals : int
                Number of evaluations in Bayesian optimization.

        """
        if bayesian_opt:
            first_layer = kwargs.get('first_layer', [20, 400, 20])
            second_layer = kwargs.get('second_layer', [20, 500, 20])
            third_layer = kwargs.get('third_layer', [20, 400, 20])
            epoch = kwargs.get('epoch', [100, 2000, 100])
            max_evals = kwargs.get('max_evals', 20)
            self.cl_bayesian_opt(first_layer, second_layer, third_layer, epoch, max_evals)

        # Train the model with fixed parameters
        else:
            first_layer = kwargs.get('first_layer', 64)
            second_layer = kwargs.get('second_layer', 128)
            third_layer = kwargs.get('third_layer', 64)
            epoch = kwargs.get('epoch', 100)
            batch_size = kwargs.get('batch_size', 32000)

            self.cl_generate_model(first_layer, second_layer, third_layer)
            self.cl_model_training(epoch, batch_size)

    def cl_bayesian_opt(self, first_layer=None, second_layer=None, third_layer=None, epoch=None, max_evals=20):
        """
        Bayesian optimization for hyperparameters optimization.

        Parameters
        ----------
        first_layer : list
            Range of neurons in the first layer.
        second_layer : list
            Range of neurons in the second layer.
        third_layer : list
            Range of neurons in the third layer.
        epoch : list
            Range of epochs.
        max_evals : int
            Number of evaluations.

        Notes
        -----
        Higher max_evals will lead to longer execution time and better optimization.

        """

        # Objective function for Bayesian optimization
        def objective(params):

            self.cl_generate_model(int(params['first_layer']), int(params['second_layer']), int(params['third_layer']))

            val_acc = self.cl_model_training(int(params['epoch']))

            print('Test mse:', val_acc)

            # Minimize the negative accuracy
            return {'loss': -val_acc, 'status': STATUS_OK}

        if first_layer is None:
            first_layer = [20, 400, 20]
        if second_layer is None:
            second_layer = [20, 500, 20]
        if third_layer is None:
            third_layer = [20, 400, 20]
        if epoch is None:
            epoch = [100, 2000, 100]

        space = {
            'first_layer': hp.quniform('first_layer', first_layer[0], first_layer[1], first_layer[2]),
            'second_layer': hp.quniform('second_layer', second_layer[0], second_layer[1], second_layer[2]),
            'third_layer': hp.quniform('third_layer', third_layer[0], third_layer[1], third_layer[2]),
            'epoch': hp.quniform('epoch', epoch[0], epoch[1], epoch[2])
        }

        trials = Trials()

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        print(best)

        model_temp = []
        loss_temp = []

        # Get the best value from three results.
        # The goal of this is to avoid the possible partial optimization of neural network.
        for i in range(3):
            self.cl_generate_model(int(best['first_layer']), int(best['second_layer']), int(best['third_layer']))
            val_accuracy = self.cl_model_training(int(best['epoch']))

            model_temp.append(self.cl_model)
            loss_temp.append(val_accuracy)

        self.cl_model = model_temp[loss_temp.index(max(loss_temp))]
        self.val_accuracy = self.calc_accuracy(self.cl_model, self.cl_x_test, self.cl_y_test)
        self.cl_best_params = best

    def cl_save_model(self, path):
        """
        Save the classification model.

        Parameters
        ----------
        path : str
            Path to save the model.

        Notes
        -----
        The model will be saved in the path with the name 'classification model'.

        """
        if not os.path.exists(path):
            os.makedirs(path)

        self.cl_model.save(os.path.join(path, 'classification model'), save_format='tf')

        model_infor = {'val_accuracy': self.val_accuracy}
        if self.cl_best_params is not None:
            model_infor['best_params'] = self.cl_best_params

        with open(os.path.join(path, 'model infor.json'), 'w') as f:
            json.dump(model_infor, f)


class CompositionModels:
    """
    Build composition models.

    """
    def __init__(self):
        """
        Initialize the class.

        Notes
        -----
        A system contains multiple composition models. Each model is trained for a specific phase.

        """
        super().__init__()
        self.cp_models = []  # Composition models
        self.val_mse = []  # Validation mean squared error
        self.cp_best_params = []
        self.phase_num = None
        self.input_dim = None
        self.output_dim = None

        self.cp_inputs = []
        self.cp_outputs = []

        self.cp_x_train = []
        self.cp_x_test = []
        self.cp_y_train = []
        self.cp_y_test = []

    def cp_prepare_for_training(self, discrete_dataset):
        """
        Dataset split for training the composition models.

        Parameters
        ----------
        discrete_dataset : list
            Discrete datasets list.

        """
        self.phase_num = len(discrete_dataset)
        self.input_dim = discrete_dataset[0][0].shape[1]
        self.output_dim = discrete_dataset[0][1].shape[1]

        for each_dataset in discrete_dataset:
            inputs = each_dataset[0]
            outputs = each_dataset[1]

            self.cp_inputs.append(inputs)
            self.cp_outputs.append(outputs)

            x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

            self.cp_x_train.append(x_train)
            self.cp_x_test.append(x_test)
            self.cp_y_train.append(y_train)
            self.cp_y_test.append(y_test)

    @staticmethod
    def cp_generate_model(input_dim, output_dim, first_layer_neuron=64, second_layer_neuron=128, third_layer_neuron=64):
        # Create the model
        model = Sequential()
        model.add(Dense(first_layer_neuron, activation='relu', input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Dense(second_layer_neuron, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dense(third_layer_neuron, activation='relu'))

        model.add(Dense(output_dim, activation='relu'))
        model.add(NormalizeActivation())

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    @staticmethod
    def cp_model_training(model, x_train, y_train, x_test, y_test, epoch=3000, batch_size=32000):
        his = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_test, y_test))

        test_loss = model.evaluate(x_test, y_test)

        return test_loss

    def cp_train_model(self, bayesian_opt=False, **kwargs):
        """
        Train the composition models.

        Parameters
        ----------
        bayesian_opt : bool
            Whether to use Bayesian optimization.
        **kwargs
            first_layer : int or list
                Number of neurons in the first layer or range of neurons.
            second_layer : int or list
                Number of neurons in the second layer or range of neurons.
            third_layer : int or list
                Number of neurons in the third layer or range of neurons.
            epoch : int or list
                Number of epochs or range of epochs.
            max_evals : int
                Number of evaluations in Bayesian optimization.

        Notes
        -----
        Higher max_evals will lead to longer execution time and better optimization.

        """
        if bayesian_opt:
            first_layer = kwargs.get('first_layer', [20, 400, 20])
            second_layer = kwargs.get('second_layer', [20, 500, 20])
            third_layer = kwargs.get('third_layer', [20, 400, 20])
            epoch = kwargs.get('epoch', [100, 2500, 200])
            max_evals = kwargs.get('max_evals', 30)

            # The strategy is different from classification model training. The composition models are trained one by
            # one.
            for i in range(self.phase_num):
                model, val_mse, best = self.cp_bayesian_opt(self.cp_x_train[i], self.cp_y_train[i], self.cp_x_test[i],
                                                            self.cp_y_test[i], first_layer, second_layer, third_layer,
                                                            epoch, max_evals)
                self.cp_models.append(model)
                self.val_mse.append(val_mse)
                self.cp_best_params.append(best)

        # Train the model with fixed parameters
        else:
            first_layer = kwargs.get('first_layer', 64)
            second_layer = kwargs.get('second_layer', 128)
            third_layer = kwargs.get('third_layer', 64)
            epoch = kwargs.get('epoch', 100)
            batch_size = kwargs.get('batch_size', 32000)

            for i in range(self.phase_num):
                model = self.cp_generate_model(self.input_dim, self.output_dim, first_layer, second_layer, third_layer)
                x_train, y_train, x_test, y_test = (self.cp_x_train[i], self.cp_y_train[i], self.cp_x_test[i],
                                                    self.cp_y_test[i])
                val_loss = self.cp_model_training(model, x_train, y_train, x_test, y_test, epoch, batch_size)
                self.cp_models.append(model)
                self.val_mse.append(val_loss)

    def cp_bayesian_opt(self, x_train, y_train, x_test, y_test, first_layer=None, second_layer=None, third_layer=None,
                        epoch=None, max_evals=30):
        """
        Bayesian optimization for hyperparameters optimization.

        Parameters
        ----------
        x_train : np.ndarray
            Training input data.
        y_train : np.ndarray
            Training output data.
        x_test : np.ndarray
            Testing input data.
        y_test : np.ndarray
            Testing output data.
        first_layer : list
            Range of neurons in the first layer.
        second_layer : list
            Range of neurons in the second layer.
        third_layer : list
            Range of neurons in the third layer.
        epoch : list
            Range of epochs.
        max_evals : int
            Number of evaluations.

        """

        # Objective function for Bayesian optimization
        def objective(params):

            model = self.cp_generate_model(self.input_dim, self.output_dim, int(params['first_layer']),
                                           int(params['second_layer']), int(params['third_layer']))

            test_mse = self.cp_model_training(model, x_train, y_train, x_test, y_test, int(params['epoch']))

            print('Test mse:', test_mse)
            return {'loss': test_mse, 'status': STATUS_OK}

        if first_layer is None:
            first_layer = [20, 400, 20]
        if second_layer is None:
            second_layer = [20, 500, 20]
        if third_layer is None:
            third_layer = [20, 400, 20]
        if epoch is None:
            epoch = [100, 2500, 200]

        space = {
            'first_layer': hp.quniform('first_layer', first_layer[0], first_layer[1], first_layer[2]),
            'second_layer': hp.quniform('second_layer', second_layer[0], second_layer[1], second_layer[2]),
            'third_layer': hp.quniform('third_layer', third_layer[0], third_layer[1], third_layer[2]),
            'epoch': hp.quniform('epoch', epoch[0], epoch[1], epoch[2])
        }

        trials = Trials()

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        print(best)

        model_temp = []
        loss_temp = []

        # Get the best value from three results.
        # The goal of this is to avoid the possible partial optimization of neural network.
        for i in range(3):
            model_opt = self.cp_generate_model(self.input_dim, self.output_dim, int(best['first_layer']),
                                               int(best['second_layer']), int(best['third_layer']))
            val_mse = self.cp_model_training(model_opt, x_train, y_train, x_test, y_test, int(best['epoch']))

            model_temp.append(model_opt)
            loss_temp.append(val_mse)

        model_best = model_temp[loss_temp.index(min(loss_temp))]

        min_val_mse = min(loss_temp)

        return model_best, min_val_mse, best

    def cp_save_model(self, path):
        """
        Save the composition models.

        Parameters
        ----------
        path : str
            Path to save the models.

        """
        if not os.path.exists(path):
            os.makedirs(path)

        for i, model in enumerate(self.cp_models):
            model.save(os.path.join(path, 'composition model' + str(i)), save_format='tf')

        model_infor = {'val_accuracy': self.val_mse}
        if self.cp_best_params is not None:
            model_infor['best_params'] = self.cp_best_params

        with open(os.path.join(path, 'model infor.json'), 'w') as f:
            json.dump(model_infor, f)


class TrainModels(ClassificationModel, CompositionModels):
    """
    Integrated training models for classification and composition.
    """
    def __init__(self, proj_root_path):
        """
        Initialize the class.

        Parameters
        ----------
        proj_root_path : str
            Project root path.

        """
        super().__init__()
        self.__proj_root_path = proj_root_path

        if os.path.exists(os.path.join(self.__proj_root_path, 'all raw data', 'all data.npy')):

            # Initialize the dataset with the CreateDataset class
            dataset_class = CreateDataset(self.__proj_root_path)  # Load the dataset
            dataset_class.create_dataset()  # Create the dataset

            self.discrete_dataset = dataset_class.discrete_dataset
            self.inputs = dataset_class.inputs
            self.classification_outputs = dataset_class.classification_outputs

        else:
            raise FileNotFoundError('The dataset does not exist.')

        # Prepare for training
        self.cl_prepare_for_training(self.inputs, self.classification_outputs)
        self.cp_prepare_for_training(self.discrete_dataset)

    def train_classification_model(self, bayesian_opt=False, **kwargs):
        """
        Train the classification model.

        Parameters
        ----------
        bayesian_opt : bool
            Whether to use Bayesian optimization.
        **kwargs
            first_layer : int or list
                Number of neurons in the first layer or range of neurons.
            second_layer : int or list
                Number of neurons in the second layer or range of neurons.
            third_layer : int or list
                Number of neurons in the third layer or range of neurons.
            epoch : int or list
                Number of epochs or range of epochs.
            max_evals : int
                Number of evaluations in Bayesian optimization.
        """
        self.cl_train_model(bayesian_opt, **kwargs)

    def train_composition_model(self, bayesian_opt=False, **kwargs):
        """
        Train the composition models.

        Parameters
        ----------
        bayesian_opt : bool
            Whether to use Bayesian optimization.
        **kwargs
            first_layer : int or list
                Number of neurons in the first layer or range of neurons.
            second_layer : int or list
                Number of neurons in the second layer or range of neurons.
            third_layer : int or list
                Number of neurons in the third layer or range of neurons.
            epoch : int or list
                Number of epochs or range of epochs.
            max_evals : int
                Number of evaluations in Bayesian optimization.
        """
        self.cp_train_model(bayesian_opt, **kwargs)

    def auto_integrate_train(self, max_evals=5):
        """
        Automatically train the classification and composition models.

        Parameters
        ----------
        max_evals : int
            Number of evaluations in Bayesian optimization.

        Notes
        -----
        auto_integrate_training will train the classification and composition models with Bayesian optimization.

        """
        self.train_classification_model(bayesian_opt=True, max_evals=max_evals)
        self.train_composition_model(bayesian_opt=True, max_evals=max_evals)

    def save_classification_model(self):
        """
        Save the classification model.

        Notes
        -----
        The model will be saved in the 'classification model' folder.

        """

        path = os.path.join(self.__proj_root_path, 'classification model')
        self.cl_save_model(path)

    def save_composition_model(self):
        """
        Save the composition models.

        Notes
        -----
        The models will be saved in the 'composition model' folder.

        """
        path = os.path.join(self.__proj_root_path, 'composition model')
        self.cp_save_model(path)

    def save_all_models(self):
        """
        Save all models.

        Notes
        -----
        All models will be organized into project path.

        """
        self.save_classification_model()
        self.save_composition_model()

