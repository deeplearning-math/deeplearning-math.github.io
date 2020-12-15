import pandas as pd
import numpy as np
from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F
import os.path as osp
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
DATA_PATH = ""
DatasetType = Tuple[np.ndarray, np.ndarray]

def read_dataset(dataset_name,
                 header='infer',
                 names=None,
                 delim_whitespace=False):
    dataset_path = osp.join(DATA_PATH, dataset_name)
    return pd.read_csv(dataset_path, header=header, names=names, delim_whitespace=delim_whitespace)


def load_hcdr_data():

    df = read_dataset('test.csv')
    df = df.dropna()
    df_train = df.drop(columns=['TARGET'])
    train_cols = df_train.columns
    label = 'TARGET'
    x_df = df[train_cols]
    y_df = df[label]
    return {
        'problem': 'classification',
        'X': x_df,
        'y': y_df,
    }


if __name__ == "__main__":
    load_hcdr_data()


class CustomPipeline(Pipeline):
    """Custom sklearn Pipeline to transform data."""

    def apply_transformation(self, x):
        """Applies all transforms to the data, without applying last estimator.
        Args:
          x: Iterable data to predict on. Must fulfill input requirements of first
            step of the pipeline.
        Returns:
          xt: Transformed data.
        """
        xt = x
        for _, transform in self.steps[:-1]:
            xt = transform.fit_transform(xt)
        return xt


def transform_data(df):
    """Apply a fixed set of transformations to the pd.Dataframe `df`.
    Args:
      df: Input dataframe containing features.
    Returns:
      Transformed dataframe and corresponding column names. The transformations
      include (1) encoding categorical features as a one-hot numeric array, (2)
      identity `FunctionTransformer` for numerical variables. This is followed by
      scaling all features to the range (-1, 1) using min-max scaling.
    """
    column_names = df.columns
    new_column_names = []
    is_categorical = np.array([dt.kind == 'O' for dt in df.dtypes])
    categorical_cols = df.columns.values[is_categorical]
    numerical_cols = df.columns.values[~is_categorical]
    for index, is_cat in enumerate(is_categorical):
        col_name = column_names[index]
        if is_cat:
            new_column_names += [
                '{}: {}'.format(col_name, val) for val in set(df[col_name])
            ]
        else:
            new_column_names.append(col_name)
    cat_ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))

    cat_pipe = Pipeline([cat_ohe_step])
    num_pipe = Pipeline([('identity', FunctionTransformer(validate=True))])
    transformers = [('cat', cat_pipe, categorical_cols),
                    ('num', num_pipe, numerical_cols)]
    column_transform = ColumnTransformer(transformers=transformers)

    pipe = CustomPipeline([('column_transform', column_transform),
                           ('min_max', MinMaxScaler((-1, 1))), ('dummy', None)])
    df = pipe.apply_transformation(df)
    return df, new_column_names


def load_dataset(dataset_name):
    """Loads the dataset according to the `dataset_name` passed.
    Args:
      dataset_name: Name of the dataset to be loaded.
    Returns:
      data_x: np.ndarray of size (n_examples, n_features) containining the
        features per input data point where n_examples is the number of examples
        and n_features is the number of features.
      data_y: np.ndarray of size (n_examples, ) containing the label/target
        for each example where n_examples is the number of examples.
      column_names: A list containing the feature names.
    Raises:
      ValueError: If the `dataset_name` is not in ('Telco', 'BreastCancer',
      'Adult', 'Credit', 'Heart', 'Mimic2', 'Recidivism', 'Fico', Housing').
    """

    if dataset_name == 'hcdr':
        dataset = load_hcdr_data()

    data_x, data_y = dataset['X'].copy(), dataset['y'].copy()
    problem_type = dataset['problem']
    data_x, column_names = transform_data(data_x)
    data_x = data_x.astype('float32')
    if (problem_type == 'classification') and \
            (not isinstance(data_y, np.ndarray)):
        data_y = pd.get_dummies(data_y).values
        data_y = np.argmax(data_y, axis=-1)
    data_y = data_y.astype('float32')
    return data_x, data_y, column_names
def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ActivationLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        raise NotImplementedError("abstract method called")


class ExULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        truncated_normal_(self.weight, mean=4.0, std=0.5)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        exu = (x - self.bias) @ torch.exp(self.weight)
        return torch.clip(exu, 0, 1)


class ReLULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return F.relu((x - self.bias) @ self.weight)


class FeatureNN(torch.nn.Module):
    def __init__(self,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 dropout: float = .5,
                 ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            hidden_layer(shallow_units if i == 0 else hidden_units[i - 1], hidden_units[i])
            for i in range(len(hidden_units))
        ])
        self.layers.insert(0, shallow_layer(1, shallow_units))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(shallow_units if len(hidden_units) == 0 else hidden_units[-1], 1, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.linear(x)


class NeuralAdditiveModel(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 feature_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 ):
        super().__init__()
        self.input_size = input_size

        if isinstance(shallow_units, list):
            assert len(shallow_units) == input_size
        elif isinstance(shallow_units, int):
            shallow_units = [shallow_units for _ in range(input_size)]

        self.feature_nns = torch.nn.ModuleList([
            FeatureNN(shallow_units=shallow_units[i],
                      hidden_units=hidden_units,
                      shallow_layer=shallow_layer,
                      hidden_layer=hidden_layer,
                      dropout=hidden_dropout)
            for i in range(input_size)
        ])
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f_out = torch.cat(self._feature_nns(x), dim=-1)
        f_out = self.feature_dropout(f_out)

        return f_out.sum(axis=-1) + self.bias, f_out

    def _feature_nns(self, x):
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = np.load("input_size.npy")
shallow_units = np.load("shallow_units.npy")
hidden_units = np.load("hidden_units.npy")
model = NeuralAdditiveModel(
        input_size=input_size,
        shallow_units=shallow_units,
        hidden_units=hidden_units,
        shallow_layer=ExULayer,
        hidden_layer=ReLULayer,
        hidden_dropout=0.5,
        feature_dropout=0).to(device)

trained_weight = torch.load("nam.pt")
model.load_state_dict(trained_weight)
test = pd.read_csv("test.csv")
x_test, y_test, column_name = load_dataset("hcdr")

logits =[]
feature_nn_outputs=[]


print("testing")
for i in range(0,48744):
    print(i)
    logits_temp, feature_nn_outputs_temp = model.forward(torch.from_numpy(x_test[None,i]).to(device))
    logits_temp = logits_temp.detach().cpu().numpy()
    feature_nn_outputs_temp = feature_nn_outputs_temp.detach().cpu().numpy()
    logits.append(logits_temp)
    feature_nn_outputs.append(feature_nn_outputs_temp)



print("saving")
np.save("logits.npy", logits)
np.save("feature_nn_outputs.npy", feature_nn_outputs)
