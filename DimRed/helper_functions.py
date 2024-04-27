from DimRed import *


def load_dataset(test_split: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, stratify=y,
                                                        random_state=RANDOM_STATE, shuffle=True)
    return X_train, X_test, y_train, y_test


def label_encoding(y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


def average_metric(metric, dictionaries: List[Dict[str, Union[str, int]]]) -> float:
    avg = 0
    for dictionary in dictionaries:
        avg += dictionary[metric]
    return float(avg / len(dictionaries))


def add_to_dictionary(dictionary: Dict[str, List[Union[str, int]]], list_of_values: List[Union[str, int]]) -> Dict[str, List[Union[str, int]]]:
    for idx, key in enumerate(dictionary):
        dictionary[key].append(list_of_values[idx])
    return dictionary
