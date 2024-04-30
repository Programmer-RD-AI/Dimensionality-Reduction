from DimRed import *


def load_dataset(
    test_split: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the digits dataset and split it into training and testing sets.
    Parameters:
        test_split (float): The proportion of the dataset to include in the test split.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the training and testing data and labels.
    """
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=RANDOM_STATE, shuffle=True
    )
    return X_train, X_test, y_train, y_test


def label_encoding(y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
    """
    Encode the labels using label encoding.
    Parameters:
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The testing labels.
    Returns:
        Tuple: A tuple containing the encoded training and testing labels.
    """
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


def average_metric(metric, dictionaries: List[Dict[str, Union[str, int]]]) -> float:
    """
    Calculate the average value of a given metric from a list of dictionaries.
    Parameters:
        metric (str): The metric to calculate the average for.
        dictionaries (List[Dict[str, Union[str, int]]]): A list of dictionaries containing the metrics.
    Returns:
        float: The average value of the metric.
    """
    avg = 0
    for dictionary in dictionaries:
        avg += dictionary[metric]
    return float(avg / len(dictionaries))


def add_to_dictionary(
    dictionary: Dict[str, List[Union[str, int]]], list_of_values: List[Union[str, int]]
) -> Dict[str, List[Union[str, int]]]:
    """
    Add a list of values to a dictionary.
    Parameters:
        dictionary (Dict[str, List[Union[str, int]]]): The dictionary to add the values to.
        list_of_values (List[Union[str, int]]): The list of values to add.
    Returns:
        Dict[str, List[Union[str, int]]]: The updated dictionary.
    """
    for idx, key in enumerate(dictionary):
        dictionary[key].append(list_of_values[idx])
    return dictionary


def director_exist(path):
    """
    Create the directory if it does not exist.
    Parameters:
        path (str): The path of the directory.
    Returns:
        str: The path of the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path
