import pandas as pd


def getreps(PATH, terminals=False):
    """Binary phonological reps from CSV.

    Parameters
    ----------
    PATH : str
        Path to the csv containing phonological representations.
    terminals : bool
        Specify whether to add end-of-string and start-of-string
        features to reps (default is not/ False). If true
        the character "%" is used for eos, and "#" for sos.
        Note that if set to true a new key-value pair is created
        in return dict for each of these characters, and a feature
        for each is added to every value in the dictionary.
    Returns
    -------
    dict
        A dictionary of phonemes and their binary representations
    """
    df = pd.read_csv(PATH)
    df = df[df["phone"].notna()]
    df = df.rename(columns={"phone": "phoneme"})
    df = df.set_index("phoneme")
    feature_cols = [column for column in df if column.startswith("#")]
    df = df[feature_cols]
    df.columns = df.columns.str[1:]
    dict = {}
    for index, row in df.iterrows():
        dict[index] = row.tolist()

    if terminals:
        for k, v in dict.items():
            dict[k].append(0)
        dict["#"] = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
        ]
        dict["%"] = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
        ]

    return dict


def key(dict, value):
    """Retrieve the key for a given value in dict. An unholy python function.

    While the function nominally retrieves THE key in dict for the value given,
    note that the function actually returns A key in dict for the value give. That is,
    this function makes an assumption about the unique identity of key-value pairs
    in the dict provided. Be cautious.

    Parameters
    ----------
    dict : dict
        The dictionary from which you want to retrieve the key for the value.

    value : any
        The value associated with a key that you want to retrieve from dict.

    Returns
    -------
    key
        The type of the key is specified based on the structure of dict. Canonically,
        the function will return a string.

    """

    for k, v in dict.items():
        if value == v:
            return k


def remove_right_pad(x, target="_"):
    while x and x[-1] == target:
        x.pop()
    return x


def convert_numeric_prediction(prediction, phonreps, trim=True, num_target_features=31):
    prediction_as_strings = [
        key(phonreps, phoneme.tolist()[:num_target_features]) for phoneme in prediction
    ]

    prediction_as_strings = ["XX" if e is None else e for e in prediction_as_strings]

    if trim:
        return remove_right_pad(prediction_as_strings)
    else:
        return prediction_as_strings
