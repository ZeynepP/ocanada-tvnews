import torch
import torchaudio
import re
from datetime import datetime, timedelta
from typing import Union, Tuple, Optional
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import requests
import collections
from sklearn.feature_extraction import text as sklearn_text


from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

def get_audio_segment(video_frames, start_time, end_time, sr=16000):
    """
    Extracts a segment of audio between start_time and end_time from video frames.

    Parameters:
        video_frames (numpy.ndarray): Audio waveform extracted from the video.
        start_time (float): Start time of the segment in seconds.
        end_time (float): End time of the segment in seconds.
        sr (int): Sampling rate of the audio (default: 16000).

    Returns:
        numpy.ndarray: Extracted audio segment as a waveform.
    """
    # Calculate the sample indices for the start and end times
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract the segment
    audio_segment = video_frames[start_sample:end_sample]

    return audio_segment
def get_dashboard(token,base_url, platform, from_date, to_date):
    target_url = f"{base_url}/dashboard/"
    headers = {
        "Authorization": f"Bearer {token}"  # Add Bearer token to the headers
    }
    params = {
        "from_date": from_date,
        "to_date": to_date,
        "platform": platform
    }

    try:
        with requests.session() as session:
            response = session.post(target_url, json=params, headers=headers)
            response.raise_for_status()  # Raise exception for non-200 status codes
            return response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


def get_dates(slicing_window_day, start_date, end_date):
    slicing_window_day = timedelta(days=slicing_window_day)

    start_date = datetime.strptime(start_date, "%d-%m-%Y")
    end_date = datetime.strptime(end_date, "%d-%m-%Y")
    current_date = start_date
    dates = []

    while current_date < end_date:
        temp_end_date = current_date + slicing_window_day
        dates.append((current_date, temp_end_date))
        current_date = temp_end_date  # endDAte is exclusive in API
    return dates
def select_topic_representation(
    ctfidf_embeddings: Optional[Union[np.ndarray, csr_matrix]] = None,
    embeddings: Optional[Union[np.ndarray, csr_matrix]] = None,
    use_ctfidf: bool = False,
    output_ndarray: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Select the topic representation.

    Arguments:
        ctfidf_embeddings: The c-TF-IDF embedding matrix
        embeddings: The topic embedding matrix
        use_ctfidf: Whether to use the c-TF-IDF representation. If False, topics embedding representation is used, if it
                    exists. Default is True.
        output_ndarray: Whether to convert the selected representation into ndarray
    Raises
        ValueError:
            - If no topic representation was found
            - If c-TF-IDF embeddings are not a numpy array or a scipy.sparse.csr_matrix

    Returns:
        The selected topic representation and a boolean indicating whether it is c-TF-IDF.
    """

    def to_ndarray(array: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        if isinstance(array, csr_matrix):
            return array.toarray()
        return array

    if use_ctfidf:
        if ctfidf_embeddings is None:

            repr_, ctfidf_used = embeddings, False
        else:
            repr_, ctfidf_used = ctfidf_embeddings, True
    else:
        if embeddings is None:
            repr_, ctfidf_used = ctfidf_embeddings, True
        else:
            repr_, ctfidf_used = embeddings, False

    return to_ndarray(repr_) if output_ndarray else repr_, ctfidf_used
def get_token(base_url,username, password):
    # Function to perform a GET request to the /meologin route
    params = {"username": username, "password": password}
    response = requests.post(f"{base_url}/meologin", params=params, verify=False)
    return response.json()["access_token"]
# ,
def get_data(token, base_url, platforms,date_start, date_end):
    dates = get_dates(1, date_start, date_end)
    print(dates)
    temp = None
    df = None

    for platform in platforms:
        for sdate, edate in dates:
            df = pd.DataFrame.from_records(
                get_dashboard(token=token,base_url=base_url, platform=platform, from_date=sdate.strftime("%d-%m-%Y"),
                              to_date=edate.strftime("%d-%m-%Y")))
            #   if platform == "twitter":
            #     df = df[df['tags'].apply(lambda x: 'original' in x)]
            df = df[df["seed_type"]!="Foreign"]
            df = df[df["seed_id"] != '0']
            # Fill NaN values in 'text_all' with an empty string
            df["text_all"] = df["text_all"].fillna("")

            df["text"] = df.apply(lambda row: re.sub(r"http\S+", "", row.text_all).lower(), 1)
            df["text"] = df.apply(lambda row: " ".join(filter(lambda x: x[0] != "@", row.text.split())), 1)

            if temp is None:
                temp = df
            else:
                temp = pd.concat([temp, df])
            print(" Done for ", platform, sdate, edate)

    return temp


def get_vocab(df, freq=10):
    combined_stop_words = get_stop_word()
    # Extract vocab to be used in BERTopic
    vocab = collections.Counter()
    tokenizer = CountVectorizer().build_tokenizer()
    for doc in tqdm(df.text.to_list()):
        vocab.update(tokenizer(doc))
    vocab = [word for word, frequency in vocab.items() if frequency >= freq and word not in combined_stop_words];
    len(vocab)


def get_stop_word():
    english_stop_words = sklearn_text.ENGLISH_STOP_WORDS

    # Define French stop words (you can also get this from external sources like NLTK)
    french_stop_words = {'rt',
                         'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux',
                         'il', 'je', 'la', 'le', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon',
                         'ne', 'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui',
                         'sa', 'se', 'ses', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une',
                         'vos', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été',
                         'étée', 'étées', 'étés', 'étant', 'étante', 'étants', 'étantes', 'suis', 'es', 'est',
                         'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront',
                         'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 'étions',
                         'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons',
                         'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent'
                         }

    # Combine English and French stop words
    return list(set(english_stop_words).union(french_stop_words))

def get_video( filepath, sampling_rate=16000) -> torch.Tensor:
    speech_array, sr = torchaudio.load(filepath)

    # Transform to mono
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(sr, sampling_rate)
        speech_array = transform(speech_array)

    speech_array = speech_array.squeeze().numpy()
    return speech_array

