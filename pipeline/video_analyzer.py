import logging
import os
import traceback
import pandas as pd
import numpy as np
import torchaudio
import whisperx
import ffmpeg
from pyannote.database.util import load_rttm
from pytubefix import YouTube
from pyannote.audio import Pipeline
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import gc
import utils
#from inaSpeechSegmenter import Segmenter
import logging.config
import torch

from transformers import pipeline
logging.config.fileConfig('logging_config.cfg')
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("moviepy").setLevel(logging.CRITICAL)
logging.getLogger("inaSpeechSegmenter.segmenter").setLevel(logging.CRITICAL)
from pytubefix.innertube import _default_clients
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID"]

class VideoAnalyzer:
    def __init__(self, output_dir, model_dir, hg_token):
        self.output_dir = output_dir
        if not os.path.exists(os.path.join(self.output_dir, "topics_gender")):
            os.makedirs(os.path.join(self.output_dir, "topics_gender"))
        self.topics_path = os.path.join(self.output_dir, "topics_gender")


        if not os.path.exists(os.path.join(self.output_dir, "videos")):
            os.makedirs(os.path.join(self.output_dir, "videos"))
        self.videos_output = os.path.join(self.output_dir, "videos")

        if not os.path.exists(os.path.join(self.output_dir, "diarization")):
            os.makedirs(os.path.join(self.output_dir, "diarization"))
        self.diarization_output = os.path.join(self.output_dir, "diarization")

        self.device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 1
        self.compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)"

        try:
            self.whisperx_model = whisperx.load_model(
                "large-v2",
                self.device,
                compute_type=self.compute_type,
                download_root=model_dir
            )
        except ValueError as e:
            logging.warning(f"Compute type {self.compute_type} is not supported. Falling back to float32.")
            self.compute_type = "float32"
            self.whisperx_model = whisperx.load_model(
                "large-v2",
                self.device,
                compute_type=self.compute_type,
                download_root=model_dir
            )

        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                             use_auth_token=hg_token)
        self.diarization_pipeline.to(torch.device(self.device))


        self.label2id = {
            "female": 0,
            "male": 1
        }

        self.id2label = {
            0: "female",
            1: "male"
        }

        self.num_labels = 2
        self.gender_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")
        self.gender_pipeline = AutoModelForAudioClassification.from_pretrained(
            pretrained_model_name_or_path="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
        )

        self.gender_pipeline.to(torch.device(self.device))
        #self.ina_segmenter = Segmenter()
        self.classifier = pipeline("text-classification", model="classla/multilingual-IPTC-news-topic-classifier",
                              device=self.device, max_length=512, truncation=True)


    def download_video(self, video_id, video_prefix=""):
        url = "https://www.youtube.com/watch?v=%s" % video_id
        filename = video_prefix + "_" + video_id + ".mp4"
        file_path = os.path.join(self.videos_output, filename)

        if not os.path.exists(file_path):
            yt = YouTube(url, use_oauth=True)
            try:
                yt.streams.get_audio_only().download(
                    output_path=self.videos_output,
                    filename=video_prefix + "_" + video_id + ".mp4")
            except Exception as ex:
                yt = YouTube(url+"&rco=1", use_oauth=True)
                yt.streams.get_audio_only().download(
                    output_path=self.videos_output,
                    filename=video_prefix + "_" + video_id + ".mp4")
        return file_path

    # def transcript_video(self, video_path, file_path):
    #     if not os.path.exists(file_path):
    #         result = self.whisperx_model.transcribe(video_path, batch_size=self.batch_size)
    #         with open(file_path, "w+") as wout:
    #             json.dump(result, wout)
    #     return file_path

    def diarize_video(self, video_id, video_prefix=""):
        video_path = os.path.join(self.videos_output, video_prefix + "_" + video_id + ".mp4")
        file_path = os.path.join(self.diarization_output, video_prefix + "__" + video_id + ".rttm")
        if not os.path.exists(file_path):
            waveform, sample_rate = torchaudio.load(video_path)
            diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
            with open(file_path, "w+") as wout:
                diarization.write_rttm(wout)
            return {"waveform":diarization}
        else:
            return load_rttm(file_path)



    def topic_detection(self, data, video, channel, video_date):
        df = pd.DataFrame(data)
        df["video"] = video
        df["id"] = video
        df["channel"] = channel
        df["date"] = video_date
        presenter = df.head(1)["speaker"].values[0]
        df["presenter"] = np.where(df["speaker"] == presenter, 1, 0)
        df['group'] = (df['speaker'] != df['speaker'].shift()).cumsum()
        concatenated_text = df.groupby('group')['text'].apply(' '.join).reset_index()

        results = self.classifier(concatenated_text["text"].tolist())

        concatenated_text['topic'] = [result['label'] for result in results]
        concatenated_text['topic_probability'] = [result['score'] for result in results]

        df = df.merge(concatenated_text[['group', 'topic', 'topic_probability']], on='group', how='left')

        filename = channel + "_" + video + ".json"
        df.to_json(f"{self.topics_path}/{filename}", orient='records', lines=True,force_ascii=False)

    def gender_detection(self, diarization, video_id, video_date, video_prefix, lang="en"):
        data = []
        video_path = os.path.join(self.videos_output, video_prefix + "_" + video_id + ".mp4")
        speakers = {}
        speakers_proba = {}
        input = ffmpeg.input(video_path)

        for turn, _, speaker in diarization["waveform"].itertracks(yield_label=True):
            logging.debug(turn)
            obj = {}
            obj["speaker"] = speaker
            obj["start"] = turn.start
            obj["end"] = turn.end
            obj["duration"] = turn.duration
            temp_file = video_prefix + "_" + video_id + "temp.mp4"
            if turn.duration > 0.3 :
                try:
                    input.output(temp_file, ss=turn.start, to=turn.end).run(overwrite_output=True)
                    speech_array = utils.get_video(temp_file)

                    if len(speech_array) < 2:  # Replace `2` with the minimum required input size for your model
                        logging.warning(f"Skipping segment for speaker {speaker}: Input size too small.")
                        continue

                    # Now run gender classification only once per speaker:
                    if speaker not in speakers:
                        input_values = self.gender_feature_extractor(
                            speech_array,
                            sampling_rate=16000,
                            padding='longest',
                            return_tensors='pt'
                        ).input_values.to(self.device)

                        with torch.no_grad():
                            logits = self.gender_pipeline(input_values).logits
                            prob = torch.nn.functional.softmax(logits, dim=1)
                            max_prob, max_prob_class = torch.max(prob, dim=1)

                        speakers[speaker] = max_prob_class.item()
                        speakers_proba[speaker] = max_prob.item()

                    obj["gender"] = speakers[speaker]
                    obj["gender_probability"] = speakers_proba[speaker]

                    result = self.whisperx_model.transcribe(
                        speech_array,
                        language=lang,
                        batch_size=self.batch_size
                    )
                    torch.cuda.empty_cache()
                    obj["text"] = " ".join([segment["text"] for segment in result["segments"]])
                    obj["date"] = video_date
                    data.append(obj)

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logging.error(f"CUDA out of memory for speaker {speaker}. Skipping this segment.")
                        torch.cuda.empty_cache()  # Clear GPU memory
                        raise
                    else:
                        logging.error(f"Error processing segment for speaker {speaker}: {str(e)}")
                        continue  # Re-raise any other runtime errors for debugging

                except Exception as e:
                    logging.error(f"Unhandled error processing segment for speaker {speaker}: {str(e)}")
                    raise  # Re-raise any unexpected errors to prevent silent failures


                finally:
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass

        return data

    def run_analysis(self, video_id, video_date, video_prefix=""):
            lang = "en"
            if "RadioCanada" in video_prefix:
                lang="fr"
            logging.debug("Working video {} {}".format(video_prefix,video_id))
            filename = video_prefix + "_" + video_id + ".json"

            try:

                video_path = self.download_video(video_id, video_prefix)
                logging.debug("{} {} Download video ok".format(video_id,video_date))

                diarization= self.diarize_video(video_id, video_prefix)
                logging.debug("{} diarization video ok".format(video_id))

                data = self.gender_detection(diarization, video_id,video_date, video_prefix,  lang=lang)
                logging.debug("{} gender video ok".format(video_id))

                self.topic_detection(data, video_id,video_prefix, video_date)
                logging.debug("{} topic video ok".format(video_id))

            except Exception as ex:
                    logging.error(f"{video_id} : Error in video {video_id} and error {ex}")
                    traceback.print_exc()

            gc.collect()
            torch.cuda.empty_cache()
