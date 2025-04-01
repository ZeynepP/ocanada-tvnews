import os
import glob
import json
import argparse
import logging
import logging.config
import pandas as pd
import numpy as np
from video_analyzer import VideoAnalyzer


def setup_logging():
    logging.config.fileConfig('logging_config.cfg')
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def load_input_data(input_folder, year=2022):
    """
    Load all video metadata JSON files from the given folder and return a DataFrame.
    """
    all_data = []
    files = glob.glob(os.path.join(input_folder, "*.json"))

    for input_file in files:
        video_prefix = os.path.splitext(os.path.basename(input_file))[0]
        with open(input_file, 'r') as f:
            data = json.load(f)
            for record in data:
                record['video_prefix'] = video_prefix
            all_data.extend(data)

    df = pd.DataFrame(all_data)

    if 'videoPublishedAt' in df.columns:
        df['videoPublishedAt'] = pd.to_datetime(df['videoPublishedAt'], errors='coerce')
        df = df[df['videoPublishedAt'].dt.year > year]
        logging.info(f"Filtered dataset to {len(df)} videos published after {year}.")

    return df


def process_video(row, analyzer):
    video_id = row.get('videoId')
    video_prefix = row.get('video_prefix')
    video_date = row.get('videoPublishedAt')

    if pd.notnull(video_id) and pd.notnull(video_date):
        try:
            analyzer.run_analysis(video_id=video_id, video_prefix=video_prefix, video_date=video_date)
            logging.debug(f"Completed: {video_id}")
        except Exception as ex:
            logging.error(f"Failed processing {video_id}: {ex}")
    else:
        logging.warning(f"Skipped: Invalid row {row}")


def main(hf_token, input_folder):
    setup_logging()

    df = load_input_data(input_folder)
    if df.empty:
        logging.error("No valid videos found in the input folder.")
        return

    analyzer = VideoAnalyzer(output_dir="./output", model_dir="./models", hg_token=hf_token)

    for idx, row in df.iterrows():
        logging.info(f"[{idx + 1}/{len(df)}] Processing video {row.get('videoId')}")
        process_video(row, analyzer)

    logging.info("✅ All videos processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gender/topic analysis pipeline on YouTube data.")
    parser.add_argument("--hf_token", required=True, help="Hugging Face token for diarization model access")
    parser.add_argument("--input_folder", required=True, help="Path to folder containing input video ID JSON files")

    args = parser.parse_args()
    main(hf_token=args.hf_token, input_folder=args.input_folder)
