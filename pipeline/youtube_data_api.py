import json
import os
import argparse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

YOUTUBE_API_MAX_VIDEOS_PER_CALL = 50


class YouTubeAPIClientGoogle:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key, cache_discovery=False)

    def get_playlist_video_ids(self, playlist_id):
        """
        Retrieve all video IDs and publish dates from a YouTube playlist.
        """
        results = []
        next_page_token = None

        while True:
            try:
                response = self.youtube.playlistItems().list(
                    part="contentDetails",
                    maxResults=YOUTUBE_API_MAX_VIDEOS_PER_CALL,
                    playlistId=playlist_id,
                    pageToken=next_page_token
                ).execute()
            except HttpError as e:
                print(f"Error retrieving playlist items: {e}")
                break

            for item in response.get("items", []):
                video_id = item["contentDetails"].get("videoId")
                published_at = item["contentDetails"].get("videoPublishedAt")
                if video_id and published_at:
                    results.append({
                        "videoId": video_id,
                        "videoPublishedAt": published_at
                    })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return results

    def get_video_metadata(self, video_ids):
        """
        Retrieve video metadata (snippet, statistics, contentDetails) for a list of video IDs.
        """
        all_videos = []
        # YouTube API allows up to 50 IDs per call
        for i in range(0, len(video_ids), YOUTUBE_API_MAX_VIDEOS_PER_CALL):
            chunk = video_ids[i:i + YOUTUBE_API_MAX_VIDEOS_PER_CALL]
            try:
                response = self.youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(chunk)
                ).execute()

                for item in response.get("items", []):
                    data = {}
                    data["video_metadata"] = {}
                    data["video_metadata"]["video"] = item
                    all_videos.append(data)
                break
            except HttpError as e:
                print(f"Error retrieving video metadata: {e}")
                continue

        return all_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch YouTube video IDs and/or metadata.")
    parser.add_argument("--api_key", required=True, help="YouTube Data API key")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--playlist_id", help="YouTube playlist ID to fetch video list, empy if you want to get videos metadata")
    group.add_argument("--input_file", help="JSON file with videoId list, empty if you want to get playlist video ids")

    parser.add_argument("--output", required=True, help="Output file path (JSON)")

    args = parser.parse_args()
    client = YouTubeAPIClientGoogle(api_key=args.api_key)

    if args.playlist_id:
        video_list = client.get_playlist_video_ids(args.playlist_id)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(video_list, f, ensure_ascii=False)
        print(f"Saved {len(video_list)} video entries to {args.output}")

    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            video_list = json.load(f)
        video_ids = [item["videoId"] for item in video_list if "videoId" in item]
        metadata = client.get_video_metadata(video_ids)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
        print(f"Saved metadata for {len(metadata)} videos to {args.output}")
