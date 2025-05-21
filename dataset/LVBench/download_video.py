import os
import re
import logging
from tqdm import tqdm
from yt_dlp import YoutubeDL

output_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_video.log')

logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S', 
    handlers=[
        logging.FileHandler(output_log_file, mode='a'),
        # logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def extract_video_id(url):
    match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Invalid YouTube URL: {url}")

def download_video(url, output_path):
    video_id = extract_video_id(url)
    options = {
        'outtmpl': f'{output_path}/{video_id}.%(ext)s',
        'format': 'best',
        'quiet': True,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        metadata = {
            "VideoID": video_id,
            "Author": info.get('uploader'),
            "Title": info.get('title'),
            "Views": info.get('view_count'),
        }
    return metadata

def download_videos_from_file(file_path, output_path):
    with open(file_path, 'r') as file:
        urls = file.readlines()

    all_metadata = []
    for url in tqdm(urls):
        url = url.strip()
        if url:
            try:
                metadata = download_video(url, output_path)
                all_metadata.append(metadata)
                logger.info(f"Downloaded: {metadata['Title']} (ID: {metadata['VideoID']})")
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                logger.error(f"Failed to download {url}: {e}")
    
    return all_metadata

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Download YouTube videos from a list of URLs.")
    parser.add_argument('--input', type=str, default='videos.txt', help='Input file with YouTube URLs')
    parser.add_argument('--output', type=str, default='videos', help='Output directory for downloaded videos')
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    download_videos_from_file(input_file, output_dir)
