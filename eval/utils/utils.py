import os
import re

def extract_text_from_vtt(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    text_lines = []
    for line in lines:
        if re.match(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', line):
            continue
        if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
            continue
        if line.strip() == '':
            continue
        clean_line = re.sub(r'<[^>]+>', '', line).strip()
        if clean_line:
            text_lines.append(clean_line)

    return ' '.join(text_lines)

def extract_text_by_time_range(filepath, start_seconds, end_seconds):
    def time_to_seconds(time_str):
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    text_lines = []
    within_range = False
    for line in lines:
        time_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
        if time_match:
            start, end = time_match.groups()
            start_sec = time_to_seconds(start)
            end_sec = time_to_seconds(end)
            within_range = start_seconds <= end_sec and end_seconds >= start_sec
            continue
        if within_range:
            if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
                continue
            if line.strip() == '':
                continue
            clean_line = re.sub(r'<[^>]+>', '', line).strip()
            if clean_line:
                text_lines.append(clean_line)

    return ' '.join(text_lines)

def get_scripts_for_videos(videos: list, startend_times=None):
    scripts = []
    if not startend_times: startend_times = [None] * len(videos)
    for video, startend_time in zip(videos, startend_times):
        video_name = os.path.splitext(video)[0]
        if "videorag" in video:
            script_path = video_name + '.txt'
            with open(script_path, 'r') as f:
                script = ' '.join([x.strip() for x in f.readlines()])
        elif "LVBench" in video or "cinepile" in video:
            script_path = video_name + '.en.vtt'
            if startend_time:
                script = extract_text_by_time_range(script_path, startend_time[0], startend_time[1])
            else:
                script = extract_text_from_vtt(script_path)
        else:
            raise ValueError(f"No script found for video: {video}")
        scripts.append(script)
    return scripts
