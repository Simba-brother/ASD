def convert_to_hms(seconds):
    hours = int(seconds // 3600)
    remaining_seconds = seconds % 3600
    minutes = int(remaining_seconds // 60)
    seconds = remaining_seconds % 60
    return hours, minutes, seconds