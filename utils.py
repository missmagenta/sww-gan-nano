def time_to_sec(time):
     min, sec = (time.split()[0], time.split()[2])
     seconds = int(min) * 60 + int(sec)
     return seconds
