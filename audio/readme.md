Konverzija fajlova

ffmpeg -i Test.m4a output.wav

Split files into segments
ffmpeg -i kuca.wav -f segment -segment_time 50 -c copy kuca%03d.wav