import time
st=time.time()
import backpropagation_code

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0} Second".format(sec))
  
time_convert(time.time()-st)