import cupy
import sys

for i in range(0,32):
    print("probing device {}...".format(i))
    try:
      device=cupy.cuda.Device(i)
      print("meminfo {}".format(device.mem_info))
    except:
      print("no device {}".format(i))
