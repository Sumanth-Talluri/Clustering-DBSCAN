from clustering import createSlices, findClusters
import datetime

st = datetime.datetime.now()
createSlices()
et = datetime.datetime.now()

st1 = datetime.datetime.now()
findClusters()
et1 = datetime.datetime.now()

print("Time Taken for slicing:", et-st)
print("Time Taken for clustering:", et1-st1)