import os
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
from pathlib import Path
import shutil
import pandas as pd
import glob

# this function will create the slices of the scan image
def createSlices():

	# deleting the slices folder if it is already present 
	slicesPresent = os.path.isdir(os.path.join(os.getcwd(),"Slices"))
	if slicesPresent:
		shutil.rmtree(os.path.join(os.getcwd(),"Slices"))
		#  waiting until the folder and it's contents are deleted
		while os.path.isdir(os.path.join(os.getcwd(),"Slices")):
			pass

	# creating a slices folder
	slicesPresent = os.path.isdir(os.path.join(os.getcwd(),"Slices"))
	if not slicesPresent:
		os.mkdir(os.path.join(os.getcwd(),"Slices"))

	# getting the images one by one from the data
	for img_path in glob.glob("./testPatient/*thresh.png"):

		fileName = Path(img_path).name.split('.')[0]

		# creating a folder with the image name inside slices folder
		folderPresent = os.path.isdir(os.path.join(os.path.join(os.getcwd(),"Slices"),fileName))
		if not folderPresent:
			os.mkdir(os.path.join(os.path.join(os.getcwd(),"Slices"),fileName))

		# reading the thresh image
		image = cv2.imread(img_path)
		# getting the length of the image
		image_length = image.shape[1]
		# cropping the top and the right side noise in the image
		final_image = image[20:,:image_length-82,:]

		#  calculating the vertical coordinates of R in the image
		v_rs = []
		for i in range(len(final_image)):
			for j in range(len(final_image[i])):
				if final_image[i][j].sum() >= 1:
					if j in [0,1,2,3,4]:
						v_rs.append(i)
		v_rs = set(v_rs)
		v_lst = list(v_rs)
		v_lst.sort()
		# storing the starting R coordinates in v_r_start
		v_r_start = [v_lst[i] for i in range(0,len(v_lst),5)]

		#  calculating the horizontal coordinates of R in the image
		h_rs = []
		for i in range(len(final_image)):
			for j in range(len(final_image[i])):
				if final_image[i][j].sum() >= 1:
					if i==v_r_start[0]:
						h_rs.append(j)
		h_rs = set(h_rs)
		h_lst = list(h_rs)
		h_lst.sort()
		# storing the starting R coordinates in h_r_start
		h_r_start = [h_lst[i] for i in range(0,len(h_lst),4)]

		# dividing the final image into columns based on R and storing in the imagesArr
		imagesArr = []
		# each R is 5 pixels hoizontall
		w = 5
		i = 1
		while i < len(h_r_start):
			col = final_image[:,w:h_r_start[i],:]
			imagesArr.append(col)
			w = h_r_start[i]+5
			i+=1
		# we will miss the last col in the above loop so appending it
		imagesArr.append(final_image[:,w:,:])

		#  from the columns we need to divide them into slices
		slices = []
		for img in imagesArr:
			h = 0
			i = 0
			while i < len(v_r_start):
				slice = img[h:v_r_start[i],:,:]
				slices.append(slice)
				#  each R is 4 pixels vertically
				h = v_r_start[i]+4
				i+=1
		#  after above loop we will get all the slices in the slices list

		# detecting contours in the slices
		i = 1
		for slice in slices:
			# Detecting the Brain boundary
			b, g, r = cv2.split(slice)
			# detecting contours using green channel and without thresholding
			contours, hierarchy = cv2.findContours(image=g, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
			# draw contours on the slice
			image_contour_green = slice.copy()
			cv2.drawContours(image=image_contour_green, contours=contours, contourIdx=-1, color=(51,214,255), thickness=1, lineType=cv2.LINE_AA)
			#  checking if there is a contour in the slice
			if len(contours):
				cv2.imwrite(os.path.join(os.path.join(os.getcwd(),"Slices"), fileName) + '/' + str(i) + ".png", slice)
				i+=1

		print("Slices for", fileName, "created successfully")


# this function is used to find the number of clusters
def findClusters():

	# deleting the clusters folder if it is already present 
	clustersPresent = os.path.isdir(os.path.join(os.getcwd(),"Clusters"))
	if clustersPresent:
		shutil.rmtree(os.path.join(os.getcwd(),"Clusters"))
		#  waiting until the folder and it's contents are deleted
		while os.path.isdir(os.path.join(os.getcwd(),"Clusters")):
			pass

	# creating a cluster folder
	clustersPresent = os.path.isdir(os.path.join(os.getcwd(),"Clusters"))
	if not clustersPresent:
		os.mkdir(os.path.join(os.getcwd(),"Clusters"))

	# getting the images one by one from the data
	slice_path = "./Slices"
	thresh_folders = os.listdir(slice_path)

	for thresh in thresh_folders:
		clusters_count = []
		slices_path = "./Slices/"+thresh
		slices = os.listdir(slices_path)

		# creating a folder with the image name inside clusters folder
		folderPresent = os.path.isdir(os.path.join(os.path.join(os.getcwd(),"Clusters"),thresh))
		if not folderPresent:
			os.mkdir(os.path.join(os.path.join(os.getcwd(),"Clusters"),thresh))

		for slice in slices:
			
			# resolving the path to each slice
			img_path = "./Slices/"+thresh+"/"+slice
			# reading the slice image
			img = cv2.imread(img_path)
			# converting from BGR colour space to RGB
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
			# converting the image into a flat image
			flat_image12 = rgb.reshape((-1,3))
			# converting from int to float
			flat_image12 = np.float32(flat_image12)

			# converting the image ndarray into a list 
			flat_image12 = flat_image12.tolist()

			# converting to black if less than 135 since we don't need that
			for j in range(0,len(flat_image12)):
				if sum(flat_image12[j]) < 135:
					flat_image12[j] = [0,0,0] 

			# converting some junk pixels to black
			for k in range(0,len(flat_image12)):
				if sum(flat_image12[k])/3 == flat_image12[k][1]:
					flat_image12[k] = [0,0,0]

			# changing the required clusters pixels to yellow
			for l in range(0,len(flat_image12)):
				if flat_image12[l] != [0,0,0]:
					flat_image12[l] = [0,255,255]

			# converting the list back into an ndarray
			flat_image12=np.array(flat_image12)

			# running DBSCAN to find the number of clusters
			db = DBSCAN(eps = 1,min_samples=112).fit(flat_image12)  
			core_samples_mask=np.zeros_like(db.labels_,dtype = bool) 
			core_samples_mask[db.core_sample_indices_] = True 
			labels = set([label for label in db.labels_ if label >= 0])

			# resolving the path to store the image
			cluster_path = "./Clusters/"+thresh+"/"+slice
			# converting it back
			new_img = flat_image12.reshape((img.shape))
			# saving the image
			cv2.imwrite(cluster_path,new_img)
			# calculating the count
			clusters_count.append(len(set(labels))-1)
		
		# getting the number of the slice from name
		slice_number = []
		for slice in slices:
			slice_number.append(int(slice.split(".")[0]))

		# creating a CSV for each thresh folder
		df = pd.DataFrame({"SliceNumber":slice_number,"ClusterCount":clusters_count})
		df.to_csv("./Clusters/"+thresh+"/"+thresh+".csv",index=False)
		# sorting the file based on Slice Number
		df1 = pd.read_csv("./Clusters/"+thresh+"/"+thresh+".csv")
		df1.sort_values(by=["SliceNumber"], inplace=True)
		df1.to_csv("./Clusters/"+thresh+"/"+thresh+".csv",index=False)

		print("Clusters of all the slices of", thresh, "are stored successfully")
