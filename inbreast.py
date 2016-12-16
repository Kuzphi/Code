import dicom # some machines not install pydicom
import gzip
import scipy.misc
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import cPickle
import matplotlib
import matplotlib.pyplot as plt 
from skimage.filter import threshold_otsu
import os
from os.path import join as join
import csv
import scipy.ndimage
#import cv2
path = '../trainingData/'
preprocesspath = '../preprocessedData/'
labelfile = '../label.txt'

def readlabel():
	'''read the label as a dict from labelfile'''
	mydict = {}
	with open(labelfile, 'r') as f:
		flines = f.readlines()
		for line in flines:
			data = line.split()
			if int(data[1]) == 0:
				mydict[data[0]] = int(data[1])
			else:
				assert(int(data[1])==2 or int(data[1])==1)
				mydict[data[0]] = int(data[1])-1
	return mydict

def readdicom(mydict):
	'''read the dicom image, rename it consistently with the name in labels, crop and resize, and save as pickle.
	mydict is the returned value of readlabel'''
	img_ext = '.dcm.gz'
	# print os.listdir(mydict)
	img_fnames = [x for x in os.listdir(mydict) if x.endswith(img_ext)]	
	for name in img_fnames:
		print name
		# with gzip.open(join(path,name),'rb') as infile:
		# 	dicom_content = dicom.read_file(infile)
		dicom_content = dicom.read_file(name)
		img = dicom_content.pixel_array
		# fig = plt.figure()
		# ax1 = plt.subplot(3,3,1)
		# ax2 = plt.subplot(3,3,2)
		# ax3 = plt.subplot(3,3,3)
		# ax4 = plt.subplot(3,3,4)
		# ax5 = plt.subplot(3,3,5)
		# ax6 = plt.subplot(3,3,6)
		# ax7 = plt.subplot(3,3,7)
		# ax8 = plt.subplot(3,3,8)
		# ax9 = plt.subplot(3,3,9)
		# ax1.imshow(img, cmap='Greys_r')
		# ax1.set_title('Original')
		# ax1.axis('off')

		thresh = threshold_otsu(img)
		binary = img > thresh
		# ax2.imshow(binary, cmap='Greys_r')
		# ax2.set_title('mask')
		# ax2.axis('off')

		minx, miny = 0, 0
		maxx, maxy = img.shape[0], img.shape[1]
		for xx in xrange(img.shape[0]):
			if sum(binary[xx, :]==0) < binary.shape[1]-60:
				minx = xx
				break

		for xx in xrange(img.shape[0]-1,0,-1):
			if sum(binary[xx, :]==0) < binary.shape[1]-60:
				maxx = xx
				break

		# if names[3] == 'R':
		# 	maxy = img.shape[1]
		for yy in xrange(img.shape[1]):
			if sum(binary[:,yy]==0) > binary.shape[0]-10: 
				miny = yy
				break

		# else:
		# 	miny = 0
		for yy in xrange(img.shape[1] - 1, 0, -1):
			if sum(binary[:,yy]==0) > binary.shape[0]-10: 
				maxy = yy
				break

		print(minx, maxx, miny, maxy)
		# ax3.set_title('Foreground')
		# ax3.imshow(img[minx:maxx+1, miny:maxy+1], cmap='Greys_r')
		# ax3.axis('off')
		#.pickle
		img = img.astype(np.float32)
		img = scipy.misc.imresize(img[minx:maxx+1, miny:maxy+1], (224, 224), interp='cubic')

		fname = name.split('.')[0]
		print fname

		# if os.path.isdir(join(preprocesspath, fname)+'.pickle'):
		# 	os.mknod(join(preprocesspath, fname)+'.pickle')

		with open(join(preprocesspath, fname)+'.pickle', 'w') as pickle_file:
			cPickle.dump(img, pickle_file)

		# ax4.set_title('Resize')
		# ax4.imshow(img, cmap='Greys_r')
		# ax4.axis('off')

		#.normpickle
		img = img.astype(np.float32)
		img -= np.mean(img)
		img /= np.std(img)
		# ax5.set_title('Norm')
		# ax5.imshow(img, cmap='Greys_r')
		# ax5.axis('off')
		with open(join(preprocesspath, fname)+'norm.pickle', 'w') as pickle_file:
			cPickle.dump(img, pickle_file)
		# cPickle.dump(img, join(preprocesspath, name)+'norm.pickle')
		#imgshape = img.shape

		# img = np.fliplr(img)
		# ax6.set_title('Flip')
		# ax6.imshow(img, cmap='Greys_r')
		# ax6.axis('off')

		# num_rot = np.random.choice(4)               #rotate 90 randomly
		# img = np.rot90(img, num_rot)
		# ax7.set_title('Rotation')
		# ax7.imshow(img, cmap='Greys_r')
		# ax7.axis('off')
		# fig.savefig(join(preprocesspath, name)+'.jpg')
		# plt.close(fig)

def cvsplit(fold, totalfold, mydict):
	'''get the split of train and test
	fold is the returned fold th data, from 0 to totalfold-1
	total fold is for the cross validation
	mydict is the return dict from readlabel'''
	skf = StratifiedKFold(n_splits=totalfold)  # default shuffle is false, okay!
	#readdicom(mydict)
	y = mydict.values()
	x = mydict.keys()
	count = 0
	for train, test in skf.split(x,y):
		if count == fold:
			return train, test
		count += 1

def cvsplitenhance(fold, totalfold, mydict):
	'''get the split of train and test
	fold is the returned fold th data, from 0 to totalfold-1
	total fold is for the cross validation
	mydict is the return dict from readlabel
	sperate the data into train, validation, test'''
	skf = StratifiedKFold(n_splits=totalfold)  # default shuffle is false, okay!
	#readdicom(mydict)
	y = mydict.values()
	x = mydict.keys()
	count = 0
	valfold = (fold+1) % totalfold
	trainls, valls, testls = [], [], []
	for train, test in skf.split(x,y):
		if count == fold:
			testls = test[:]
		elif count == valfold:
			valls = test[:]
		else:
			for i in test:
				trainls.append(i)
		count += 1
	return trainls, valls, testls

def loadim(fname, aug=False, preprocesspath=preprocesspath):
	''' from preprocess path load fname
	fname file name in preprocesspath
	aug is true, we augment im fliplr, rot 4'''
	ims = []
	with open(join(preprocesspath, fname), 'rb') as inputfile:
		im = cPickle.load(inputfile)
		#up_bound = np.random.choice(174)                          #zero out square
		#right_bound = np.random.choice(174)
		img = im
		#img[up_bound:(up_bound+50), right_bound:(right_bound+50)] = 0.0
		ims.append(img)
		if aug:
			print('rotate')
			#ims.append(np.fliplr(im))
			#ims.append(np.rot90(im, 0))
			#ims.append(np.rot90(im, 1))
			#ims.append(np.rot90(im, 2))
			#ims.append(np.rot90(im, 3))
			for deg in xrange(15, 360, 15):
				#rotation_matrix = cv2.getRotationMatrix2D((224/2, 224/2), deg, 1)
				#img_rotation = cv2.warpAffine(im, rotation_matrix, (224, 224))
				imrot = scipy.ndimage.rotate(im,deg*1.)
				imrot = scipy.misc.imresize(imrot, (224, 224), interp='cubic')
				#up_bound = np.random.choice(194)                          #zero out square
				#right_bound = np.random.choice(194)
				#imrot[up_bound:(up_bound+30), right_bound:(right_bound+30)] = 0.0
				ims.append(imrot)
				#ims.append(img_rotation)
	return ims

def randomnoisedata(Xtr, noisesize=30):
	rXtr = Xtr
	for i in xrange(Xtr.shape[0]):
		up_bound = np.random.choice(224-noisesize)                          #zero out square
		right_bound = np.random.choice(224-noisesize)
		rXtr[i,0,up_bound:(up_bound+noisesize), right_bound:(right_bound+noisesize)] = 0.0
	return rXtr

def loaddata(fold, totalfold, usedream=True, aug=True):
	'''get the fold th train and  test data from inbreast
	fold is the returned fold th data, from 0 to totalfold-1
	total fold is for the cross validation'''
	mydict = readlabel()
	mydictkey = mydict.keys()
	mydictvalue = mydict.values()
	trainindex, testindex = cvsplit(fold, totalfold, mydict)
	if aug == True:
		traindata, trainlabel = np.zeros((6*len(trainindex),224,224)), np.zeros((6*len(trainindex),))
	else:
		traindata, trainlabel = np.zeros((len(trainindex),224,224)), np.zeros((len(trainindex),))
	testdata, testlabel =  np.zeros((len(testindex),224,224)), np.zeros((len(testindex),))
	traincount = 0
	for i in xrange(len(trainindex)):
		ims = loadim(mydictkey[trainindex[i]]+'.pickle', aug=aug)
		for im in ims:
			traindata[traincount, :, :] = im
			trainlabel[traincount] = mydictvalue[trainindex[i]]
			traincount += 1
	assert(traincount==traindata.shape[0])
	testcount = 0
	for i in xrange(len(testindex)):
		ims = loadim(mydictkey[testindex[i]]+'.pickle', aug=aug)
		testdata[testcount,:,:] = ims[0]
		testlabel[testcount] = mydictvalue[testindex[i]]
		testcount += 1
	assert(testcount==testdata.shape[0])
	if usedream:
		outx, outy = extractdreamdata()
		traindata = np.concatenate((traindata,outx), axis=0)
		trainlabel = np.concatenate((trainlabel,outy), axis=0)
	return traindata, trainlabel, testdata, testlabel

def loaddataenhance(fold, totalfold, usedream=True, loadtrain=False, aug=True):
	'''get the fold th train and  test data from inbreast
	fold is the returned fold th data, from 0 to totalfold-1
	total fold is for the cross validation'''
	mydict = readlabel()
	mydictkey = mydict.keys()
	mydictvalue = mydict.values()
	trainindex, valindex, testindex = cvsplitenhance(fold, totalfold, mydict)
	traindata, trainlabel = np.zeros((24*len(trainindex),224,224)), np.zeros((24*len(trainindex),))
	if aug == False:
		traindata, trainlabel = np.zeros((len(trainindex),224,224)), np.zeros((len(trainindex),))
	valdata, vallabel =  np.zeros((len(valindex),224,224)), np.zeros((len(valindex),))
	testdata, testlabel =  np.zeros((len(testindex),224,224)), np.zeros((len(testindex),))
	traincount = 0
	for i in xrange(len(trainindex)):
		ims = loadim(mydictkey[trainindex[i]]+'.pickle', aug=aug)
		for im in ims:
			traindata[traincount, :, :] = im
			trainlabel[traincount] = mydictvalue[trainindex[i]]
			traincount += 1
	assert(traincount==traindata.shape[0])
	valcount = 0
	if not loadtrain:
		for i in xrange(len(valindex)):
			ims = loadim(mydictkey[valindex[i]]+'.pickle')
			valdata[valcount,:,:] = ims[0]
			vallabel[valcount] = mydictvalue[valindex[i]]
			valcount += 1
		assert(valcount==valdata.shape[0])
	if not loadtrain:
		testcount = 0
		for i in xrange(len(testindex)):
			ims = loadim(mydictkey[testindex[i]]+'.pickle')
			testdata[testcount,:,:] = ims[0]
			testlabel[testcount] = mydictvalue[testindex[i]]
			testcount += 1
		assert(testcount==testdata.shape[0])
	#print(valdata.shape)
	randindex = np.random.permutation(valdata.shape[0])
	valdata = valdata[randindex,:,:]
	vallabel = vallabel[randindex]
	#print(valdata.shape)
	traindata = np.concatenate((traindata, valdata[60:,:,:]), axis=0)
	trainlabel = np.concatenate((trainlabel, vallabel[60:]), axis=0)
	valdata = valdata[:60,:,:]
	vallabel = vallabel[:60]
	maxvalue = (traindata.max()*1.0)
	traindata = traindata / maxvalue
	valdata = valdata / maxvalue
	testdata = testdata / maxvalue
	print('train dara feature')
	print(traindata.mean(), traindata.std(), traindata.max(), traindata.min())
	print('val data feature')
	print(valdata.mean(), valdata.std(), valdata.max(), valdata.min())
	print('test data feature')
	print(testdata.mean(), testdata.std(), testdata.max(), testdata.min())
	#meandata = traindata.mean()
	#stddata = traindata.std()
	#traindata = traindata - meandata
	#traindata = traindata / stddata
	#valdata = valdata - meandata
	#valdata = valdata / stddata
	#testdata = testdata - meandata
	#testdata = testdata / stddata
	if usedream:
		outx, outy = extractdreamdata(aug=aug)
		print('dream data feature')
		print(outx.mean(), outx.std(), outx.max(), outx.min())
		traindata = np.concatenate((traindata,outx), axis=0)
		trainlabel = np.concatenate((trainlabel,outy), axis=0)
	return traindata, trainlabel, valdata, vallabel, testdata, testlabel

### Outside data source from dream challenge
dreamtrainpath = '../trainData/'
dreamsavepth = '../Data/preprocessedData/'
dreamcsv1 = '../metaData/images_crosswalk.tsv'
dreamcsv2 = '../metaData/exams_metadata.tsv'

def extractdreamdata(aug=True):
	"""
	Goes through data folder and divides train/val.
	There should be two csv files.  The first will relate the filename
	to the actual patient ID and L/R side, then the second csv file
	will relate this to whether we get the cancer.  This is ridiculous.
	Very very very bad filesystem.  Hope this gets better.
	"""
	# First, let's map the .dcm.gz file to a (patientID, examIndex, imageView) tuple.
	path_csv_crosswalk, path_csv_metadata = dreamcsv1, dreamcsv2
	dict_img_to_patside, counter = {}, 0
	#checklabel = {}
	with open(path_csv_crosswalk, 'r') as file_crosswalk:
		reader_crosswalk = csv.reader(file_crosswalk, delimiter='\t')[1:]
		for row in reader_crosswalk:
			dict_img_to_patside[row[5].strip()] = (row[0].strip(), row[4].strip())
			# checklabel[row[5].strip()] = int(row[6])
			# Now, let's map the tuple to cancer or non-cancer.
	dict_tuple_to_cancer, counter = {}, 0
	with open(path_csv_metadata, 'r') as file_metadata:
		reader_metadata = csv.reader(file_metadata, delimiter='\t')[1:]
		for row in reader_metadata:
			if counter == 0:
				counter += 1
				continue
			#print(row[0].strip(), row[1], row[2], row[3], row[4])
			if row[3] == '0' or row[3] == '1':
				dict_tuple_to_cancer[(row[0].strip(), 'L')] = int(row[3])
			if row[4] == '0' or row[4] == '1':
				dict_tuple_to_cancer[(row[0].strip(), 'R')] = int(row[4])
	# Alright, now, let's connect those dictionaries together...
	X_tot, Y_tot = [], []
	for img_name in dict_img_to_patside:
		X_tot.append(img_name)
		assert(dict_tuple_to_cancer[dict_img_to_patside[img_name]] ==0 or 
		dict_tuple_to_cancer[dict_img_to_patside[img_name]] == 1)
		Y_tot.append(dict_tuple_to_cancer[dict_img_to_patside[img_name]])
		# if checklabel[img_name] != dict_tuple_to_cancer[dict_img_to_patside[img_name]]:
		# 	print img_name, checklabel[img_name], dict_tuple_to_cancer[dict_img_to_patside[img_name]]
	#Making train/val split and returning.
	#X_tr, X_te, Y_tr, Y_te = train_test_split(X_tot, Y_tot, test_size=0.2, random_state=1) # 0.2
	if aug == True:
		trainx, trainy = np.zeros((24*len(X_tot), 224, 224)), np.zeros((24*len(Y_tot),))
	else:
		trainx, trainy = np.zeros((len(X_tot), 224, 224)), np.zeros((len(Y_tot),))
	traincount = 0
	for i in xrange(len(Y_tot)):
		ims = loadim(X_tot[i][:-4]+'resizenorm.pickle', aug=aug, preprocesspath=dreampreprocesspath)
		for im in ims:
			trainx[traincount,:,:] = im 
			trainy[traincount] = Y_tot[i]
			traincount += 1
	if aug == True:
		assert(traincount==len(X_tot)*24)
	else: assert(traincount==len(X_tot))
	#meanim = trainx.mean(axis=0)
	trainx = trainx / (trainx.max()*1.0)
	#stdim = trainx
	return trainx, trainy #dict_img_to_patside

if __name__ == '__main__':
	readdicom("../trainingData/")
	# traindata, trainlabel, testdata, testlabel = loaddata(0, 5)
	# print(sum(trainlabel), sum(testlabel))

	# traindata, trainlabel, valdata, vallabel, testdata, testlabel = loaddataenhance(0, 5)
	# print(sum(trainlabel), sum(vallabel), sum(testlabel))
