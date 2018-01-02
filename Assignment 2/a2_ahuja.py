import zipfile
from tifffile import TiffFile
import io
import numpy as np
# import scipy.stats as ss
# from PIL import Image #this is an image reading library
from numpy.linalg import svd
import hashlib
from pyspark import SparkContext, SparkConf
from math import floor, sqrt

# Read image from zip file
def getOrthoTif(zfBytes):
 #given a zipfile as bytes (i.e. from reading from a binary file),
 # return a np array of rgbx values for each pixel
 bytesio = io.BytesIO(zfBytes)
 zfiles = zipfile.ZipFile(bytesio, "r")
 #find tif:
 from tifffile import TiffFile
 for fn in zfiles.namelist():
  if fn[-4:] == '.tif':#found it, turn into array:
   tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
   return tif.asarray()
 return None

# from absolute path of a file, find out file name
def getFileName(path):
    tokens = path.split('/')
    return tokens[-1]

# Divide a square shaped image into pieces of 500x500
def divideImage(kv):
    filename = kv[0]
    img = kv[1]
    r,c,ch = img.shape
    rp, cp = floor(r/500), floor(c/500)
    parts = []
    for i in range(rp):
        for j in range(cp):
            partNumber = filename + '-' + str(cp*i + j);
            partImg = img[i*500:(i+1)*500, j*500:(j+1)*500,:]
            #print('Rows:',i*500,' ', (i+1)*500, 'Cols', j*500, ' ', (j+1)*500)
            parts.append((partNumber, partImg))
    return parts

# Convert rgbi image (4channel) to single channel
def convert4dImageToFeature(img):
    rgbSum = img[:,:,0].astype(np.float32) + img[:,:,1].astype(np.float32) + img[:,:,2].astype(np.float32)
    feature = np.multiply((1.0/3.0)*rgbSum, 0.01*(img[:,:,3].astype(np.float32)))
    return feature

# Perform averaging over squares of 10x10
def averagingOver10X10Windows(img):
    r, c = img.shape
    rsub, csub = int(r/10), int(c/10)
    imgSub = np.zeros((rsub, csub), dtype=np.float32)
    for i in range(rsub):
        for j in range(csub):
            imgSub[i,j] = np.sum(img[i*10:(i+1)*10,j*10:(j+1)*10])
            #print('Rows:',i*10,' ', (i+1)*10, 'Cols', j*10, ' ', (j+1)*10)
    return imgSub/100.0

# Perform averaging over squares of 5x5
def averagingOver5X5Windows(img):
    r, c = img.shape
    rsub, csub = int(r/5), int(c/5)
    imgSub = np.zeros((rsub, csub), dtype=np.float32)
    for i in range(rsub):
        for j in range(csub):
            imgSub[i,j] = np.sum(img[i*5:(i+1)*5,j*5:(j+1)*5])
            #print('Rows:',i*10,' ', (i+1)*10, 'Cols', j*10, ' ', (j+1)*10)
    return imgSub/25.0

# Compute feature vector of a single channel image
# Compute geadient in x and y directions and quantize it
# Then flatten the gradient matrices and concatenate
def computeFeatureVector(img):
    rowDiff = np.diff(img)
    colDiff = np.diff(img, axis = 0)
    lowTh = -1
    highTh = 1
    rowDiffTh = -1*((rowDiff < lowTh).astype(np.int32)) + (rowDiff > highTh).astype(np.int32)
    colDiffTh = -1*((colDiff < lowTh).astype(np.int32)) + (colDiff > highTh).astype(np.int32)
    rowDiffFeatureVec = rowDiffTh.flatten()
    colDiffFeatureVec = colDiffTh.flatten()
    feature = np.concatenate((rowDiffFeatureVec, colDiffFeatureVec))
    return feature

# Not used in updated assignment description
def computemd5Hash(featureVector):
    hashObj = hashlib.md5()
    hashObj.update(featureVector)
    hashCode = hashObj.hexdigest()
    return hashCode

# Compute signature of an image expressed as feature vector (composed of {-1,0,1})
# Divide the feature vector into similarly sized 128 chunks, find md5 hash of each chunk
# Take a character of md5 digest of each chunk to form 128 char long signature
def computeSignature(featureVector, idx=0):
    hashCode = ''
    n = len(featureVector)
    numPieces = 128
    sIdx = 0
    step = floor(n/numPieces)
    eIdx = step
    iteration = 1
    mod = n%128
    while(eIdx <= len(featureVector)):
        if(iteration <= mod):
            eIdx += 1
        hashObj = hashlib.md5()
        #print(sIdx,eIdx)
        hashObj.update(featureVector[sIdx:eIdx])
        hashCodeCurr = hashObj.hexdigest()
        #hashCode += format(int(hashCodeCurr[0],16),'b').zfill(4)[0]
        hashCode += hashCodeCurr[idx]
        sIdx = eIdx
        eIdx = eIdx + step
        iteration += 1
    return hashCode

# Not used. Instead, computeBuckets2 is used
def computeBuckets(hashCode, bands=None):
    if(bands == None):
        bands = 8
    bandwidth = floor(len(hashCode)/bands)
    buckets = []
    for i in range(0,bands):
        substr = hashCode[i*bandwidth:(i+1)*bandwidth]
        buckets.append(int(substr, 16))
    return np.array(buckets)

# Divides a signature into bands of size bandwidth
# For rows in each band, compute a hashcode. The hashcode is
# decimal representation of string formed by characters in a row:
# For ex, if a row is 102a, its hashcode is 10 + 2*2 + 0*4 + 1*8 = 22
# This hashfunction is unique for every unique key, which is very good for LSH
# Output of this function is bucket number (i.e. hashcode) for every band in the signature.
# Bandwidth parameter was tweaked to get 15-25 candidates for every query image
def computeBuckets2(hashCode, bandwidth=None):
    if(bandwidth == None):
        bandwidth = 1
    hashCode = format(int(hashCode,16),'b').zfill(4*128)
    bands = floor(len(hashCode)/bandwidth)
    buckets = []
    for i in range(0,bands):
        substr = hashCode[i*bandwidth:(i+1)*bandwidth]
        buckets.append(int(substr, 16))
    return np.array(buckets)

# Not used in updated assignment description
def computeSimilarity(hashCode, queryBuckets):
    n = len(queryBuckets)
    isSimilar = np.zeros((n,), dtype=np.bool)
    for i in range(n):
        # Bool can be assigned an int!!
        isSimilar[i] = np.sum(hashCode == queryBuckets[i][1])
    return isSimilar

# Given bucket information for all the bands of query images,
# check with buckets an image in rdd to see if this image is
# similar to any qurey image in ANY band!
# If yes, yield a tuple (query image, current image)
def computeSimilarCandidates(inputTup, queryBuckets):
    n = len(queryBuckets)
    candidates = []
    for i in range(n):
        if((np.sum(inputTup[1] == queryBuckets[i][1]) > 0)):
            candidates.append((queryBuckets[i][0], inputTup[0]))
    return candidates

# Not used in updated assignment description
def isCandidate(hashCode1, hashCode2, bands=None):
    #hashCode1 = bin(int(hashCode1, 16))
    hashCode1 = format(int(hashCode1, 16),'b').zfill(128)
    hashCode2 = format(int(hashCode2, 16),'b').zfill(128)
    if(bands == None):
        bands = 8
    bandwidth = floor(len(hashCode1)/bands)
    for i in range(0,bands):
        if(hashCode1[i*bandwidth:(i+1)*bandwidth] == hashCode2[i*bandwidth:(i+1)*bandwidth]):
            return True
    return False

# Not used in updated assignment description
def isCandidate2(hashCode1, hashCode2, bands=None):
    if(bands == None):
        bands = 8
    bandwidth = floor(len(hashCode1)/bands)
    for i in range(0,bands):
        if(hashCode1[i*bandwidth:(i+1)*bandwidth] == hashCode2[i*bandwidth:(i+1)*bandwidth]):
            return True
    return False

# Used for debugging
def findFeature(filenamesBC, rdd):
    rddFiltered = rdd.filter(lambda x: x[0] in filenamesBC.value).collect()
    print(len(rddFiltered))
    features = []
    for kv in rddFiltered:
        features.append(kv)
    return features

# Create a sparse random projection matrix.
# Projection to this matrix will preserve distaces between original images
# Ref paper: https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
def findRandomProjectionMatrix(dim):
    s = sqrt(dim)
    randMat = np.random.uniform(size=(dim, floor(s)))
    th1 = 1.0/(2.0*s)
    th2 = th1 + 1.0 - 1.0/(1.0*s)
    projMat = -1*((randMat < th1).astype(np.int32)) + 1*((randMat > th2).astype(np.int32))
    return projMat

# Sequence of operations performed for completing the assignemnt
def assignmentQuestions(rdd4):
    
    rdd5 = rdd4.map(lambda x: (x[0], averagingOver10X10Windows(x[1])))
    rdd6 = rdd5.map(lambda x: (x[0], computeFeatureVector(x[1])))
    # rdd6 will also be used later, so call persist
    rdd6.persist()
    # For part 2(f)
    filenames2f = ['3677454_2025195.zip-1', '3677454_2025195.zip-18']
    rddFiltered2f = rdd6.filter(lambda x: x[0] in filenames2f).collect()
    print('Output 2f:')
    for entry in rddFiltered2f:
        print(entry[0], entry[1])
    # Part 2 complete

    # Part 3 start
    idx = 1
    # Idx chosen as 1 after experiments with other values.
    rdd7 = rdd6.map(lambda x: (x[0], computeSignature(x[1], idx)))
    filenames3 = ['3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19']
    bandwidth = 13
    # bandwidth chosen as 13 after experiments with other values.
    rdd8 = rdd7.map(lambda x: (x[0], computeBuckets2(x[1], bandwidth)))
    queryBuckets = rdd8.filter(lambda x: x[0] in filenames3).collect()
    rdd9 = rdd8.flatMap(lambda x: (computeSimilarCandidates(x, queryBuckets)))
    rdd10 = rdd9.groupByKey().map(lambda x: (x[0], list(x[1])))
    similarItems = rdd10.collect()
    print('Number of similarity candidates:')
    for v in similarItems:
        print(v[0], len(v[1]))
    
    # Part 3(c)
    # Discussed this approach with Prof. Schwartz
    # He approved it. During our discussion, he suggested to boradcast the
    # projection matrix, so that every row of original matrix (distributed in rdd) can be multiplied with
    # every column of projection matrix
    # Ref paper: https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
    # Sparse random projections preserve distance between data points
    # Used PCA after random projections
    projectionMat = findRandomProjectionMatrix(4900)
    projectionMatBC = sc.broadcast(projectionMat)
    # Project original features to 70 dimensions using sparse random projection matrix
    rdd11 = rdd6.map(lambda x: (x[0], np.matmul(x[1], projectionMatBC.value)))
    # Preparing for PCA
    # Standardize the data to have 0 mean and unit variance
    rdd11mean = rdd11.map(lambda x: (0, x[1])).reduceByKey(lambda a,b: a+b).collect()
    meanReducedFeature = np.divide(rdd11mean[0][1], rdd11.count())
    rdd12 = rdd11.map(lambda x: (x[0], x[1] - meanReducedFeature)) # stores x-mu
    # Find variance of each feature
    varianceReducedFeature = rdd12.map(lambda x: (0, np.square(x[1]))).reduceByKey(lambda a,b: a+b).collect()
    varianceReducedFeature = np.divide(varianceReducedFeature[0][1], rdd11.count())
    sigmaReducedFeature = np.sqrt(varianceReducedFeature)
    # Standardize: divide (x-mu) by sigma
    rdd13 = rdd12.map(lambda x: (x[0], np.divide(x[1], sigmaReducedFeature))) # stores standardized random projections
    # x = rdd13 (4000x70). compute xt*x
    # xt*x is small matrix! (70x70)
    rdd14 = rdd13.map(lambda x: np.reshape(x[1],(70, 1))).map(lambda x: np.matmul(x, x.T)).reduce(lambda a,b: a+b)
    # covMat is 70x70, can be easily stored on a single machine
    covMatrix = rdd14
    # SVD and Eigen decomposition of square symmetric matrix yield same result
    # Eigenvectors of covariance matrix of X are principal component vectors of X
    u, s, v = svd(covMatrix)
    # Confirmed that u*v is nearly identity matrix
    # Get top 10 eigenvectors
    evProj = u[:,0:10]
    evProjBC = sc.broadcast(evProj)
    # Fetch feature vectors for query images and their similarity candidates
    vectorsToFetch = dict()
    for item in similarItems:
        vectorsToFetch[item[0]] = True
        for candidate in item[1]:
            vectorsToFetch[candidate] = True


    queryVectors = rdd13.filter(lambda x: x[0] in vectorsToFetch)
    # Project query images and similarity candidates onto eigenvectors
    queryVectorsProj = queryVectors.map(lambda x: (x[0], np.matmul(x[1], evProjBC.value))).collect()
    queryVectorsDict = dict()
    queryVectorsProjDict = dict()
    for vec in queryVectorsProj:
        queryVectorsProjDict[vec[0]] = vec[1]
    for vec in queryVectors.collect():
        queryVectorsDict[vec[0]] = vec[1]
    filenames3c = ['3677454_2025195.zip-1', '3677454_2025195.zip-18']
    dist = np.linalg.norm(queryVectorsProjDict[filenames3c[0]] - queryVectorsProjDict[filenames3c[1]])
    distBeforeProj = np.linalg.norm(queryVectorsDict[filenames3c[0]] - queryVectorsDict[filenames3c[1]])
    print('Distance between %s and %s after projection is %f' %(filenames3c[0], filenames3c[1], dist))
    distDict = dict()
    distDictBeforeProj = dict()
    # Find distance between query images and their similarity candidates
    for item in similarItems:
        distDict[item[0]] = []
        distDictBeforeProj[item[0]] = []
        for candidate in item[1]:
            distDict[item[0]].append((candidate, np.linalg.norm(queryVectorsProjDict[item[0]] - queryVectorsProjDict[candidate])))
            distDictBeforeProj[item[0]].append((candidate, np.linalg.norm(queryVectorsDict[item[0]] - queryVectorsDict[candidate])))

    for item in distDict:
        distDict[item].sort(key = lambda x: x[1])
        distDictBeforeProj[item].sort(key = lambda x: x[1])

    for item in filenames3c:
        print('Distances of similarity candidates of %s after projection:'% (item))
        for item2 in distDict[item]:
            print(item2)

# Sequence of operations performed for bonus part of assignemnt
def bonusQuestion(rdd4):
    print('Bonus question: using scale factor of 5')
    rdd5 = rdd4.map(lambda x: (x[0], averagingOver5X5Windows(x[1])))
    rdd6 = rdd5.map(lambda x: (x[0], computeFeatureVector(x[1])))
    # rdd6 will also be used later, so call persist
    rdd6.persist()  

    # Part 3 start
    idx = 1
    rdd7 = rdd6.map(lambda x: (x[0], computeSignature(x[1], idx)))
    filenames3 = ['3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19']
    bandwidth = 13
    rdd8 = rdd7.map(lambda x: (x[0], computeBuckets2(x[1], bandwidth)))
    queryBuckets = rdd8.filter(lambda x: x[0] in filenames3).collect()
    rdd9 = rdd8.flatMap(lambda x: (computeSimilarCandidates(x, queryBuckets)))
    rdd10 = rdd9.groupByKey().map(lambda x: (x[0], list(x[1])))
    similarItems = rdd10.collect()
    print('Number of similarity candidates:')
    for v in similarItems:
        print(v[0], len(v[1]))
    
    # Part 3(c)
    # Discussed this approach with Prof. Schwartz
    # He approved it. During our discussion, he suggested to boradcast the
    # projection matrix, so that every row of original matrix can be multiplied with
    # every column of projection matrix
    # This time, feature vectos is of size 19800
    projectionMat = findRandomProjectionMatrix(19800)
    projectionMatBC = sc.broadcast(projectionMat)
    rdd11 = rdd6.map(lambda x: (x[0], np.matmul(x[1], projectionMatBC.value)))
    rdd11mean = rdd11.map(lambda x: (0, x[1])).reduceByKey(lambda a,b: a+b).collect()
    meanReducedFeature = np.divide(rdd11mean[0][1], rdd11.count())
    rdd12 = rdd11.map(lambda x: (x[0], x[1] - meanReducedFeature)) # stores x-mu
    varianceReducedFeature = rdd12.map(lambda x: (0, np.square(x[1]))).reduceByKey(lambda a,b: a+b).collect()
    varianceReducedFeature = np.divide(varianceReducedFeature[0][1], rdd11.count())
    sigmaReducedFeature = np.sqrt(varianceReducedFeature)
    rdd13 = rdd12.map(lambda x: (x[0], np.divide(x[1], sigmaReducedFeature))) # stores standardized random projections
    # x = rdd13 (4000x70). compute xt*x
    rdd14 = rdd13.map(lambda x: np.reshape(x[1],(140, 1))).map(lambda x: np.matmul(x, x.T)).reduce(lambda a,b: a+b)
    # covMat is 140x140, can be easily stored on a single machine
    covMatrix = rdd14
    # SVD and Eigen decomposition of square symmetric matrix yield same result
    # Eigenvectors of covariance matrix of X are principal component vectors of X
    u, s, v = svd(covMatrix)
    # Get top 10 eigenvectors
    evProj = u[:,0:10]
    evProjBC = sc.broadcast(evProj)
    # Fetch feature vectors for query images and their similarity candidates
    vectorsToFetch = dict()
    for item in similarItems:
        vectorsToFetch[item[0]] = True
        for candidate in item[1]:
            vectorsToFetch[candidate] = True


    queryVectors = rdd13.filter(lambda x: x[0] in vectorsToFetch)
    queryVectorsProj = queryVectors.map(lambda x: (x[0], np.matmul(x[1], evProjBC.value))).collect()
    queryVectorsDict = dict()
    queryVectorsProjDict = dict()
    for vec in queryVectorsProj:
        queryVectorsProjDict[vec[0]] = vec[1]
    for vec in queryVectors.collect():
        queryVectorsDict[vec[0]] = vec[1]
    filenames3c = ['3677454_2025195.zip-1', '3677454_2025195.zip-18']
    dist = np.linalg.norm(queryVectorsProjDict[filenames3c[0]] - queryVectorsProjDict[filenames3c[1]])
    distBeforeProj = np.linalg.norm(queryVectorsDict[filenames3c[0]] - queryVectorsDict[filenames3c[1]])
    print('Distance between %s and %s after projection is %f' %(filenames3c[0], filenames3c[1], dist))
    distDict = dict()
    distDictBeforeProj = dict()

    for item in similarItems:
        distDict[item[0]] = []
        distDictBeforeProj[item[0]] = []
        for candidate in item[1]:
            distDict[item[0]].append((candidate, np.linalg.norm(queryVectorsProjDict[item[0]] - queryVectorsProjDict[candidate])))
            distDictBeforeProj[item[0]].append((candidate, np.linalg.norm(queryVectorsDict[item[0]] - queryVectorsDict[candidate])))

    for item in distDict:
        distDict[item].sort(key = lambda x: x[1])
        distDictBeforeProj[item].sort(key = lambda x: x[1])

    for item in filenames3c:
        print('Distances of similarity candidates of %s after projection:'% (item))
        for item2 in distDict[item]:
            print(item2)




if __name__ == '__main__':
    sc = SparkContext.getOrCreate()
    #datasetDir = 'D:\Acads\CSE545\Assignment 2\data\small_sample'
    # Part 1 start
    datasetDir = 'hdfs:/data/large_sample'
    rdd = sc.binaryFiles(datasetDir)
    #rdd.map(lambda x: x[0]).collect()
    rdd2 = rdd.map(lambda x: (getFileName(x[0]), getOrthoTif(x[1])))
    rdd3 = rdd2.flatMap(divideImage)
    # For part 1(e)
    filenames1e = ['3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19']
    rddFiltered1e = rdd3.filter(lambda x: x[0] in filenames1e).collect()
    print('Output 1e:')
    for entry in rddFiltered1e:
        print(entry[0], entry[1][0,0,:])

    # Part 1 complete

    # Part 2 start
    rdd4 = rdd3.map(lambda x: (x[0], convert4dImageToFeature(x[1])))
    rdd4.persist()
    assignmentQuestions(rdd4)
    bonusQuestion(rdd4)