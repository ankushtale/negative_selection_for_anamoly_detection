import random
import csv


def loadCSVData(filename):

	max_val = -10
	min_val = 10

	with open(filename) as fp:
		read_csv = csv.reader(fp)
		next(read_csv)
		data = list()

		for row in read_csv:
			f_row = list(map(float, row))
			max_val = max(max(f_row[1:]), max_val)
			min_val = min(min(f_row[1:]), min_val)
			data.append(f_row[1:])

	return max_val, min_val, data


def loadTextData(filename):
	"""
	text data loader, returns list of strings in data as well as alphabet
	:param filename:
	:return:
	"""
	with open(filename) as fp:
		fullString = fp.read()

	data = fullString.split("\n")
	for d in data:
		d.replace("\n", "")
	alphabet = set()
	for c in fullString:
		if not c == "\n":
			alphabet.add(c)

	return data, list(alphabet)


def check_for_split(data):

	range_list = {-1: 0, -0.7: 0, -0.5: 0, -0.4: 0,
				  -0.2: 0, 0: 0, 0.5: 0, 1: 0, 2:0 }

	for row in data:
		for r in row:
			if r <= -1:
				range_list[-1] += 1
			elif -1 < r <= -0.7:
				range_list[-0.7] += 1
			elif -0.7 < r <= -0.5:
				range_list[-0.5] += 1
			elif -0.5 < r <= -0.4:
				range_list[-0.4] += 1
			elif -0.4 < r <= -0.2:
				range_list[-0.2] += 1
			elif -0.2 < r <= 0:
				range_list[0] += 1
			elif 0 < r <= 0.5:
				range_list[0.5] += 1
			elif 0.5 < r <= 1:
				range_list[1] += 1
			elif 1 < r:
				range_list[2] += 1

	print(range_list)


def binned_data(data, n, max_val, min_val, fixed_categories=True):
	"""

	:param data:
	:param n:
	:param max_val:
	:param min_val:
	:return:
	"""

	if fixed_categories:
		splits = [-1, -0.7, -0.5, -0.4, -0.2, 0, 0.5, 1, 2]
		for i in range(len(data)):
			for j in range(len(data[i])):
				if data[i][j] <= splits[0]:
					data[i][j] = 'a'
				elif splits[0] < data[i][j] <= splits[1]:
					data[i][j] = 'b'
				elif splits[1] < data[i][j] <= splits[2]:
					data[i][j] = 'c'
				elif splits[2] < data[i][j] <= splits[3]:
					data[i][j] = 'd'
				elif splits[3] < data[i][j] <= splits[4]:
					data[i][j] = 'e'
				elif splits[4] < data[i][j] <= splits[5]:
					data[i][j] = 'f'
				elif splits[5] < data[i][j] <= splits[6]:
					data[i][j] = 'g'
				elif splits[6] < data[i][j] <= splits[7]:
					data[i][j] = 'h'
				elif splits[7] < data[i][j] <= splits[8]:
					data[i][j] = 'i'
				elif splits[8] < data[i][j]:
					data[i][j] = 'j'
			data[i] = ''.join(data[i])
	return data


class rchunkDetector:

	def __init__(self, r, l, A):
		"""
		r: length of chunk for matching
		l: length of strings in training set
		A: alphabet to generate detector string
		"""
		self.r = r
		self.i = random.randint(0, l-r-1)
		self.l = l
		self.detectorString = ""
		for i in range(r):
			self.detectorString += A[random.randint(0, len(A)-1)]

	def testDetector(self, s):
		if s[self.i:self.i + self.r] == self.detectorString:
			return True
		return False


class rcontigDetector:
	def __init__(self, r, l, A):
		"""
		:param r: length of substring for matching
		:param l: length of strings in training set
		:param A: alphabet to generate detector string
		"""
		self.r = r
		self.l = l
		self.detectorString = ""
		for i in range(l):
			self.detectorString += A[random.randint(0,len(A)-1)]

	def testDetector(self, s):
		"""
		tests if the detector string matches s
		:param s:
		:return:
		"""
		for i in range(self.l-self.r+1):
			if self.detectorString[i:self.r+i] in s:
				return True
		return False


def trainRChunk(T, A, n, k, r, l, unique=True):
	"""
	training (r-chunk)
	trains a population of detectors
	input T: A list of training strings
	input A: an alphabet with which to construct detectors
	input n: The number of detectors in resulting population
	input k: The number of strings to sample from T for training
	input r: The length of the chunk for matching
	input l: The length of strings in the training set
	"""
	detectors = set()
	Tlist = T.copy()

	while len(detectors) < n:
		td = rchunkDetector(r, l, A)
		if unique:
			while td in detectors:
				td = rchunkDetector(r, l, A)
		random.shuffle(Tlist)
		goodDetector = True
		for i in range(k):
			if td.testDetector(Tlist[i]):
				goodDetector = False
				break
		if goodDetector:
			detectors.add(td)

	return list(detectors)


def trainRContig(T, A, n, k, r, l, unique=True):
	"""
	training (r-contiguous)
	trains a population of detectors
	:param T: A list of training strings
	:param A: an alphabet with which to construct detectors
	:param n: The number of detectors in resulting population
	:param k: The number of strings to sample from T for training
	:param r: The length of the chunk for matching
	:param l: The length of strings in the training set
	:param unique:
	:return:
	"""
	detectors = set()

	while len(detectors) < n:
		td = rcontigDetector(r, l, A)
		if unique:
			while td in detectors:
				td = rcontigDetector(r, l, A)
		random.shuffle(T)
		goodDetector = True
		for i in range(k):
			if td.testDetector(T[i]):
				goodDetector = False
				break
		if goodDetector:
			detectors.add(td)

	return list(detectors)
