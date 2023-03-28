import pickle
from pyspark.ml.linalg import SparseVector, VectorUDT
def tokenizerText(text):
    return list(text)
class SearchText:
	def __init__(self, optionSearch='',pathModelCharacter='',pathModelAuthor='',pathModelSeries='',pathDictKeyWord=''):
		if optionSearch == '':
			self.optionSearch = 'tfidf'
		else:
			self.optionSearch = optionSearch
		if self.optionSearch == 'tfidf':
			if pathDictKeyWord == '':
				self.pathDictKeyWord = 'modelTFIDF/modelGenFeatureTfidf.pickle'
			else:
				self.pathDictKeyWord = pathDictKeyWord
			if pathModelCharacter == '':
				self.pathModelCharacter = 'modelTFIDF/searchCharacter.pickle'
			else:
				self.pathModelCharacter = pathModelCharacter
			if pathModelAuthor == '':
				self.pathModelAuthor = 'modelTFIDF/searchAuthor.pickle'
			else:
				self.pathModelAuthor = pathModelAuthor
			if pathModelSeries == '':
				self.pathModelSeries = 'modelTFIDF/searchSeries.pickle'
			else:
				self.pathModelSeries = pathModelSeries
		elif self.optionSearch == 'parse':
			if pathDictKeyWord == '':
				self.pathDictKeyWord = 'ModelParse/dict_keyword_EN.pickle'
			else:
				self.pathDictKeyWord = pathDictKeyWord
			if pathModelCharacter == '':
				self.pathModelCharacter = 'ModelParse/searchCharacter_EN.pickle'
			else:
				self.pathModelCharacter = pathModelCharacter
			if pathModelAuthor == '':
				self.pathModelAuthor = 'ModelParse/searchAuthor_EN.pickle'
			else:
				self.pathModelAuthor = pathModelAuthor
			if pathModelSeries == '':
				self.pathModelSeries = 'ModelParse/searchSeries_EN.pickle'
			else:
				self.pathModelSeries = pathModelSeries
		else:
			raise Exception("Option search is tfidf/parse")
		self.modelSearchAuthor = pickle.load(open(self.pathModelAuthor, 'rb'))
		self.modelSearchCharacter = pickle.load(open(self.pathModelCharacter, 'rb'))
		self.modelSearchSeries = pickle.load(open(self.pathModelSeries, 'rb'))
		self.dict_keyword = pickle.load(open(self.pathDictKeyWord, 'rb'))
	def search(self,listText,modelSearch,topk=5):
		listText = [s.lower() for s in listText]
		if self.optionSearch == 'tfidf':
			search_features_vec = self.dict_keyword.transform(listText)
			rs = modelSearch.search(search_features_vec, k=topk, return_distance=True)
		else:
			listVec = []
			for i in listText:
				testVector = self.genfeaturesSparseTest(i.lower(),self.dict_keyword)
				listVec.append(testVector)
			rs = modelSearch.search(listVec, k=5, return_distance=True)
		return rs
	def searchCharacter(self,listText,topk=5):
		return self.search(listText, self.modelSearchCharacter,topk)
	def searchAuthor(self,listText,topk=5):
		return self.search(listText,self.modelSearchAuthor,topk)
	def searchSeries(self,listText,topk=5):
		return self.search(listText,self.modelSearchSeries,topk)
	def genfeaturesSparseTest(self, words , dict_keyword ):
		vocabSize = len(dict_keyword.keys())
		dictVec = {}
		for w in words:
			if w not in dict_keyword.keys():
				continue
			idx = dict_keyword[w]
			if idx not in dictVec.keys():
				dictVec[idx] = 1
			else:
				dictVec[idx] += 1
		sortedVec = sorted(dictVec.keys())
		sortedValue = [dictVec[key] for key in sortedVec]
		return SparseVector(vocabSize,list(sortedVec),list(sortedValue)) 