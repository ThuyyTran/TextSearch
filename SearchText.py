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
				pathDictKeyWord = '/media/anlab/data-2tb/ANLAB_THUY/SearchText/modelTFIDF/modelGenFeatureTfidf.pickle'
			if pathModelCharacter == '':
				pathModelCharacter = '/media/anlab/data-2tb/ANLAB_THUY/SearchText/modelTFIDF/searchCharacter.pickle'
			if pathModelAuthor == '':
				pathModelAuthor = '/media/anlab/data-2tb/ANLAB_THUY/SearchText/modelTFIDF/searchAuthor.pickle'
			if pathModelSeries == '':
				pathModelSeries = '/media/anlab/data-2tb/ANLAB_THUY/SearchText/modelTFIDF/searchSeries.pickle'
		elif self.optionSearch == 'parse':
			if pathDictKeyWord == '':
				pathDictKeyWord = '/media/anlab/data-2tb/ANLAB_THUY/SearchText/ModelParse/dict_keyword_EN.pickle'
			if pathModelCharacter == '':
				pathModelCharacter = '/media/anlab/data-2tb/ANLAB_THUY/SearchText/ModelParse/searchCharacter_EN.pickle'
			if pathModelAuthor == '':
				pathModelAuthor = '/media/anlab/data-2tb/ANLAB_THUY/SearchText/ModelParse/searchAuthor_EN.pickle'
			if pathModelSeries == '':
				pathModelSeries = '/media/anlab/data-2tb/ANLAB_THUY/SearchText/ModelParse/searchSeries_EN.pickle'
		else:
			raise Exception("Option search is tfidf/parse")
		self.modelSearchAuthor = pickle.load(open(pathModelAuthor, 'rb'))
		self.modelSearchCharacter = pickle.load(open(pathModelCharacter, 'rb'))
		self.modelSearchSeries = pickle.load(open(pathModelSeries, 'rb'))
		self.dict_keyword = pickle.load(open(pathDictKeyWord, 'rb'))
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
		return self.search(self.modelSearchSeries,listText,topk)
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