from tqdm import tqdm
from nltk.tokenize.util import is_cjk
import pandas as pd
import pickle
import time
from scipy.sparse import csr_matrix
import pysparnn.cluster_index as ci
import sys
import numpy as np
from pyspark.ml.linalg import SparseVector, VectorUDT
from sklearn.feature_extraction.text import TfidfVectorizer
def tokenizerText(text):
    return list(text)
def genDatabase():
	list_character = get_dictionary("CharacterMaster_20230317170824.xlsx", key="キャラクター名称")
	list_series = get_dictionary("SeriesMaster.xlsx", key="作品名称")
	list_author = get_dictionary("AuthorMaster_20230317170934.xlsx", key="作家名称")
	dict_keyword = {}
	updateKeyWord(list_character, dict_keyword)
	updateKeyWord(list_series, dict_keyword)
	updateKeyWord(list_author, dict_keyword)
	dataSparsecharacter = genfeaturesSparse(list_character , dict_keyword)
	dataSparseSeries = genfeaturesSparse(list_series , dict_keyword)
	dataSparseAuthor = genfeaturesSparse(list_author , dict_keyword)
	return dataSparsecharacter , list_character, dataSparseSeries , list_series, dataSparseAuthor, list_author , dict_keyword
def genDataTfidf():
	list_character = getListText("CharacterMaster_20230317170824.xlsx", key="キャラクター名称")
	list_series = getListText("SeriesMaster.xlsx", key="作品名称")
	list_author = getListText("AuthorMaster_20230317170934.xlsx", key="作家名称")
	tv = TfidfVectorizer(tokenizer=tokenizerText)
	tv.fit(list(list_character.keys())+list(list_series.keys())+list(list_author.keys()))
	featuresVeccharacter = tv.transform(list_character.keys())
	featuresSeries = tv.transform(list_series.keys())
	featuresAuthor = tv.transform(list_author.keys())
	return featuresVeccharacter, list(list_character.values()), featuresSeries, list(list_series.values()), featuresAuthor, list(list_author.values()), tv
def getListText(path_data, key=None):
	listText = {}
	excel_data = pd.read_excel(path_data)
	data = pd.DataFrame(excel_data, columns=[key])
	data_size = len(data[key])
	correct = 0
	for i in tqdm(range(data_size)):
		name = str(data[key][i])
		if len(name) > 0 :
			name_search = name.lower().replace(" ", "")
			if len(name_search) == 1:
				correct += 1
			listText[name_search] = name
	print("data " , correct)
	return listText
def get_dictionary(path_data, key=None):
	list_dictionary = {}
	if key is None:
		return list_dictionary
	excel_data = pd.read_excel(path_data)
	# Read the values of the file in the dataframe
	data = pd.DataFrame(excel_data, columns=[key])
	
	data_size = len(data[key])
	correct = 0
	for i in tqdm(range(data_size)):
		name = str(data[key][i])
		# print("name" , name)
		if len(name) > 0 :
			name_search = name.lower().replace(" ", "_")
			if len(name_search) == 1:
				correct += 1
			list_dictionary[name_search] = name
	print("data " , correct)
	return list_dictionary
def updateKeyWord(list_character, dict_keyword):
	count_key = 0
	for key in list_character.keys():
		words = list(key)
		for w in words:
			# if not(is_cjk(w)):
			# 	continue
			if w not in dict_keyword.keys():
				dict_keyword[w] = count_key
				count_key += 1 
			
	return dict_keyword
def genfeaturesSparse(list_character , dict_keyword):
	datas = list(list_character.keys())
	size_data = len(datas)
	vocabSize = len(dict_keyword.keys())
	listVector = []
	for key in datas:
		dictVec = {}
		words = list(key)
		for w in words:
			# if not(is_cjk(w)):
			# 	continue
			idx = dict_keyword[w]
			if idx not in dictVec.keys():
				dictVec[idx] = 1
			else:
				dictVec[idx]+=1
		sortedVec = sorted(dictVec.keys())
		sortedValue = [dictVec[key] for key in sortedVec]
		listVector.append(SparseVector(vocabSize,list(sortedVec),list(sortedValue)))
	return listVector
def checkRam(val):
	total_size = sum(sys.getsizeof(d) for d in val)
	total_size_gb = total_size / (1024 ** 3)
	print("Total size of the dictionaries in memory:", total_size_gb, "GB")
def searchTFIDF(listText,modelSearch,modelGenFeature,topk=5):
	search_features_vec = modelGenFeature.transform(listText)
	rs = modelSearch.search(search_features_vec, k=topk, return_distance=True)
	for i in range(len(rs)):
		print('Text query: ',listText[i])
		for j in range(topk):
			print(rs[i][j])
if __name__ == "__main__":
	# """ Create model and gen sparse feature with pysparnn.cluster_index """
	# dataSparsecharacter , list_character, dataSparseSeries , list_series, dataSparseAuthor, list_author , dict_keyword = genDatabase()

	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/dict_keyword_EN.pickle', 'wb') as handle:
	# 	pickle.dump(dict_keyword, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# # Convert data sparse to CSR_matrix
	# dataSparsecharacterCSR = csr_matrix(dataSparsecharacter)
	# dataSparseSeriesCSR = csr_matrix(dataSparseSeries)
	# listNameAuthor = list(list_author.values())
	# dataSparseAuthorCSR = csr_matrix(dataSparseAuthor[0:10000])
	# searchAuthor = ci.MultiClusterIndex(dataSparseAuthorCSR[:10000], listNameAuthor[:10000])
	# for i in range(10000,len(dataSparseAuthor)):
	# 	vecCSR = csr_matrix(dataSparseAuthor[i])
	# 	searchAuthor.insert(dataSparseAuthorCSR[i],listNameAuthor[i])
	# listNameChararer = list(list_character.values())
	# listNameSeries = list(list_series.values())
	# # Create model search
	# searchcharacter = ci.MultiClusterIndex(dataSparsecharacterCSR, listNameChararer)
	# searchSeries = ci.MultiClusterIndex(dataSparseSeriesCSR, listNameSeries)

		
	# #Save model
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/searchcharacter_EN.pickle', 'wb') as handle:
	# 	pickle.dump(searchcharacter, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/searchSeries_EN.pickle', 'wb') as handle:
	# 	pickle.dump(searchSeries, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/searchAuthor_EN_Test.pickle', 'wb') as handle:
	# 	pickle.dump(searchAuthor, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/listNameChararer.pickle', 'wb') as handle:
	# 	pickle.dump(listNameChararer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/listNameSeries.pickle', 'wb') as handle:
	# 	pickle.dump(listNameSeries, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/searchAuthor.pickle', 'wb') as handle:
	# 	pickle.dump(searchAuthor, handle, protocol=pickle.HIGHEST_PROTOCOL)
	""" Create model and gen tfidf feature with pysparnn.cluster_index """
	featuresVeccharacter, list_character, featuresSeries, list_series, featuresAuthor, list_author, tv = genDataTfidf()
	# searchcharacter = ci.MultiClusterIndex(featuresVeccharacter,list_character)
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/modelGenFeatureTfidf.pickle', 'wb') as handle:
	# 	pickle.dump(tv, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/searchCharacter.pickle', 'wb') as handle:
	# 	pickle.dump(searchcharacter, handle, protocol=pickle.HIGHEST_PROTOCOL)
	searchSeries = ci.MultiClusterIndex(featuresSeries,list_series)
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/searchSeries.pickle', 'wb') as handle:
	# 	pickle.dump(searchSeries, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# searchAuthor = ci.MultiClusterIndex(featuresAuthor,list_author)
	# with open('/media/anlab/data-2tb/ANLAB_THUY/SparseToDenseVector/LashinbangMasterSample/ModelSearch/modelTFIDF/searchAuthor.pickle', 'wb') as handle:
	# 	pickle.dump(searchAuthor, handle, protocol=pickle.HIGHEST_PROTOCOL)
	search_data = [
		'ー',
		'ヴラド三世',
		'bb'
	]
	searchTFIDF(search_data,searchSeries,tv,topk=5)