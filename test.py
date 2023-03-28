from SearchText import SearchText,tokenizerText
if __name__ == "__main__":
	search_data = [
		'ラブライ',
		# 'ヴラド三世',
		# 'ドーターメーカー',
	]
	searchData = SearchText(optionSearch='tfidf')
	print(searchData.searchSeries(search_data))