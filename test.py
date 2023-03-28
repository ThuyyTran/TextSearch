from SearchText import SearchText,tokenizerText
if __name__ == "__main__":
	search_data = [
		'ー',
		'ヴラド三世',
		'ドーターメーカー',
	]
	searchData = SearchText(optionSearch='parse')
	print(searchData.searchAuthor(search_data))