import tensorflow.keras as keras

class ManualTestPrint:

    def __init__(self, articles, themes, maxWordCount, model, themeTokenizer):
        self.articles = articles
        self.themes = themes
        self.maxWordCount = maxWordCount
        self.model = model
        self.themeTokenizer = themeTokenizer

    def print(self, percent = 100):
        idx = 0
        articleCount = len(self.articles)

        for article in self.articles:
            print("article ", idx, " themes: ", self.themes[idx], " # predicted ===> ", self.predictionToHumanReadable(self.articles[idx]),
                  " ||||| original article: ", self.articles[idx])
            idx = idx + 1

            if (idx / articleCount * 100 > percent):
                break

    def doPrediction(self, text):
        text = self.preprocessor.process_article(text)
        vector = self.tokenizer.texts_to_sequences([text])
        vector = keras.preprocessing.sequence.pad_sequences(vector,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=self.maxWordCount)

        return self.model.predict(vector)

    def predictionToHumanReadable(self, text):
        predictions = self.doPrediction(text)[0]
        idxWord = self.themeTokenizer.index_word
        return " - ".join(
            "{}({})".format(idxWord[idx], prediction) for idx, prediction in enumerate(predictions, start=1))
