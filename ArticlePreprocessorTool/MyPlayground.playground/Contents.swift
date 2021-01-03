import Cocoa
import ArticlePreprocessorFramework
import NaturalLanguage

var str = "Les produits sont bien. ios 11"

let preprocessor = ACLemmatizerV2(language: .french)

print("Processed:")
print(preprocessor.lemmatize(text: str))



