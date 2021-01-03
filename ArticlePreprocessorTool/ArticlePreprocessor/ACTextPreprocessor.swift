import Foundation
import NaturalLanguage
import ArticleClassifierCore

public class ACTextPreprocessor {
    
    public let language: NLLanguage
    public var lemmatizer: ACLemmatizer!
    
    public init(representativeText: String) {
        let languageRecognizer = NLLanguageRecognizer()
        languageRecognizer.processString(representativeText)
        language = languageRecognizer.dominantLanguage!
    }
    
    public func process(text: String) -> String {
        return lemmatizer.lemmatize(text: text)
    }
}
