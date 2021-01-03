import Foundation
import NaturalLanguage

public class ACTextPreprocessor {
    
    private let language: NLLanguage
    private let lemmatizer: ACLemmatizer! = nil
    
    public init(representativeText: String) {
        let languageRecognizer = NLLanguageRecognizer()
        languageRecognizer.processString(representativeText)
        language = languageRecognizer.dominantLanguage!
    }
    
    public func process(text: String) -> String {
        return lemmatizer.lemmatize(text: text)
    }
}
