import Foundation
import NaturalLanguage


/// A Lemmatizer, supporting additionnal lemmas and that also has the capability of removing stop-words
public class ACLemmatizerBasic {
    
    private let otherLemma: [String : String] = ["apps" : "application",
                                         "app" : "application"]
    
    private let language: NLLanguage
    
    private lazy var stopWords: ACStopWords? = {
        return ACStopWords(language: language)
    }()
    
//    private let expressionAfter: [(String, String)] =
//        [
//            ("l'", ""),
//            ("d'", "appstore"),
//            ("t'", "itunesstore")
//        ]
//        
//    private let expressionsBefore: [(String, String)] =
//        [
//            ("mac app store", "macappstore"),
//            ("app store", "appstore"),
//            ("itunes store", "itunesstore")
//        ]
//    
//    private let regularExpressions: [(String, String)] =
//        [("itunes store", "itunesstore"),
//         ("iphone xs", "iphonexs"),
//         ("iphone xr", "iphonexr"),
//         ("iphone x", "iphonex"),
//         ("pixel 3a", "pixel3a"),
//         ("galaxy s9", "galaxys9"),
//         ("galaxy s10", "galaxys10"),
//         ("macbook pro", "macbookpro"),
//         ("mac pro", "macpro"),
//         ("galaxy s10", "galaxys10")
//        ]
    
    public init(language: NLLanguage) {
        self.language = language
    }
    
    
    /// Lemmatize a text. Can as well remove stop-words.
    /// - Parameters:
    ///   - text: Text to lemmatize
    ///   - removeStopWords: Boolean indicating if stop-words should be removed.
    /// - Returns: Lemmatized text.
    public func lemmatize(text: String, removeStopWords: Bool) -> String {
        
        // To be returned
        var orderedWords = ContiguousArray<String>()
        orderedWords.reserveCapacity(text.count / 5)
        
        // For the processing
        let tagger = NLTagger(tagSchemes: [.lemma])
        var textToProcess = text
        
        // Process expressions.
//        for expression in expressions {
//            textToProcess = textToProcess.replacingOccurrences(of: expression.0, with: expression.1)
//        }
        
        tagger.string = textToProcess
        tagger.enumerateTags(in: textToProcess.startIndex..<textToProcess.endIndex,
                             unit: .word,
                             scheme: .lemma,
                             options: [.omitWhitespace],
                             using: { tag, range in
                                let term = termFromTagOrString(tag: tag, range: range, text: textToProcess)
                                
                                if term.count == 0 {
                                    return true
                                }
                                
                                if removeStopWords, stopWords?.IsStopWord(word: term) ?? false {
                                    return true
                                }
                                
                                orderedWords.append(term)
                                return true
        })
        
        return orderedWords.joined(separator: " ")
    }
    
    /// Returns the value of the tag or if the tag is unknown, returns the original word.
    private func termFromTagOrString(tag: NLTag?, range: Range<String.Index>, text: String) -> String {
        if let tag = tag {
            return tag.rawValue.lowercased()
        }
        
        // Extract the term from the text based on the range.
        // We need to remove the punctuation, because the tagger didn't know this term at all, so
        // he lets the punctuation with it.
        let extractedTerm = String(text[range]).trimmingCharacters(in: CharacterSet.punctuationCharacters).lowercased()
        
        return
            otherLemma[extractedTerm.trimmingCharacters(in: CharacterSet.punctuationCharacters)]
                ?? extractedTerm
    }
    
}
