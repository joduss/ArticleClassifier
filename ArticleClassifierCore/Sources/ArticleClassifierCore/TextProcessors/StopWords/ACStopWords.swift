import Foundation
import NaturalLanguage


/// Holds stop-words for a given language.
public class ACStopWords {
    
    private let stopWords: Set<String>
    
    /// Constructor for the ACStopWords for a specific language.
    public init(language: NLLanguage) {
        
        switch language {
        case NLLanguage.french:
            stopWords = Set<String>(ACStopWordsFR().stopWordsArray)
        default:
            stopWords = Set<String>()
            
        }
    }
    
    /// Indicates if a word is a stop-word.
    public func IsStopWord(word: String) -> Bool {
        return stopWords.contains(word)
    }
}
