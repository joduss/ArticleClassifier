//
//  ACLemmatizer.swift
//  ArticleClassifierCore-macos
//
//  Created by Jonathan Duss on 09.06.19.
//

import Foundation
import NaturalLanguage

///The ACLemmaTokenizer is a special lemmatizer and tokenizer together.
/// It can only return a list of tokens - which are lemmas - associated with their frequency. It cannot return the tokens
/// in the order of appearance in the text.
public class ACLemmaTokenizer {
    
    //List of word that might be followed by a version (such as ios 10)
    private var wordWithVersion: [String] = ["ios", "tvos", "macos", "android", "windows", "iphone", "pixel"]
    
    let otherLemma: [String : String] = ["apps" : "application",
                                         "app" : "application"]
    /// Multi words terms to be merge in one
    let expressions: [(String, String)] = [("mac app store", "macappstore"),
                                         ("app store", "appstore"),
                                         ("itunes store", "itunesstore"),
                                         ("iphone xs", "iphonexs"),
                                         ("iphone xr", "iphonexr"),
                                         ("iphone x", "iphonex"),
                                         ("pixel 3a", "pixel3a"),
                                         ("galaxy s9", "galaxys9"),
                                         ("galaxy s10", "galaxys10"),
                                         ("macbook pro", "macbookpro"),
                                         ("mac pro", "macpro"),
                                         ("galaxy s10", "galaxys10")
    ]
    
    public init() {}

    public func lemmatize(text: String) -> String {
        
        // To be returned
        var orderedWords = ContiguousArray<String>()
        orderedWords.reserveCapacity(text.count / 5)
        
        // For the processing
        let tagger = NLTagger(tagSchemes: [.lemma])
        var textToProcess = text
        
        // Process expressions.
        for expression in expressions {
            textToProcess = textToProcess.replacingOccurrences(of: expression.0, with: expression.1)
        }
        
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
                                
                                orderedWords.append(term)
                                return true
        })
        
        return orderedWords.joined(separator: " ")
    }
    
    /// Compute the frequency of each lemma in a text.
    ///
    /// - Parameters:
    ///   - text: text to be lemmatized
    /// - Returns: a dictionary of lemma with the frequency of each.
    public func lemmaFrequencies(text: String) -> [String: Int] {

        // To be returned
        var tokens: [String: Int] = [:]
        tokens.reserveCapacity(text.count / 3)
        
        // For the processing
        let tagger = NLTagger(tagSchemes: [.lemma])
        var textToProcess = text
        var previous = ""
        
        // Process expressions.
        for expression in expressions {
            textToProcess = textToProcess.replacingOccurrences(of: expression.0, with: expression.1)
        }
        
        tagger.string = textToProcess
        tagger.enumerateTags(in: textToProcess.startIndex..<textToProcess.endIndex,
                             unit: .word,
                             scheme: .lemma,
                             options: [.omitPunctuation, .omitWhitespace],
                             using: { tag, range in
                                let term = termFromTagOrString(tag: tag, range: range, text: textToProcess)
                                
                                if term.count == 0 {
                                    return true
                                }
                                
                                if Double(term) ?? 0 > 0 && wordWithVersion.contains(previous) {
                                    let concatenedTerm = "\(previous) \(term)"
                                    tokens[concatenedTerm] = 1 + (tokens[concatenedTerm] ?? 0)
                                    
                                    let countPrevious = tokens[previous] ?? 1 //1 so it will be eliminated anyway
                                    tokens[previous] = countPrevious - 1
                                    
                                    if (countPrevious == 1) {
                                        tokens.removeValue(forKey: previous)
                                    }
                                }
                                else {
                                    tokens[term] = 1 + (tokens[term] ?? 0)
                                }
                                previous = term
                                return true
        })
        
        return tokens
    }
    
    /// If the tag exists, extract the lemma.
    /// otherwise
    /// extract the term and check if it matches one extra lemma that we supprot
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
