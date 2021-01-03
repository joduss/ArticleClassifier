//
//  ACLemmatizerV2.swift
//  ArticlePreprocessor
//
//  Created by Jonathan Duss on 03.01.21.
//  Copyright © 2021 ZaJo. All rights reserved.
//

import Foundation
import NaturalLanguage
import ArticleClassifierCore


/// A Lemmatizer, supporting additionnal lemmas and that also has the capability of removing stop-words
public class ACLemmatizerV2: ACLemmatizer {
    
    private let otherLemma: [String : String] = ["apps" : "application",
                                         "app" : "application"]
    
    private static var gazetteer: NLGazetteer?
    
    private let language: NLLanguage
    
    private lazy var stopWords: ACStopWords! = {
        return ACStopWords(language: language)
    }()
    
    
    public init(language: NLLanguage) {
        self.language = language
        ACLemmatizerV2.gazetteer = createGazetteer()
    }
    
    
    /// Lemmatize a text. Can as well remove stop-words.
    /// - Parameters:
    ///   - text: Text to lemmatize
    ///   - removeStopWords: Boolean indicating if stop-words should be removed.
    /// - Returns: Lemmatized text.
    public func lemmatize(text: String) -> String {
                
        // For the processing
        let tagger = NLTagger(tagSchemes: [.lemma, .nameTypeOrLexicalClass])
        //tagger.setGazetteers([createGazetteer()], for: .lemma)
        tagger.setGazetteers([ACLemmatizerV2.gazetteer!], for: .nameTypeOrLexicalClass)

        
        // STEP 1: Aggressive stopwords remover (based on word type)
        var wordsStep1 = ContiguousArray<String>()
        wordsStep1.reserveCapacity(text.count / 5)
        
        tagger.string = text

        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameTypeOrLexicalClass, options: [.omitWhitespace, .omitPunctuation, .joinNames, .omitOther], using: {
            tag, range in
            
//            let term = String(text[range])
            //print("\(term) => \(tag?.rawValue ?? "?")")
                        
            switch (tag) {
            case NLTag.preposition:
                return true
            case NLTag.adverb:
                return true
            case NLTag.conjunction:
                return true
            case NLTag.determiner:
                return true
            default:
                wordsStep1.append(String(text[range]))
                break
            }
            
            return true
        })
        
        let processedTextStep1 = wordsStep1.joined(separator: " ")
        
        // STEP 2: Lemmatization
        
        // To be returned
        var words = ContiguousArray<String>()
        words.reserveCapacity(processedTextStep1.count)
                
        tagger.string = processedTextStep1
        tagger.enumerateTags(in: processedTextStep1.startIndex..<processedTextStep1.endIndex, unit: .word, scheme: .lemma, options: [.omitWhitespace, .joinNames], using: {
            tag, range in

            guard let lemmatizedWord = tag?.rawValue else {
                words.append(String(processedTextStep1[range]))
                return true
            }
            
            guard stopWords.IsStopWord(word: lemmatizedWord) == false else {
                return true
            }

            words.append(lemmatizedWord)
            return true
        })
        
        return words.joined(separator: " ")
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
    
    private func createGazetteer() -> NLGazetteer {
        
        let gazets =
        [
            "iphone": ["iphone"],
            "ipad": ["ipad"],
            "ios": ["ios"],
            "ipados": ["ipados", "ipad os"],
            "iphoneV": iPhoneVersions(),
            "ipadV" : iPadVersions(),
            "iosV" : iOSVersions(),
            "produits": ["les produits", "le produit", "un produit", "des produits"]
        ]
        
        return try! NLGazetteer(dictionary: gazets, language: language)
    }
    
    private func iPhoneVersions() -> [String] {
        let versions = (1...15).map({$0.description}) + ["xr", "xs", "x"]
        
        var alliphoneVersions = [String]()
        
        for version in versions {
            alliphoneVersions.append("iphone \(version)")
            alliphoneVersions.append("iphone \(version) mini")
            alliphoneVersions.append("iphone \(version) pro")
            alliphoneVersions.append("iphone \(version) pro max")
        }
                
        return alliphoneVersions
    }
    
    private func iPadVersions() -> [String] {
        var versions = ["3", "4", "v2", "v3", "v1", "(troisième génération)", "(deuxième génération)", "(première génération)", "(3e génération)", "(2e génération)", "(2nd génération)", "10,2", "10,8", "12,9", "3e génération", "4e génération", "5e génération", "6e génération", "7e génération", "8e génération", "9e génération", "10e génération"]
        versions += (2017...2025).map({$0.description})
        
        
        var alliPadVersions = [String]()
        
        for version in versions {
            alliPadVersions.append("ipad \(version)")
            alliPadVersions.append("ipad mini \(version)")
            alliPadVersions.append("ipad air \(version)")
            alliPadVersions.append("ipad pro \(version)")
        }
        
        return alliPadVersions
    }
    
    private func iOSVersions() -> [String] {
        
        let mainVersions = (4...17)
        let minorVersions = (0...10)
        
        let minorForMainVersionsLimit = [
            4: 3,
            5: 1,
            6: 1,
            7: 1,
            8: 4,
            9: 3,
            10: 3,
            11: 4,
            12: 5,
            13: 7,
        ]

        
        var alliOSVersions = [String]()
        
        for mainVersion in mainVersions {
            alliOSVersions.append("ios \(mainVersion)")
            alliOSVersions.append("ipados \(mainVersion)")
            alliOSVersions.append("ipad os \(mainVersion)")

            for minorVersion in minorVersions {
                
                if let maxMinor = minorForMainVersionsLimit[mainVersion] {
                    if maxMinor > minorVersion {
                        continue
                    }
                }
                
                alliOSVersions.append("ios \(mainVersion).\(minorVersion)")
                alliOSVersions.append("ipados \(mainVersion).\(minorVersion)")
                alliOSVersions.append("ipad os \(mainVersion).\(minorVersion)")
                
                for subminorVersion in minorVersions {
                    alliOSVersions.append("ios \(mainVersion).\(minorVersion).\(subminorVersion)")
                    alliOSVersions.append("ipados \(mainVersion).\(minorVersion).\(subminorVersion)")
                    alliOSVersions.append("ipad os \(mainVersion).\(minorVersion).\(subminorVersion)")
                }
            }
        }
        
        return alliOSVersions
    }
    
//    private func macOSVersions() -> [String] {
//        let mainVersions = (10...15).map({$0.description})
//        let minorVersionsOSX = (0...16).map({$0.description})
//        let subminorVersions = (0...10).map({$0.description})
//
//
//        var allmacOSVersions = [String]()
//
//        for mainVersion in mainVersions {
//            allmacOSVersions.append("macos \(mainVersion)")
//
//            for minorVersion in minorVersions {
//                allmacOSVersions.append("macos \(mainVersion).\(minorVersion)")
//
//                for subminorVersion in subminorVersions {
//                    allmacOSVersions.append("macos \(mainVersion).\(minorVersion).\(subminorVersion)")
//                }
//            }
//        }
//
//        return allmacOSVersions
//    }
}
