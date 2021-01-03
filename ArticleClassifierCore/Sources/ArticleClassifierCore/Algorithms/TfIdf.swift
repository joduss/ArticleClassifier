//
//  TfIdf.swift
//  ArticleClassifierCore-macos
//
//  Created by Jonathan Duss on 02.03.19.
//

import Foundation
import JoLibrary
import NaturalLanguage


public class TfIdf {
    
    //======================================================================
    // MARK: - Caching
    
    //Cache for Dictionary of terms associated with the number of document that contains them.
    private var _termsDocumentFrequency: [String: Int] = [:]
    private var _idf: ContiguousArray<Double>?
    private var _allTermsVectorCached: ContiguousArray<String>?
    private let lemmatizer = ACLemmaTokenizer()
    
    public var importantTerms: [String] = []

    //======================================================================
    // MARK: - Properties
    
    ///Dictionary of terms associated with the number of document that contains them.
    private(set) public var termsDocumentFrequency: [String: Int] {
        get {
            if (_termsDocumentFrequency.isEmpty) {
                computeTermAndTextContainingTerm()
            }
            return _termsDocumentFrequency
        }
        set {
            _termsDocumentFrequency = newValue
        }
    }

    private var textsHashCode: [Int]
    private(set) public var texts: [String]
    
    
    
    public var allTermsVector: ContiguousArray<String> {
        if _allTermsVectorCached == nil {
            _allTermsVectorCached = ContiguousArray(termsDocumentFrequency.keys.sorted())
        }
        return _allTermsVectorCached!
    }
    
    public var allTerms: Set<String> {
        return Set(termsDocumentFrequency.keys)
    }
    
    private var termCount: Int {
        return termsDocumentFrequency.count
    }
    
    //======================================================================
    // MARK: - Initialization
    
    
    /// Init a TfIdf object with the given set of texts. This should be a non-exhautive
    /// list of texts. A list of terms will be generated, therefore if not all texts are added,
    /// other texts might have terms that have never been encountered and will be ignored.
    ///
    /// Each document should be lowercased!
    ///
    /// - Parameter texts: The lowercased texts.
    public init(texts: [String]) {
        self.texts = texts.map({$0.lowercased()})
        textsHashCode = texts.map({$0.hashValue})
    }
    
    //======================================================================
    // MARK: - Perform computation for caching
    
    private func computeTermAndTextContainingTerm() {
        guard _termsDocumentFrequency.isEmpty else {
            return
        }
        
        for text in texts {
            for term in self.lemmasIn(text: text) {
                if let textContainingTerm = self._termsDocumentFrequency[term] {
                    self._termsDocumentFrequency[term] = textContainingTerm + 1
                }
                else {
                    self._termsDocumentFrequency[term] = 1;
                }
            }
        }
    }
    
    //======================================================================
    // MARK: - Modification operations on this tfIdf
    
    //======================================================================
    // MARK: - TF IDF

    
    /// Compute the frequency of each known terms
    ///
    /// - Parameter text: <#text description#>
    /// - Returns: <#return value description#>
    public func termFrequencyVector(text: String) -> ContiguousArray<Int> {
        var vector = ContiguousArray<Int>()
        
        let termFreqInText: [String : Int] = termFrequencyInText(text: text)
        
        Performance.measure(title: "append terms") {
            vector = []
            vector.reserveCapacity(termCount / 3)
            for term in allTermsVector {
                vector.append(termFreqInText[term] ?? 0)
            }
        }

        return vector
    }
    
    public func invertedDocumentFrequencyVector() -> ContiguousArray<Double> {
        
        if let idf = self._idf {
            return idf
        }
        
        let numberOfTerms = texts.count
        var idfVector: ContiguousArray<Double> = []
        idfVector.reserveCapacity(termCount)
        
        for term in allTermsVector {
            if importantTerms.contains(term) {
                idfVector.append(1)
                continue
            }
            let termDocumentFrequency = termsDocumentFrequency[term]!
            let idf = log(Double(numberOfTerms) / Double(termDocumentFrequency)) //no risk of division by 0. A term is always included in at least 1 text.
            idfVector.append(idf)
        }
        self._idf = idfVector
        return idfVector
    }
    
    public func tfIdfVector(text: String) ->  ContiguousArray<Double> {
        var tf: ContiguousArray<Int>!
        var idf: ContiguousArray<Double>!
        var results: ContiguousArray<Double>!
        
        Performance.measure(title: "tf") {
            tf = termFrequencyVector(text: text.lowercased())
        }
        Performance.measure(title: "idf") {
            idf = invertedDocumentFrequencyVector()
        }
        Performance.measure(title: "tf*idf") {
            results = tf.HadamarProduct(secondArray: idf)
        }
        return results
    }
    
    //======================================================================
    // MARK: - String Utility
    
    
    /// Returns a dictionary of word along with their frequency. Expect lowercased text!
    public func termFrequencyInText(text: String) -> [String: Int] {
        return lemmatizer.lemmaFrequencies(text: text)
    }
    
    func lemmasIn(text: String) -> Set<String> {
        return Set(lemmatizer.lemmaFrequencies(text: text).keys)
    }
    
    func lemmaVector(from text: String) -> [String] {
        return lemmatizer.lemmaFrequencies(text: text).keys.sorted()
    }
    
    
    
    
}
